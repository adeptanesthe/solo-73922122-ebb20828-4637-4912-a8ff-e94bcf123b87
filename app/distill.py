"""Top-k + rest-class teacher distillation check (CPU, float64).

The teacher distribution is given only as 1..8 listed token probabilities
plus the *total* probability of every unlisted token (the "rest class").
The student reuses the shared ``W`` of the hard-label engine both as input
embedding (row lookup of the prefix id) and as output projection
(``logits = h @ W.T``) and is normalised over the **full** vocabulary.

Unlisted tokens are grouped into one aggregate class.  The loss is the
sample-averaged cross entropy between teacher and student *class*
probabilities, followed by exactly one SGD step with the original learning
rate (``LEARNING_RATE``).

Math (per sample, logits ``z``, ``lse = logsumexp(z)``,
``lse_R = logsumexp(z)`` over unlisted tokens only)::

    loss  = sum_i p_i * (lse - z_i) + r * (lse - lse_R)
    d loss / d z_j = s_j - p_j            (j listed)
                   = s_j - r * q_j        (j unlisted)

with ``s_j = exp(z_j - lse)`` the full-vocabulary student probability and
``q_j = exp(z_j - lse_R)`` the student probability renormalised inside the
rest class.  Both normalisers are formed with ``logsumexp`` so the rest
class stays accurate even when ``scale = 100`` makes ``s_R`` itself
underflow to 0: the loss uses ``log s_R = lse_R - lse`` directly and the
gradient uses ``q_j`` instead of the naive ``s_j / s_R`` (which would be
0/0).  A dense reference computes the same quantities without any chunking
restriction; both must agree to <= 1e-8.

Memory contract (same as the hard-label engine): every projection temporary
has a vocab axis of at most ``MAX_VOCAB_CHUNK`` (128), the forward pass
caches no scores or probabilities, and the backward pass recomputes them
chunk by chunk from the saved operands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from .engine import (
    EMBED_DIM,
    LEARNING_RATE,
    MAX_SCALE,
    MAX_SEED,
    MAX_VOCAB,
    MAX_VOCAB_CHUNK,
    MIN_SCALE,
    MIN_VOCAB,
    TOLERANCE,
    _as_float,
    _as_int,
    _chunk_ranges,
    _record_saved,
    _require,
    _SAVED_SHAPES,
    _token_logsumexp,
    make_weight,
)

MIN_SAMPLES = 1
MAX_SAMPLES = 32
MIN_LISTED = 1
MAX_LISTED = 8
PROB_SUM_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Spec & validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistillSample:
    prefix: int
    listed_ids: Tuple[int, ...]
    listed_probs: Tuple[float, ...]
    rest_prob: float


@dataclass
class DistillSpec:
    samples: List[DistillSample]
    vocab_size: int
    seed: int
    scale: float


def _as_probability(name: str, value: object) -> float:
    prob = _as_float(name, value)
    _require(prob >= 0.0, f"{name} 必须非负，当前为 {prob}")
    return prob


def parse_distill_payload(payload: dict) -> DistillSpec:
    """Parse and validate the JSON payload sent by the distillation form."""
    _require(isinstance(payload, dict), "请求体必须是 JSON 对象")
    allowed = {"seed", "scale", "vocab_size", "samples"}
    unknown = sorted(set(payload) - allowed)
    _require(not unknown, f"未知字段：{', '.join(unknown)}")

    seed = _as_int("seed", payload.get("seed", 0))
    _require(0 <= seed <= MAX_SEED, f"seed 必须在 [0, {MAX_SEED}] 内")

    scale = _as_float("scale", payload.get("scale", 1.0))
    _require(
        MIN_SCALE <= scale <= MAX_SCALE,
        f"权重倍率 scale 必须在 [{MIN_SCALE}, {MAX_SCALE}] 内",
    )

    vocab_size = _as_int("vocab_size", payload.get("vocab_size", MIN_VOCAB))
    _require(
        MIN_VOCAB <= vocab_size <= MAX_VOCAB,
        f"词表大小 vocab_size 必须在 [{MIN_VOCAB}, {MAX_VOCAB}] 内",
    )

    raw_samples = payload.get("samples")
    _require(
        isinstance(raw_samples, list) and len(raw_samples) >= MIN_SAMPLES,
        f"至少需要 {MIN_SAMPLES} 条样本",
    )
    _require(
        len(raw_samples) <= MAX_SAMPLES,
        f"样本数量不能超过 {MAX_SAMPLES} 条",
    )

    samples: List[DistillSample] = []
    for idx, raw in enumerate(raw_samples):
        label = f"第 {idx + 1} 条样本"
        _require(isinstance(raw, dict), f"{label} 必须是对象")
        unknown_keys = sorted(set(raw) - {"prefix", "listed", "rest"})
        _require(not unknown_keys, f"{label} 含未知字段：{', '.join(unknown_keys)}")

        prefix = _as_int(f"{label} 的前缀词号", raw.get("prefix"))
        _require(
            0 <= prefix < vocab_size,
            f"{label} 的前缀词号 {prefix} 超出词表 [0, {vocab_size})",
        )

        listed = raw.get("listed")
        _require(
            isinstance(listed, list) and MIN_LISTED <= len(listed) <= MAX_LISTED,
            f"{label} 需要 {MIN_LISTED}–{MAX_LISTED} 个列出词",
        )
        ids: List[int] = []
        probs: List[float] = []
        for entry in listed:
            _require(
                isinstance(entry, (list, tuple)) and len(entry) == 2,
                f"{label} 的列出项必须是 [词号, 概率]",
            )
            token = _as_int(f"{label} 的列出词号", entry[0])
            _require(
                0 <= token < vocab_size,
                f"{label} 的列出词号 {token} 超出词表 [0, {vocab_size})",
            )
            _require(
                token not in ids,
                f"{label} 的列出词号 {token} 重复（同一样本内词号必须不同）",
            )
            ids.append(token)
            probs.append(_as_probability(f"{label} 列出词 {token} 的教师概率", entry[1]))

        rest = _as_probability(f"{label} 的其余词总概率", raw.get("rest"))
        total = sum(probs) + rest
        _require(
            abs(total - 1.0) <= PROB_SUM_TOLERANCE,
            f"{label} 的教师概率合计必须为 1（列出词概率之和 + 其余词总概率），"
            f"当前为 {total!r}",
        )
        samples.append(
            DistillSample(
                prefix=prefix,
                listed_ids=tuple(ids),
                listed_probs=tuple(probs),
                rest_prob=rest,
            )
        )
    return DistillSpec(samples=samples, vocab_size=vocab_size, seed=seed, scale=scale)


def _padded(spec: DistillSpec) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad the ragged listed ids/probs to ``[N, MAX_LISTED]`` tensors.

    Padding uses id ``-1`` and probability ``0`` so padded slots never match a
    vocab chunk and contribute exactly zero to the loss.
    """
    n = len(spec.samples)
    listed_ids = torch.full((n, MAX_LISTED), -1, dtype=torch.long)
    listed_probs = torch.zeros(n, MAX_LISTED, dtype=torch.float64)
    rest = torch.zeros(n, dtype=torch.float64)
    for row, sample in enumerate(spec.samples):
        k = len(sample.listed_ids)
        listed_ids[row, :k] = torch.tensor(sample.listed_ids, dtype=torch.long)
        listed_probs[row, :k] = torch.tensor(sample.listed_probs, dtype=torch.float64)
        rest[row] = sample.rest_prob
    return listed_ids, listed_probs, rest


# ---------------------------------------------------------------------------
# Chunked numerics helpers (all temps have a vocab axis <= 128)
# ---------------------------------------------------------------------------
def _rest_logsumexp(
    hidden: torch.Tensor, weight: torch.Tensor, listed_ids: torch.Tensor
) -> torch.Tensor:
    """Per-sample log-sum-exp over *unlisted* tokens only, chunk by chunk."""
    n = hidden.shape[0]
    acc = torch.full((n,), -math.inf, dtype=torch.float64)
    for start, end in _chunk_ranges(weight.shape[0]):
        logits = hidden @ weight[start:end].t()  # [N, <=128]
        sel = (listed_ids >= start) & (listed_ids < end)
        if bool(sel.any()):
            rows, cols = torch.nonzero(sel, as_tuple=True)
            logits[rows, listed_ids[rows, cols] - start] = -math.inf
        acc = torch.logaddexp(acc, torch.logsumexp(logits, dim=1))
        del logits
    return acc


def _listed_logits(
    hidden: torch.Tensor, weight: torch.Tensor, listed_ids: torch.Tensor
) -> torch.Tensor:
    """Logits of the listed ids (padded slots stay 0), gathered chunk by chunk."""
    n, k = listed_ids.shape
    picked = torch.zeros(n, k, dtype=torch.float64)
    for start, end in _chunk_ranges(weight.shape[0]):
        sel = (listed_ids >= start) & (listed_ids < end)
        if not bool(sel.any()):
            continue
        logits = hidden @ weight[start:end].t()  # [N, <=128]
        rows, cols = torch.nonzero(sel, as_tuple=True)
        picked[rows, cols] = logits[rows, listed_ids[rows, cols] - start]
        del logits
    return picked


class ChunkedDistillLoss(torch.autograd.Function):
    """Sample-averaged top-k + rest-class distillation cross entropy.

    Forward saves ``(W, hidden, listed_ids, listed_probs, rest_probs)`` only;
    it never caches per-chunk scores or probabilities.  Backward recomputes
    both normalisers and the class-posterior gradients chunk by chunk.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        listed_ids: torch.Tensor,
        listed_probs: torch.Tensor,
        rest_probs: torch.Tensor,
    ) -> torch.Tensor:
        lse = _token_logsumexp(hidden, weight)
        lse_rest = _rest_logsumexp(hidden, weight, listed_ids)
        z_listed = _listed_logits(hidden, weight, listed_ids)
        per_sample = (listed_probs * (lse.unsqueeze(1) - z_listed)).sum(dim=1)
        per_sample = per_sample + rest_probs * (lse - lse_rest)
        loss = per_sample.mean()

        ctx.save_for_backward(weight, hidden, listed_ids, listed_probs, rest_probs)
        _record_saved((weight, hidden, listed_ids, listed_probs, rest_probs))
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, hidden, listed_ids, listed_probs, rest_probs = ctx.saved_tensors
        n = hidden.shape[0]

        # Recompute both normalisers: nothing chunk-wise was cached.
        lse = _token_logsumexp(hidden, weight)
        lse_rest = _rest_logsumexp(hidden, weight, listed_ids)
        factor = grad_output / n

        grad_weight = torch.zeros_like(weight)
        grad_hidden = torch.zeros_like(hidden)
        for start, end in _chunk_ranges(weight.shape[0]):
            logits = hidden @ weight[start:end].t()  # [N, <=128]
            s_full = torch.exp(logits - lse.unsqueeze(1))  # student probs, chunk
            q_rest = torch.exp(logits - lse_rest.unsqueeze(1))  # within-rest probs
            sel = (listed_ids >= start) & (listed_ids < end)
            if bool(sel.any()):
                rows, cols = torch.nonzero(sel, as_tuple=True)
                local = listed_ids[rows, cols] - start
                q_rest[rows, local] = 0.0  # listed tokens are not in the rest class
                s_full[rows, local] -= listed_probs[rows, cols]
            grad_logits = s_full - rest_probs.unsqueeze(1) * q_rest
            grad_logits.mul_(factor)
            grad_weight[start:end] += grad_logits.t() @ hidden
            grad_hidden += grad_logits @ weight[start:end]
            del logits, s_full, q_rest, grad_logits
        return grad_weight, grad_hidden, None, None, None


# ---------------------------------------------------------------------------
# Independent dense reference (no memory restrictions)
# ---------------------------------------------------------------------------
def dense_distill_reference(
    weight: torch.Tensor, spec: DistillSpec
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense ground truth: full ``[N, V]`` logits, autograd, no chunking."""
    w = weight.detach().clone().requires_grad_(True)
    idx = torch.tensor([s.prefix for s in spec.samples], dtype=torch.long)
    hidden = w.index_select(0, idx)
    logits = hidden @ w.t()  # [N, V] full dense scores
    lse = torch.logsumexp(logits, dim=1)

    total = torch.zeros((), dtype=torch.float64)
    for row, sample in enumerate(spec.samples):
        ids = torch.tensor(sample.listed_ids, dtype=torch.long)
        mask = torch.ones(spec.vocab_size, dtype=torch.bool)
        mask[ids] = False
        # log of the rest-class student probability, computed as a logsumexp
        # so it stays finite even when the probability itself underflows.
        lse_rest = torch.logsumexp(logits[row][mask], dim=0)
        probs = torch.tensor(sample.listed_probs, dtype=torch.float64)
        total = total + (probs * (lse[row] - logits[row][ids])).sum()
        total = total + sample.rest_prob * (lse[row] - lse_rest)
    loss = total / len(spec.samples)
    loss.backward()
    return loss.detach(), w.grad.detach().clone()


# ---------------------------------------------------------------------------
# Chunked run
# ---------------------------------------------------------------------------
@dataclass
class DistillResult:
    loss: float
    grad: torch.Tensor
    updated_weight: torch.Tensor
    base_weight: torch.Tensor
    sample_count: int
    saved_shapes: List[List[list]] = field(default_factory=list)


def run_distill_chunked(spec: DistillSpec) -> DistillResult:
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    idx = torch.tensor([s.prefix for s in spec.samples], dtype=torch.long)
    listed_ids, listed_probs, rest = _padded(spec)

    previous = getattr(_SAVED_SHAPES, "value", None)
    _SAVED_SHAPES.value = []
    try:
        hidden = weight.index_select(0, idx)
        loss = ChunkedDistillLoss.apply(weight, hidden, listed_ids, listed_probs, rest)
        saved_shapes = _SAVED_SHAPES.value
    finally:
        _SAVED_SHAPES.value = previous

    loss.backward()
    grad = weight.grad.detach().clone()
    with torch.no_grad():
        updated = weight.detach().clone()
        updated -= LEARNING_RATE * grad

    return DistillResult(
        loss=float(loss.item()),
        grad=grad,
        updated_weight=updated,
        base_weight=weight.detach().clone(),
        sample_count=len(spec.samples),
        saved_shapes=saved_shapes,
    )


# ---------------------------------------------------------------------------
# Full comparison report
# ---------------------------------------------------------------------------
def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def evaluate_distill(payload: dict) -> dict:
    spec = parse_distill_payload(payload)

    # Ground truth: dense scores, one backward, no chunking.
    ref_weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    ref_loss, ref_grad = dense_distill_reference(ref_weight, spec)
    ref_updated = ref_weight - LEARNING_RATE * ref_grad

    # System under test: custom autograd loss, chunked projection, recompute.
    result = run_distill_chunked(spec)
    err_loss = abs(result.loss - float(ref_loss.item()))
    err_grad = _max_abs_diff(result.grad, ref_grad)
    err_weight = _max_abs_diff(result.updated_weight, ref_updated)

    checks = {
        "loss": err_loss <= TOLERANCE,
        "grad": err_grad <= TOLERANCE,
        "updated_weight": err_weight <= TOLERANCE,
    }
    passed = all(checks.values())

    return {
        "ok": passed,
        "tolerance": TOLERANCE,
        "message": (
            "通过：损失 / 共享 W 梯度 / 更新后权重三项对照误差均 <= 1e-8"
            if passed
            else "未通过：至少一项误差超过 1e-8"
        ),
        "config": {
            "vocab_size": spec.vocab_size,
            "embed_dim": EMBED_DIM,
            "seed": spec.seed,
            "scale": spec.scale,
            "learning_rate": LEARNING_RATE,
            "samples": len(spec.samples),
            "listed_total": sum(len(s.listed_ids) for s in spec.samples),
        },
        "chunked": {
            "loss": result.loss,
            "grad_norm": float(result.grad.norm().item()),
            "weight_norm": float(result.updated_weight.norm().item()),
        },
        "dense": {
            "loss": float(ref_loss.item()),
            "grad_norm": float(ref_grad.norm().item()),
            "weight_norm": float(ref_updated.norm().item()),
        },
        "errors": {
            "loss_abs_diff": err_loss,
            "grad_max_abs_diff": err_grad,
            "weight_max_abs_diff": err_weight,
        },
        "checks": checks,
        "memory": {
            "max_vocab_chunk": MAX_VOCAB_CHUNK,
            "saved_shapes": result.saved_shapes,
            "saved_tensor_roles": [
                "shared_weight_parameter",
                "hidden_states",
                "listed_token_ids",
                "listed_teacher_probs",
                "rest_class_probs",
            ],
            "scores_or_probabilities_saved": False,
            "recompute_in_backward": True,
        },
    }
