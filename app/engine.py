"""Memory-lean shared-embedding continuation-loss engine (CPU, float64).

Design contract
---------------
* ``W`` is a ``vocab x 16`` standard-normal float64 matrix generated from a
  seed and multiplied by a user scale in [0.1, 100].
* The same ``W`` is the input embedding (row lookup) and the output projection
  (``logits = h @ W.T``), i.e. weight tying.
* Loss is the mean cross entropy over *all valid* targets (``-100`` ignored),
  and exactly one SGD step with lr = 0.01 is applied.
* The loss is implemented as a custom ``torch.autograd.Function``.  Its
  projection temporaries never exceed a vocab axis of 128, and the forward
  pass keeps no scores or probabilities for backward: only the hidden states,
  the target ids and the valid-row indices (plus the shared parameter itself)
  cross the forward/backward boundary.  Everything chunk-wise is recomputed.
* An independent dense reference (full ``[tokens, vocab]`` logits, no limits)
  provides the ground truth for loss, gradient and post-update weights.
* Mini-batch splitting is mathematically neutral; the engine verifies it
  empirically for splits into 1..4 micro-batches.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import torch

IGNORE_INDEX = -100
EMBED_DIM = 16
MIN_VOCAB = 256
MAX_VOCAB = 4096
MAX_TOTAL_TOKENS = 128
MIN_TOTAL_TOKENS = 1
MAX_MICROBATCHES = 4
MAX_VOCAB_CHUNK = 128  # hard cap on every projection temp's vocab axis
LEARNING_RATE = 0.01
MIN_SCALE = 0.1
MAX_SCALE = 100.0
TOLERANCE = 1e-8
MAX_SEED = 2**63 - 1


class ValidationError(ValueError):
    """Raised for input that violates the documented contract."""


@dataclass(frozen=True)
class MicroBatch:
    name: str
    input_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]


@dataclass
class RunSpec:
    microbatches: List[MicroBatch]
    vocab_size: int
    seed: int
    scale: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{name} 必须是整数，不能是布尔值")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValidationError(f"{name} 必须是整数")
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10)
        except ValueError:
            raise ValidationError(f"{name} 必须是整数") from None
    raise ValidationError(f"{name} 必须是整数")


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} 必须是数字，不能是布尔值")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValidationError(f"{name} 必须是数字") from None
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} 必须是数字")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} 必须是有限数字")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def parse_payload(payload: dict) -> RunSpec:
    """Parse and validate the JSON payload sent by the web form."""
    _require(isinstance(payload, dict), "请求体必须是 JSON 对象")
    allowed = {"seed", "scale", "vocab_size", "microbatches", "tokens"}
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

    raw_batches = payload.get("microbatches")
    if raw_batches is None and "tokens" in payload:
        raw_batches = [{"name": "mb1", "tokens": payload["tokens"]}]
    _require(
        isinstance(raw_batches, list) and len(raw_batches) >= 1,
        "至少需要 1 个微批（microbatches 不能为空）",
    )
    _require(
        len(raw_batches) <= MAX_MICROBATCHES,
        f"微批数量不能超过 {MAX_MICROBATCHES} 个",
    )

    batches: List[MicroBatch] = []
    total = 0
    for idx, raw in enumerate(raw_batches):
        _require(isinstance(raw, dict), f"第 {idx + 1} 个微批必须是对象")
        name = str(raw.get("name") or f"mb{idx + 1}").strip() or f"mb{idx + 1}"
        pairs = raw.get("tokens")
        _require(
            isinstance(pairs, list) and len(pairs) >= 1,
            f"微批 {name!r} 至少需要一组 [输入号, 目标号]",
        )
        inputs: List[int] = []
        targets: List[int] = []
        for pair in pairs:
            _require(
                isinstance(pair, (list, tuple)) and len(pair) == 2,
                f"微批 {name!r}: 每一项必须是 [输入号, 目标号]",
            )
            inp = _as_int("输入 token 号", pair[0])
            tgt = _as_int("目标 token 号", pair[1])
            _require(
                0 <= inp < vocab_size,
                f"微批 {name!r}: 输入号 {inp} 超出词表 [0, {vocab_size})",
            )
            _require(
                tgt == IGNORE_INDEX or 0 <= tgt < vocab_size,
                f"微批 {name!r}: 目标号 {tgt} 必须为 {IGNORE_INDEX} "
                f"或位于 [0, {vocab_size})",
            )
            inputs.append(inp)
            targets.append(tgt)
        total += len(inputs)
        _require(
            total <= MAX_TOTAL_TOKENS,
            f"全部微批的 token 总数（含忽略项）不能超过 {MAX_TOTAL_TOKENS}",
        )
        batches.append(
            MicroBatch(name=name, input_ids=tuple(inputs), target_ids=tuple(targets))
        )

    _require(total >= MIN_TOTAL_TOKENS, "至少需要 1 个 token")
    _require(
        any(t != IGNORE_INDEX for b in batches for t in b.target_ids),
        "所有目标都被 -100 忽略：损失无定义，已拒绝本次运行，权重不做任何更新",
    )
    return RunSpec(microbatches=batches, vocab_size=vocab_size, seed=seed, scale=scale)


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------
def make_weight(vocab_size: int, seed: int, scale: float) -> torch.Tensor:
    """Standard-normal float64 ``W`` from ``seed``, multiplied by ``scale``."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    weight = torch.randn(vocab_size, EMBED_DIM, generator=gen, dtype=torch.float64)
    return weight * scale


def _chunk_ranges(vocab_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, vocab_size, MAX_VOCAB_CHUNK):
        yield start, min(start + MAX_VOCAB_CHUNK, vocab_size)


# ---------------------------------------------------------------------------
# Chunked numerics helpers (all temps have a vocab axis <= 128)
# ---------------------------------------------------------------------------
def _token_logsumexp(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-token log-sum-exp over the full vocabulary, chunk by chunk."""
    lse = torch.full((hidden.shape[0],), -math.inf, dtype=torch.float64)
    for start, end in _chunk_ranges(weight.shape[0]):
        logits = hidden @ weight[start:end].t()  # [N, <=128]
        lse = torch.logaddexp(lse, torch.logsumexp(logits, dim=1))
        del logits
    return lse


def _target_logits(
    hidden: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Logits of the target ids, computed chunk by chunk."""
    n = target.numel()
    picked = torch.empty(n, dtype=torch.float64)
    for start, end in _chunk_ranges(weight.shape[0]):
        sel = (target >= start) & (target < end)
        if not bool(sel.any()):
            continue
        rows = torch.nonzero(sel, as_tuple=False).flatten()
        logits = hidden.index_select(0, rows) @ weight[start:end].t()
        cols = target.index_select(0, rows) - start
        picked[rows] = logits.gather(1, cols.unsqueeze(1)).squeeze(1)
        del logits
    return picked


# Thread-local record of what each forward pass hands to save_for_backward, so
# the API can report the true saved shapes without any global mutable state.
_SAVED_SHAPES = threading.local()


def _record_saved(tensors: Sequence[torch.Tensor]) -> None:
    shapes = getattr(_SAVED_SHAPES, "value", None)
    if shapes is not None:
        shapes.append([list(t.shape) for t in tensors])


class ChunkedSoftmaxLoss(torch.autograd.Function):
    """Mean cross entropy with recompute-in-backward projection.

    Forward saves ``(W, hidden, target, valid_idx)`` only.  It does *not*
    cache per-chunk logits or probabilities.  Backward recomputes chunk logits
    from the saved hidden states and W, normalises them with a recomputed
    global log-sum-exp, and accumulates ``dL/dW`` and ``dL/dH`` chunk-wise.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        valid_idx = torch.nonzero(target != IGNORE_INDEX, as_tuple=False).flatten()
        n_valid = int(valid_idx.numel())
        if n_valid == 0:
            raise ValidationError(
                "该微批所有目标都被 -100 忽略：损失无定义，"
                "已拒绝运行，权重不做任何更新"
            )

        h = hidden.index_select(0, valid_idx)
        tgt = target.index_select(0, valid_idx)

        lse = _token_logsumexp(h, weight)
        target_logit = _target_logits(h, tgt, weight)
        loss = (lse - target_logit).mean()

        ctx.save_for_backward(weight, hidden, target, valid_idx)
        ctx.n_valid = n_valid
        _record_saved((weight, hidden, target, valid_idx))
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, hidden, target, valid_idx = ctx.saved_tensors
        n_valid = ctx.n_valid

        h = hidden.index_select(0, valid_idx)
        tgt = target.index_select(0, valid_idx)

        # Recompute the global normaliser: nothing chunk-wise was cached.
        lse = _token_logsumexp(h, weight)
        scale = grad_output / n_valid

        grad_weight = torch.zeros_like(weight)
        grad_h = torch.zeros_like(h)

        for start, end in _chunk_ranges(weight.shape[0]):
            logits = h @ weight[start:end].t()  # [N, <=128]
            probs = torch.exp(logits - lse.unsqueeze(1))
            sel = (tgt >= start) & (tgt < end)
            if bool(sel.any()):
                rows = torch.nonzero(sel, as_tuple=False).flatten()
                cols = tgt.index_select(0, rows) - start
                probs[rows, cols] -= 1.0
            probs.mul_(scale)
            grad_weight[start:end] += probs.t() @ h
            grad_h += probs @ weight[start:end]
            del logits, probs

        grad_hidden = torch.zeros_like(hidden)
        # index_add_ (not copy) so repeated input ids accumulate their
        # gradients exactly like autograd's index_select.
        grad_hidden.index_add_(0, valid_idx, grad_h)
        return grad_weight, grad_hidden, None


# ---------------------------------------------------------------------------
# Independent dense reference (no memory restrictions)
# ---------------------------------------------------------------------------
def dense_reference(
    weight: torch.Tensor,
    input_ids: Sequence[int],
    target_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().clone().requires_grad_(True)
    idx = torch.tensor(input_ids, dtype=torch.long)
    tgt = torch.tensor(target_ids, dtype=torch.long)
    valid = tgt != IGNORE_INDEX

    hidden = w.index_select(0, idx)  # input embedding
    logits = hidden @ w.t()  # [T, V] full dense scores
    log_probs = torch.log_softmax(logits, dim=1)
    picked = log_probs.gather(1, tgt.clamp_min(0).unsqueeze(1)).squeeze(1)
    loss = -picked[valid].mean()
    loss.backward()
    return loss.detach(), w.grad.detach().clone()


# ---------------------------------------------------------------------------
# Chunked run
# ---------------------------------------------------------------------------
@dataclass
class ChunkedResult:
    loss: float
    grad: torch.Tensor
    updated_weight: torch.Tensor
    base_weight: torch.Tensor
    valid_tokens: int
    token_count: int
    saved_shapes: List[List[list]] = field(default_factory=list)


def run_chunked(spec: RunSpec) -> ChunkedResult:
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)

    valid_counts = [
        sum(1 for t in mb.target_ids if t != IGNORE_INDEX) for mb in spec.microbatches
    ]
    n_valid_total = sum(valid_counts)
    if n_valid_total == 0:
        raise ValidationError("所有目标都被 -100 忽略：拒绝运行，权重不做任何更新")

    loss = torch.zeros((), dtype=torch.float64)
    previous = getattr(_SAVED_SHAPES, "value", None)
    _SAVED_SHAPES.value = []
    try:
        for mb, n_valid in zip(spec.microbatches, valid_counts):
            if n_valid == 0:
                continue
            idx = torch.tensor(mb.input_ids, dtype=torch.long)
            tgt = torch.tensor(mb.target_ids, dtype=torch.long)
            hidden = weight.index_select(0, idx)
            batch_loss = ChunkedSoftmaxLoss.apply(weight, hidden, tgt)
            # Weight each per-micro-batch mean by its share of valid tokens so
            # the sum is exactly the global mean over all valid tokens.
            loss = loss + batch_loss * (n_valid / n_valid_total)
        saved_shapes = _SAVED_SHAPES.value
    finally:
        _SAVED_SHAPES.value = previous

    loss.backward()
    grad = weight.grad.detach().clone()
    with torch.no_grad():
        updated = weight.detach().clone()
        updated -= LEARNING_RATE * grad

    return ChunkedResult(
        loss=float(loss.item()),
        grad=grad,
        updated_weight=updated,
        base_weight=weight.detach().clone(),
        valid_tokens=n_valid_total,
        token_count=sum(len(mb.input_ids) for mb in spec.microbatches),
        saved_shapes=saved_shapes,
    )


# ---------------------------------------------------------------------------
# Helpers for split experiments
# ---------------------------------------------------------------------------
def _flatten(spec: RunSpec) -> Tuple[List[int], List[int]]:
    inputs: List[int] = []
    targets: List[int] = []
    for mb in spec.microbatches:
        inputs.extend(mb.input_ids)
        targets.extend(mb.target_ids)
    return inputs, targets


def _even_groups(n: int, parts: int) -> List[List[int]]:
    base, extra = divmod(n, parts)
    groups: List[List[int]] = []
    cursor = 0
    for p in range(parts):
        size = base + (1 if p < extra else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _split_spec(spec: RunSpec, parts: int) -> RunSpec:
    inputs, targets = _flatten(spec)
    groups = _even_groups(len(inputs), parts)
    batches = [
        MicroBatch(
            name=f"split{parts}-{i + 1}",
            input_ids=tuple(inputs[j] for j in group),
            target_ids=tuple(targets[j] for j in group),
        )
        for i, group in enumerate(groups)
    ]
    return RunSpec(
        microbatches=batches,
        vocab_size=spec.vocab_size,
        seed=spec.seed,
        scale=spec.scale,
    )


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


# ---------------------------------------------------------------------------
# Full comparison report
# ---------------------------------------------------------------------------
def evaluate(payload: dict) -> dict:
    spec = parse_payload(payload)
    all_inputs, all_targets = _flatten(spec)

    # Ground truth: dense scores, one backward, no chunking.
    ref_weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    ref_loss, ref_grad = dense_reference(ref_weight, all_inputs, all_targets)
    ref_updated = ref_weight - LEARNING_RATE * ref_grad

    # System under test: 1..4 user micro-batches, custom autograd loss.
    user = run_chunked(spec)
    err_loss = abs(user.loss - float(ref_loss.item()))
    err_grad = _max_abs_diff(user.grad, ref_grad)
    err_weight = _max_abs_diff(user.updated_weight, ref_updated)

    # Mini-batch splitting invariance: re-split the same token stream into
    # every micro-batch count from 1..min(4, tokens).
    split_runs = []
    max_split_loss = 0.0
    max_split_grad = 0.0
    max_split_weight = 0.0
    max_split_vs_dense_loss = 0.0
    max_split_vs_dense_grad = 0.0
    max_split_vs_dense_weight = 0.0
    max_parts = min(MAX_MICROBATCHES, len(all_inputs))
    for parts in range(1, max_parts + 1):
        candidate = run_chunked(_split_spec(spec, parts))
        d_loss = abs(candidate.loss - user.loss)
        d_grad = _max_abs_diff(candidate.grad, user.grad)
        d_weight = _max_abs_diff(candidate.updated_weight, user.updated_weight)
        max_split_loss = max(max_split_loss, d_loss)
        max_split_grad = max(max_split_grad, d_grad)
        max_split_weight = max(max_split_weight, d_weight)
        max_split_vs_dense_loss = max(
            max_split_vs_dense_loss, abs(candidate.loss - float(ref_loss.item()))
        )
        max_split_vs_dense_grad = max(
            max_split_vs_dense_grad, _max_abs_diff(candidate.grad, ref_grad)
        )
        max_split_vs_dense_weight = max(
            max_split_vs_dense_weight, _max_abs_diff(candidate.updated_weight, ref_updated)
        )
        split_runs.append(
            {
                "microbatches": parts,
                "valid_tokens": candidate.valid_tokens,
                "token_count": candidate.token_count,
                "loss": candidate.loss,
                "loss_diff_vs_user": d_loss,
                "grad_diff_vs_user": d_grad,
                "weight_diff_vs_user": d_weight,
            }
        )

    checks = {
        "loss": err_loss <= TOLERANCE,
        "grad": err_grad <= TOLERANCE,
        "updated_weight": err_weight <= TOLERANCE,
        "split_loss": max_split_loss <= TOLERANCE,
        "split_grad": max_split_grad <= TOLERANCE,
        "split_weight": max_split_weight <= TOLERANCE,
    }
    passed = all(checks.values())

    return {
        "ok": passed,
        "tolerance": TOLERANCE,
        "message": (
            "通过：三项对照误差与拆分不变量均 <= 1e-8"
            if passed
            else "未通过：至少一项误差超过 1e-8"
        ),
        "config": {
            "vocab_size": spec.vocab_size,
            "embed_dim": EMBED_DIM,
            "seed": spec.seed,
            "scale": spec.scale,
            "learning_rate": LEARNING_RATE,
            "microbatches": len(spec.microbatches),
            "total_tokens": len(all_inputs),
            "valid_tokens": user.valid_tokens,
        },
        "chunked": {
            "loss": user.loss,
            "grad_norm": float(user.grad.norm().item()),
            "weight_norm": float(user.updated_weight.norm().item()),
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
        "split": {
            "max_loss_diff": max_split_loss,
            "max_grad_diff": max_split_grad,
            "max_weight_diff": max_split_weight,
            "max_vs_dense": {
                "loss": max_split_vs_dense_loss,
                "grad": max_split_vs_dense_grad,
                "weight": max_split_vs_dense_weight,
            },
            "runs": split_runs,
        },
        "checks": checks,
        "memory": {
            "max_vocab_chunk": MAX_VOCAB_CHUNK,
            "saved_shapes_per_microbatch": user.saved_shapes,
            "saved_tensor_roles": [
                "shared_weight_parameter",
                "hidden_states",
                "target_ids",
                "valid_row_indices",
            ],
            "scores_or_probabilities_saved": False,
            "recompute_in_backward": True,
        },
    }
