"""Sanity checks for the memory-lean shared-embedding engine.

Run with:  python -m pytest tests -q      (or python tests/test_engine.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import (  # noqa: E402
    IGNORE_INDEX,
    MAX_VOCAB_CHUNK,
    TOLERANCE,
    ChunkedSoftmaxLoss,
    ValidationError,
    evaluate,
    make_weight,
    parse_payload,
    run_chunked,
)


def _payload(tokens, seed=7, scale=1.0, vocab=256):
    return {
        "seed": seed,
        "scale": scale,
        "vocab_size": vocab,
        "microbatches": [{"name": "mb", "tokens": [list(p) for p in tokens]}],
    }


def test_deterministic_weight_from_seed():
    a = make_weight(256, 1234, 1.0)
    b = make_weight(256, 1234, 1.0)
    c = make_weight(256, 1235, 1.0)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.dtype == torch.float64 and a.shape == (256, 16)
    scaled = make_weight(256, 1234, 2.5)
    assert torch.allclose(scaled, a * 2.5, rtol=0, atol=0)


def test_weight_tying_is_shared():
    spec = parse_payload(_payload([[3, 5], [7, 9]]))
    res = run_chunked(spec)
    # Gradient must include the input-embedding contribution as well as the
    # output-projection contribution.  Compare against a manual autograd graph
    # that ties the same tensor in both roles.
    w = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    idx = torch.tensor([3, 7])
    tgt = torch.tensor([5, 9])
    hidden = w.index_select(0, idx)
    logits = hidden @ w.t()
    loss = -torch.log_softmax(logits, dim=1).gather(1, tgt[:, None]).mean()
    loss.backward()
    assert abs(float(loss.detach()) - res.loss) < 1e-12
    assert float((w.grad - res.grad).abs().max()) < 1e-12


def test_matches_dense_reference_with_ignored_targets():
    tokens = [[1, 2], [3, IGNORE_INDEX], [4, 5], [1, 6]]
    report = evaluate(_payload(tokens))
    assert report["ok"], report["errors"]
    assert report["errors"]["loss_abs_diff"] <= TOLERANCE
    assert report["errors"]["grad_max_abs_diff"] <= TOLERANCE
    assert report["errors"]["weight_max_abs_diff"] <= TOLERANCE


def test_microbatch_split_invariance():
    tokens = [[1, 2], [3, 4], [5, IGNORE_INDEX], [7, 8], [2, 2], [9, 10], [11, 12]]
    report = evaluate(_payload(tokens, seed=99, scale=3.0, vocab=512))
    assert report["ok"]
    assert report["split"]["max_loss_diff"] <= TOLERANCE
    assert report["split"]["max_grad_diff"] <= TOLERANCE
    assert report["split"]["max_weight_diff"] <= TOLERANCE
    assert len(report["split"]["runs"]) == 4


def test_sgd_update_matches_manual_step():
    spec = parse_payload(_payload([[1, 2], [3, 4], [5, 6]], seed=11, scale=0.7))
    res = run_chunked(spec)
    expected = res.base_weight - 0.01 * res.grad
    assert float((res.updated_weight - expected).abs().max()) == 0.0


def test_all_ignored_rejected_without_update():
    with pytest.raises(ValidationError):
        parse_payload(_payload([[1, IGNORE_INDEX], [2, IGNORE_INDEX]]))
    # Also via the Function directly: no update should be possible.
    w = make_weight(256, 1, 1.0).requires_grad_(True)
    hidden = w.index_select(0, torch.tensor([1, 2]))
    with pytest.raises(ValidationError):
        ChunkedSoftmaxLoss.apply(w, hidden, torch.tensor([IGNORE_INDEX, IGNORE_INDEX]))
    assert w.grad is None


def test_boundaries_and_validation():
    parse_payload({"seed": 0, "scale": 0.1, "vocab_size": 256,
                   "microbatches": [{"tokens": [[0, 255]]}]})
    parse_payload({"seed": 2**63 - 1, "scale": 100.0, "vocab_size": 4096,
                   "microbatches": [{"tokens": [[4095, 0]]}]})
    bad = [
        {"seed": 0, "scale": 0.09, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 100.1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 255,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 4097,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[256, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 256]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}] * 5},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]] * 129}]},
        {"seed": 0, "scale": 1, "vocab_size": 256, "microbatches": []},
        {"seed": 0, "scale": float("inf"), "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
    ]
    for payload in bad:
        with pytest.raises(ValidationError):
            parse_payload(payload)
    with pytest.raises(ValidationError):
        parse_payload({"seed": 0, "scale": 1, "vocab_size": 256,
                       "microbatches": [{"tokens": [[1, 2]]}], "bogus": 1})
    with pytest.raises(ValidationError):
        parse_payload({"seed": "abc", "scale": 1, "vocab_size": 256,
                       "microbatches": [{"tokens": [[1, 2]]}]})
    with pytest.raises(ValidationError):
        parse_payload({"seed": 1.5, "scale": 1, "vocab_size": 256,
                       "microbatches": [{"tokens": [[1, 2]]}]})


def test_token_budget_counts_whole_payload():
    # 4 micro-batches x 32 tokens = 128, allowed; 129 rejected.
    mbs = [{"name": str(i), "tokens": [[i, i + 1]] * 32} for i in range(4)]
    parse_payload({"seed": 0, "scale": 1, "vocab_size": 4096, "microbatches": mbs})
    mbs[3]["tokens"] = mbs[3]["tokens"] + [[0, 1]]
    with pytest.raises(ValidationError):
        parse_payload({"seed": 0, "scale": 1, "vocab_size": 4096, "microbatches": mbs})


@pytest.mark.parametrize("scale", [0.1, 1.0, 25.0, 100.0])
def test_large_scores_remain_accurate(scale):
    tokens = [[1, 4095], [2048, 0], [4095, 4095], [17, 33], [999, IGNORE_INDEX]]
    report = evaluate(_payload(tokens, seed=424242, scale=scale, vocab=4096))
    assert report["ok"], report["errors"]
    for key in ("loss_abs_diff", "grad_max_abs_diff", "weight_max_abs_diff"):
        assert report["errors"][key] <= TOLERANCE


def test_duplicate_inputs_accumulate_gradients():
    tokens = [[4, 4], [4, 4], [4, 7], [9, 4]]
    report = evaluate(_payload(tokens, seed=3, vocab=300))
    assert report["ok"]
    # Gradient rows for ids 4 and 9 must be non-trivially populated.
    spec = parse_payload(_payload(tokens, seed=3, vocab=300))
    res = run_chunked(spec)
    assert float(res.grad[4].abs().sum()) > 0
    assert float(res.grad[9].abs().sum()) > 0


def test_no_scores_or_probabilities_saved_across_forward_backward():
    """Record every tensor handed to save_for_backward during the run."""
    saved = []

    def pack(tensor):
        saved.append(tensor.detach().clone())
        return tensor

    def unpack(tensor):
        return tensor

    spec = parse_payload(_payload([[1, 2], [3, 4], [5, 6]], vocab=1024))
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    hidden = weight.index_select(0, torch.tensor([1, 3, 5]))
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = ChunkedSoftmaxLoss.apply(weight, hidden, torch.tensor([2, 4, 6]))
    loss.backward()

    assert saved, "save_for_backward was not called"
    shape_set = {tuple(t.shape) for t in saved}
    # Only W (V x 16), hidden states (T x 16), targets (T), valid idx (T).
    assert (1024, 16) in shape_set
    assert (3, 16) in shape_set
    for tensor in saved:
        assert tensor.numel() <= weight.numel()
        assert tensor.dim() <= 2
        if tensor.dim() == 2:
            assert tensor.shape[1] == 16 or tensor.shape[0] == 1024

    spec2 = parse_payload(_payload([[1, 2], [3, 4], [5, 6]], vocab=4096))
    res = run_chunked(spec2)
    for shapes in res.saved_shapes:
        # (W, hidden, target, valid_idx) for each micro-batch.
        assert shapes[0] == [4096, 16]
        assert shapes[1][1] == 16
        assert shapes[2][0] == shapes[3][0]


def test_gradient_matches_finite_differences():
    spec = parse_payload(
        {
            "seed": 12,
            "scale": 1.3,
            "vocab_size": 256,
            "microbatches": [
                {"tokens": [[3, 4], [3, 4], [200, 201], [7, IGNORE_INDEX]]},
                {"tokens": [[9, 3], [3, 3]]},
            ],
        }
    )
    res = run_chunked(spec)
    weight = make_weight(256, 12, 1.3)
    inputs = [3, 3, 200, 7, 9, 3]
    targets = [4, 4, 201, IGNORE_INDEX, 3, 3]

    def loss_at(w):
        ww = w.detach().clone().requires_grad_(True)
        idx = torch.tensor(inputs)
        tgt = torch.tensor(targets)
        valid = tgt != IGNORE_INDEX
        hidden = ww.index_select(0, idx)
        logits = hidden @ ww.t()
        picked = torch.log_softmax(logits, dim=1).gather(
            1, tgt.clamp_min(0)[:, None]
        ).squeeze(1)
        return -picked[valid].mean().detach()

    eps = 1e-6
    for i, j in [(3, 0), (4, 5), (200, 3), (9, 9)]:
        plus = weight.clone()
        minus = weight.clone()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric = float((loss_at(plus) - loss_at(minus)) / (2 * eps))
        analytic = float(res.grad[i, j])
        assert abs(numeric - analytic) <= 1e-7, (i, j, numeric, analytic)


def test_chunk_axis_never_exceeds_128(monkeypatch):
    import app.engine as engine

    widths = []
    real_range = engine._chunk_ranges

    def spy(vocab_size):
        for start, end in real_range(vocab_size):
            widths.append(end - start)
            yield start, end

    monkeypatch.setattr(engine, "_chunk_ranges", spy)
    spec = parse_payload(_payload([[1, 2], [3, 4]], vocab=4096))
    run_chunked(spec)
    assert widths and max(widths) <= MAX_VOCAB_CHUNK


def test_numeric_loss_equals_manual_formula():
    spec = parse_payload(_payload([[0, 1], [2, IGNORE_INDEX], [3, 3]], seed=5, scale=2.0))
    res = run_chunked(spec)
    w = make_weight(spec.vocab_size, spec.seed, spec.scale)
    hidden = w[[0, 2, 3]]
    logits = hidden @ w.t()
    expected = -torch.log_softmax(logits, dim=1)[[0, 2], [1, 3]].mean()
    assert abs(float(expected) - res.loss) < 1e-12
    assert math.isfinite(res.loss)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
