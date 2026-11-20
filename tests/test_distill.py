"""Checks for the top-k + rest-class distillation engine.

Run with:  python -m pytest tests -q      (or python tests/test_distill.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.distill import (  # noqa: E402
    MAX_LISTED,
    MAX_SAMPLES,
    ChunkedDistillLoss,
    dense_distill_reference,
    evaluate_distill,
    parse_distill_payload,
    run_distill_chunked,
    _padded,
)
from app.engine import (  # noqa: E402
    LEARNING_RATE,
    MAX_VOCAB_CHUNK,
    TOLERANCE,
    ValidationError,
    make_weight,
)


def _payload(samples, seed=7, scale=1.0, vocab=256):
    return {
        "seed": seed,
        "scale": scale,
        "vocab_size": vocab,
        "samples": samples,
    }


def _sample(prefix, listed, rest):
    return {"prefix": prefix, "listed": [list(p) for p in listed], "rest": rest}


BASIC = _payload(
    [
        _sample(5, [[5, 0.6], [9, 0.1]], 0.3),
        _sample(5, [[7, 0.25], [8, 0.25], [9, 0.25]], 0.25),  # duplicate prefix
        _sample(200, [[0, 0.5]], 0.5),
        _sample(17, [[k, 0.1] for k in range(1, 9)], 0.2),
    ]
)


def test_deterministic_spec_parsing():
    a = parse_distill_payload(BASIC)
    b = parse_distill_payload(BASIC)
    assert a.samples == b.samples
    assert len(a.samples) == 4
    assert a.samples[0].listed_ids == (5, 9)
    assert a.samples[0].listed_probs == (0.6, 0.1)
    assert a.samples[0].rest_prob == 0.3


def test_weight_tying_is_shared_with_manual_graph():
    """Gradient must include input-embedding and output-projection roles."""
    spec = parse_distill_payload(BASIC)
    res = run_distill_chunked(spec)

    w = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    idx = torch.tensor([s.prefix for s in spec.samples])
    hidden = w.index_select(0, idx)
    logits = hidden @ w.t()
    lse = torch.logsumexp(logits, dim=1)
    total = torch.zeros((), dtype=torch.float64)
    for row, s in enumerate(spec.samples):
        ids = torch.tensor(s.listed_ids)
        mask = torch.ones(spec.vocab_size, dtype=torch.bool)
        mask[ids] = False
        lse_rest = torch.logsumexp(logits[row][mask], dim=0)
        probs = torch.tensor(s.listed_probs, dtype=torch.float64)
        total = total + (probs * (lse[row] - logits[row][ids])).sum()
        total = total + s.rest_prob * (lse[row] - lse_rest)
    loss = total / len(spec.samples)
    loss.backward()
    assert abs(float(loss.detach()) - res.loss) < 1e-12
    assert float((w.grad - res.grad).abs().max()) < 1e-12


def test_matches_dense_reference_basic():
    report = evaluate_distill(BASIC)
    assert report["ok"], report["errors"]
    for key in ("loss_abs_diff", "grad_max_abs_diff", "weight_max_abs_diff"):
        assert report["errors"][key] <= TOLERANCE


@pytest.mark.parametrize("scale", [0.1, 1.0, 25.0, 100.0])
def test_large_scores_remain_accurate(scale):
    samples = [
        _sample(1, [[4095, 0.7], [0, 0.1]], 0.2),
        _sample(2048, [[1, 0.25], [2, 0.25], [3, 0.25]], 0.25),
        _sample(4095, [[4095, 0.5]], 0.5),
        _sample(17, [[33, 0.4], [4000, 0.1], [17, 0.1]], 0.4),
    ]
    report = evaluate_distill(_payload(samples, seed=424242, scale=scale, vocab=4096))
    assert report["ok"], report["errors"]
    for key in ("loss_abs_diff", "grad_max_abs_diff", "weight_max_abs_diff"):
        assert report["errors"][key] <= TOLERANCE


def test_rest_class_accurate_when_listed_token_dominates():
    """scale=100 with the argmax token listed: the rest-class probability
    underflows to 0 in plain exp, yet its loss/gradient contribution must
    still match the dense reference to <= 1e-8."""
    vocab, seed, scale = 4096, 424242, 100.0
    w = make_weight(vocab, seed, scale)
    prefix = 7
    logits = w[prefix] @ w.t()
    top = int(logits.argmax())
    samples = [
        _sample(prefix, [[top, 0.9], [123, 0.05]], 0.05),
        _sample(3000, [[1, 0.3], [2, 0.3]], 0.4),
    ]
    payload = _payload(samples, seed=seed, scale=scale, vocab=vocab)
    report = evaluate_distill(payload)
    assert report["ok"], report["errors"]
    assert math.isfinite(report["chunked"]["loss"])

    # The naive rest-class computation really is broken here, which is why
    # the engine forms log s_rest as a logsumexp instead.
    naive_rest = 1.0 - float(torch.softmax(w[prefix] @ w.t(), dim=0)[[top, 123]].sum())
    assert naive_rest == 0.0  # exp underflow: log(0) would be -inf
    # ... while the true rest-class log-probability is finite:
    spec = parse_distill_payload(payload)
    assert spec.samples[0].rest_prob == 0.05
    for key in ("loss_abs_diff", "grad_max_abs_diff", "weight_max_abs_diff"):
        assert report["errors"][key] <= TOLERANCE


def test_rest_mass_zero_and_full():
    samples = [
        _sample(3, [[3, 0.7], [4, 0.3]], 0.0),   # teacher fully listed
        _sample(9, [[10, 0.0], [11, 0.0]], 1.0),  # all mass on the rest class
        _sample(9, [[250, 0.45]], 0.55),          # duplicate prefix again
    ]
    report = evaluate_distill(_payload(samples, seed=991, scale=0.5, vocab=300))
    assert report["ok"], report["errors"]


def test_single_sample_and_max_samples():
    one = evaluate_distill(_payload([_sample(0, [[1, 1.0]], 0.0)], vocab=256))
    assert one["ok"], one["errors"]

    many = [
        _sample((13 * i + 1) % 512,
                [[(29 * i + 3) % 512, 0.5], [(53 * i + 7) % 512, 0.25]],
                0.25)
        for i in range(MAX_SAMPLES)
    ]
    report = evaluate_distill(_payload(many, seed=7, scale=2.0, vocab=512))
    assert report["ok"], report["errors"]
    assert report["config"]["samples"] == MAX_SAMPLES


def test_sgd_update_matches_manual_step():
    spec = parse_distill_payload(BASIC)
    res = run_distill_chunked(spec)
    expected = res.base_weight - LEARNING_RATE * res.grad
    assert float((res.updated_weight - expected).abs().max()) == 0.0


def test_duplicate_prefix_rows_accumulate_gradient():
    """Two samples sharing a prefix must both contribute to that row's grad."""
    solo = parse_distill_payload(_payload([_sample(5, [[6, 1.0]], 0.0)], seed=3))
    both = parse_distill_payload(
        _payload(
            [_sample(5, [[6, 1.0]], 0.0), _sample(5, [[7, 1.0]], 0.0)], seed=3
        )
    )
    g_solo = run_distill_chunked(solo).grad
    g_both = run_distill_chunked(both).grad
    # The shared prefix row differs once a second sample uses the same row.
    assert float((g_both[5] - g_solo[5]).abs().max()) > 0
    # And it equals the mean of the two per-sample contributions.
    only_second = parse_distill_payload(_payload([_sample(5, [[7, 1.0]], 0.0)], seed=3))
    g_second = run_distill_chunked(only_second).grad
    assert float((g_both[5] - 0.5 * (g_solo[5] + g_second[5])).abs().max()) < 1e-12


def test_gradient_matches_finite_differences():
    payload = _payload(
        [
            _sample(3, [[4, 0.5], [200, 0.25]], 0.25),
            _sample(9, [[3, 0.4], [3 - 3, 0.1], [250, 0.2]], 0.3),
            _sample(200, [[201, 0.8]], 0.2),
        ],
        seed=12,
        scale=1.3,
    )
    spec = parse_distill_payload(payload)
    res = run_distill_chunked(spec)
    weight = make_weight(256, 12, 1.3)

    def loss_at(w):
        ww = w.detach().clone().requires_grad_(True)
        ref_loss, _ = dense_distill_reference(ww, spec)
        return float(ref_loss)

    eps = 1e-6
    for i, j in [(3, 0), (4, 5), (200, 3), (9, 9), (250, 15)]:
        plus = weight.clone()
        minus = weight.clone()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric = (loss_at(plus) - loss_at(minus)) / (2 * eps)
        analytic = float(res.grad[i, j])
        assert abs(numeric - analytic) <= 1e-7, (i, j, numeric, analytic)


def test_chunk_axis_never_exceeds_128(monkeypatch):
    import app.distill as distill
    import app.engine as engine

    widths = []
    real_range = engine._chunk_ranges

    def spy(vocab_size):
        for start, end in real_range(vocab_size):
            widths.append(end - start)
            yield start, end

    monkeypatch.setattr(distill, "_chunk_ranges", spy)
    spec = parse_distill_payload(
        _payload([_sample(1, [[2, 0.6], [4095, 0.4]], 0.0)], vocab=4096)
    )
    run_distill_chunked(spec)
    assert widths and max(widths) <= MAX_VOCAB_CHUNK


def test_no_scores_or_probabilities_saved_across_forward_backward():
    saved = []

    def pack(tensor):
        saved.append(tensor.detach().clone())
        return tensor

    def unpack(tensor):
        return tensor

    spec = parse_distill_payload(
        _payload(
            [_sample(1, [[2, 0.6], [3, 0.4]], 0.0), _sample(4, [[5, 1.0]], 0.0)],
            vocab=1024,
        )
    )
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    listed_ids, listed_probs, rest = _padded(spec)
    hidden = weight.index_select(0, torch.tensor([1, 4]))
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = ChunkedDistillLoss.apply(weight, hidden, listed_ids, listed_probs, rest)
    loss.backward()

    assert saved, "save_for_backward was not called"
    shape_set = {tuple(t.shape) for t in saved}
    # Only W (V x 16), hidden (N x 16), listed ids/probs (N x 8), rest (N,).
    assert (1024, 16) in shape_set
    assert (2, 16) in shape_set
    assert (2, MAX_LISTED) in shape_set
    assert (2,) in shape_set
    for tensor in saved:
        assert tensor.numel() <= weight.numel()
        assert tensor.dim() <= 2
        if tensor.dim() == 2:
            assert tensor.shape[1] in (16, MAX_LISTED)

    res = run_distill_chunked(spec)
    assert len(res.saved_shapes) == 1
    shapes = res.saved_shapes[0]
    assert shapes[0] == [1024, 16]
    assert shapes[1] == [2, 16]
    assert shapes[2] == [2, MAX_LISTED]
    assert shapes[3] == [2, MAX_LISTED]
    assert shapes[4] == [2]


def test_validation_boundaries():
    # Valid extremes.
    parse_distill_payload(_payload([_sample(0, [[255, 1.0]], 0.0)],
                                   seed=0, scale=0.1, vocab=256))
    parse_distill_payload(_payload([_sample(4095, [[0, 0.5], [4095, 0.5]], 0.0)],
                                   seed=2**63 - 1, scale=100.0, vocab=4096))
    parse_distill_payload(_payload([_sample(1, [[k, 0.125] for k in range(8)], 0.0)]))

    good = _sample(1, [[2, 0.5]], 0.5)
    bad_samples = [
        [],                                            # no samples
        [good] * (MAX_SAMPLES + 1),                    # too many samples
        [_sample(1, [], 1.0)],                         # no listed tokens
        [_sample(1, [[k, 0.125] for k in range(9)], 0.0)],  # too many listed
        [_sample(1, [[2, 0.5], [2, 0.5]], 0.0)],       # duplicate listed id
        [_sample(1, [[2, 0.6]], 0.5)],                 # sum != 1
        [_sample(1, [[2, 1.2]], 0.0)],                 # sum > 1 (rest would be < 0)
        [_sample(1, [[2, -0.1]], 1.1)],                # negative listed prob
        [_sample(1, [[2, 0.5]], -0.1)],                # negative rest
        [_sample(256, [[2, 0.5]], 0.5)],               # prefix out of range
        [_sample(1, [[256, 0.5]], 0.5)],               # listed id out of range
        [_sample(1, [[2, float("nan")]], 0.5)],        # NaN prob
        [_sample(1, [[2, float("inf")]], 0.0)],        # inf prob
        [_sample(1.5, [[2, 0.5]], 0.5)],               # non-integer prefix
        [_sample(True, [[2, 0.5]], 0.5)],              # bool prefix
        [{"prefix": 1, "listed": [[2, 0.5]], "rest": 0.5, "bogus": 1}],
        ["not-a-dict"],
    ]
    for samples in bad_samples:
        with pytest.raises(ValidationError):
            parse_distill_payload(_payload(samples))

    with pytest.raises(ValidationError):
        parse_distill_payload({"seed": 0, "scale": 1, "vocab_size": 256,
                               "samples": [good], "bogus": 1})
    with pytest.raises(ValidationError):
        parse_distill_payload({"seed": 0, "scale": 1, "vocab_size": 256})
    with pytest.raises(ValidationError):
        parse_distill_payload(_payload([good], scale=100.1))
    with pytest.raises(ValidationError):
        parse_distill_payload(_payload([good], vocab=255))


def test_probability_sum_tolerance():
    # 0.1 + 0.2 style rounding is accepted within 1e-9...
    ok = _sample(1, [[2, 0.1], [3, 0.2]], 0.7)
    parse_distill_payload(_payload([ok]))
    # ...but a genuinely wrong total is rejected.
    off = _sample(1, [[2, 0.1], [3, 0.2]], 0.700001)
    with pytest.raises(ValidationError):
        parse_distill_payload(_payload([off]))


def test_listed_token_may_equal_prefix():
    report = evaluate_distill(_payload([_sample(5, [[5, 0.9], [6, 0.1]], 0.0)]))
    assert report["ok"], report["errors"]


def test_padded_layout():
    spec = parse_distill_payload(
        _payload([_sample(1, [[7, 0.5]], 0.5), _sample(2, [[3, 0.3], [4, 0.7]], 0.0)])
    )
    listed_ids, listed_probs, rest = _padded(spec)
    assert listed_ids.shape == (2, MAX_LISTED)
    assert listed_ids[0, 0].item() == 7 and listed_ids[0, 1].item() == -1
    assert listed_ids[1, 1].item() == 4
    assert listed_probs[0, 0].item() == 0.5 and listed_probs[0, 1].item() == 0.0
    assert rest.tolist() == [0.5, 0.0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
