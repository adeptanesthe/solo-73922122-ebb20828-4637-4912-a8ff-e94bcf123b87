"""HTTP-level smoke test for the FastAPI app (no server process needed)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402
from app.samples import DISTILL_SAMPLES, SAMPLES  # noqa: E402

client = TestClient(app)


def test_index_and_health():
    assert client.get("/healthz").json() == {"status": "ok"}
    html = client.get("/")
    assert html.status_code == 200
    assert "共享词嵌入" in html.text
    assert client.get("/static/index.html").status_code == 200


def test_samples_endpoint_and_each_sample_passes():
    body = client.get("/api/samples").json()
    assert len(body["samples"]) == len(SAMPLES) == 3
    for sample in body["samples"]:
        resp = client.post("/api/run", json=sample["payload"])
        assert resp.status_code == 200, (sample["id"], resp.text)
        report = resp.json()
        assert report["ok"], (sample["id"], report["errors"], report["split"])
        assert report["errors"]["loss_abs_diff"] <= 1e-8
        assert report["errors"]["grad_max_abs_diff"] <= 1e-8
        assert report["errors"]["weight_max_abs_diff"] <= 1e-8


def test_invalid_requests_return_422():
    cases = [
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, -100]]}]},
        {"seed": 0, "scale": 200, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 128,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}] * 5},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]] * 129}]},
    ]
    for payload in cases:
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["detail"]


def test_flat_tokens_shorthand():
    resp = client.post("/api/run", json={
        "seed": 1, "scale": 1, "vocab_size": 256, "tokens": [[1, 2], [3, -100]],
    })
    assert resp.status_code == 200
    assert resp.json()["ok"]


def test_distill_samples_endpoint_and_each_sample_passes():
    body = client.get("/api/distill/samples").json()
    assert len(body["samples"]) == len(DISTILL_SAMPLES) == 4
    for sample in body["samples"]:
        resp = client.post("/api/distill/run", json=sample["payload"])
        assert resp.status_code == 200, (sample["id"], resp.text)
        report = resp.json()
        assert report["ok"], (sample["id"], report["errors"])
        assert report["errors"]["loss_abs_diff"] <= 1e-8
        assert report["errors"]["grad_max_abs_diff"] <= 1e-8
        assert report["errors"]["weight_max_abs_diff"] <= 1e-8
        assert report["memory"]["max_vocab_chunk"] == 128
        assert report["memory"]["scores_or_probabilities_saved"] is False
        assert report["memory"]["recompute_in_backward"] is True


def test_distill_invalid_requests_return_422():
    good = {"prefix": 1, "listed": [[2, 0.5]], "rest": 0.5}
    cases = [
        {"seed": 0, "scale": 1, "vocab_size": 256, "samples": []},
        {"seed": 0, "scale": 1, "vocab_size": 256, "samples": [good] * 33},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 1, "listed": [], "rest": 1.0}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 1, "listed": [[k, 0.125] for k in range(9)],
                      "rest": 0.0}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 1, "listed": [[2, 0.4], [2, 0.6]], "rest": 0.0}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 1, "listed": [[2, 0.6]], "rest": 0.5}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 1, "listed": [[2, 0.5]], "rest": -0.1}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 300, "listed": [[2, 0.5]], "rest": 0.5}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "samples": [{"prefix": 1, "listed": [[2, 0.5]], "rest": 0.5}],
         "bogus": 1},
    ]
    for payload in cases:
        resp = client.post("/api/distill/run", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["detail"]


def test_distill_run_ok_shape():
    resp = client.post("/api/distill/run", json={
        "seed": 3, "scale": 1.0, "vocab_size": 256,
        "samples": [{"prefix": 4, "listed": [[4, 0.5], [9, 0.25]], "rest": 0.25}],
    })
    assert resp.status_code == 200
    report = resp.json()
    assert report["ok"]
    assert set(report["checks"]) == {"loss", "grad", "updated_weight"}
    assert all(report["checks"].values())
