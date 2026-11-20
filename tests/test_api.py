"""HTTP-level smoke test for the FastAPI app (no server process needed)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402
from app.samples import SAMPLES  # noqa: E402

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
