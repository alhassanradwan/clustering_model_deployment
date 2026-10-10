"""Tests for the HTTP layer.

Marked `needs_model` because importing `api` constructs a Segmenter at module
level, which reads models/model.joblib - tracked by DVC, not git.

These cover what the Python-level tests cannot: status codes, JSON shape, and
the validation Pydantic applies before any of our code runs.
"""

from __future__ import annotations

import pytest

from serving import ROOT

REQUIRED = [
    ROOT / "models" / "model.joblib",
    ROOT / "models" / "scaler.joblib",
    ROOT / "reports" / "cluster_profile.csv",
]

pytestmark = [
    pytest.mark.needs_model,
    pytest.mark.skipif(
        not all(p.exists() for p in REQUIRED),
        reason="model artifacts missing - run `uv run dvc pull`",
    ),
]


@pytest.fixture(scope="module")
def client():
    # Imported inside the fixture so collection does not fail when the model is
    # absent and the whole module is being skipped.
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


def test_health_reports_status_version_and_replica(client) -> None:
    """Docker, Compose, Swarm and all three Kubernetes probes call this."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "version", "replica"}
    assert body["replica"]  # the hostname, so you can tell replicas apart


def test_segments_returns_every_segment(client) -> None:
    response = client.get("/segments")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert {row["segment"] for row in body} == {
        "champions", "loyal", "slipping", "lapsed", "new or light"
    }


def test_predict_returns_a_cluster_and_a_segment(client) -> None:
    response = client.post(
        "/predict", json={"recency": 30, "frequency": 5, "monetary": 1200}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"cluster", "segment"}
    assert isinstance(body["cluster"], int)
    assert isinstance(body["segment"], str)


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"recency": -5, "frequency": 5, "monetary": 100}, "recency below zero"),
        ({"recency": 30, "frequency": 0, "monetary": 100}, "frequency below one"),
        ({"recency": 30, "frequency": 5, "monetary": 0}, "monetary not positive"),
        ({"recency": 30, "frequency": 5}, "monetary missing"),
        ({"recency": "soon", "frequency": 5, "monetary": 100}, "recency not a number"),
    ],
)
def test_bad_input_is_rejected_before_it_reaches_the_model(
    client, payload: dict, reason: str
) -> None:
    """Pydantic's Field constraints, not hand-written checks."""
    response = client.post("/predict", json=payload)

    assert response.status_code == 422, reason


def test_the_docs_page_is_served(client) -> None:
    """/docs is generated from the type hints - it is how the API documents
    itself, and the page a reviewer is most likely to open."""
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_unknown_paths_404(client) -> None:
    assert client.get("/").status_code == 404
