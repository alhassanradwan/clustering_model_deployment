"""Tests that load the real trained model.

Marked `needs_model` because models/model.joblib is tracked by DVC, not git. A
fresh checkout has only the hash, so these are skipped unless `dvc pull` has
run. CI runs them in a separate job that fetches the model first.
"""

from __future__ import annotations

import pytest

from serving import ROOT, Segmenter

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

SEGMENTS = {"champions", "loyal", "slipping", "lapsed", "new or light"}


@pytest.fixture(scope="module")
def segmenter() -> Segmenter:
    """Loaded once: reading the model per test would dominate the runtime."""
    return Segmenter()


def test_the_model_and_scaler_load(segmenter: Segmenter) -> None:
    assert segmenter.model is not None
    assert segmenter.scaler is not None
    # The scaler holds the training set's statistics for all three features.
    assert segmenter.scaler.mean_.shape == (3,)


def test_predict_returns_a_real_cluster_and_name(segmenter: Segmenter) -> None:
    cluster, segment = segmenter.predict(recency=30, frequency=5, monetary=1200)

    assert isinstance(cluster, int)
    assert cluster in set(segmenter.profile["Cluster"])
    assert segment in SEGMENTS


def test_every_cluster_has_a_name(segmenter: Segmenter) -> None:
    assert set(segmenter.names.values()) == SEGMENTS


@pytest.mark.parametrize(
    "recency,frequency,monetary",
    [
        (1, 50, 100_000.0),  # a wholesale buyer
        (400, 1, 5.0),       # long gone, spent almost nothing
        (30, 5, 1200.0),     # unremarkable
    ],
)
def test_predictions_stay_inside_the_known_segments(
    segmenter: Segmenter, recency: int, frequency: int, monetary: float
) -> None:
    """Extremes must still land somewhere real, not raise or invent a name."""
    cluster, segment = segmenter.predict(recency, frequency, monetary)

    assert cluster in set(segmenter.profile["Cluster"])
    assert segment in SEGMENTS


def test_the_same_customer_always_gets_the_same_answer(segmenter: Segmenter) -> None:
    first = segmenter.predict(30, 5, 1200)
    second = segmenter.predict(30, 5, 1200)

    assert first == second


def test_the_scaler_is_actually_applied(segmenter: Segmenter) -> None:
    """Guards the bug that would otherwise be invisible.

    If the scaler were refitted on the incoming row instead of loaded, every
    customer would scale to [0, 0, 0] and land in the same cluster. Two
    customers at opposite extremes must not agree.
    """
    best = segmenter.predict(recency=1, frequency=50, monetary=100_000.0)
    worst = segmenter.predict(recency=400, frequency=1, monetary=5.0)

    assert best[0] != worst[0]


def test_segments_returns_one_row_per_cluster(segmenter: Segmenter) -> None:
    rows = segmenter.segments()

    assert len(rows) == len(segmenter.profile)
    assert all("segment" in row for row in rows)
    assert {row["segment"] for row in rows} == SEGMENTS
