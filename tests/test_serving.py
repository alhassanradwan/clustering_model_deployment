"""Tests for the segment naming logic.

These deliberately need no model, no scaler and no credentials, so they run in
seconds on any machine - including a CI runner that has never seen this
project. Anything that needs the trained model belongs in a separate test that
runs after `dvc pull`.
"""

from __future__ import annotations

import pandas as pd

from serving import name_clusters

# One row per cluster, shaped like reports/cluster_profile.csv. The numbers are
# the real ones from the trained model.
PROFILE = pd.DataFrame(
    [
        {"Cluster": 0, "Recency": 214.4, "Frequency": 1.2, "Monetary": 275.5},
        {"Cluster": 1, "Recency": 13.6, "Frequency": 20.1, "Monetary": 10909.4},
        {"Cluster": 2, "Recency": 96.3, "Frequency": 3.4, "Monetary": 1278.5},
        {"Cluster": 3, "Recency": 30.3, "Frequency": 1.6, "Monetary": 353.3},
        {"Cluster": 4, "Recency": 17.0, "Frequency": 5.7, "Monetary": 1922.2},
    ]
)


def test_names_follow_behaviour() -> None:
    """The biggest spenders are champions, the most distant are lapsed."""
    names = name_clusters(PROFILE)

    assert names[1] == "champions"      # GBP 10,909, 20 orders
    assert names[0] == "lapsed"         # 214 days since last order
    assert names[4] == "loyal"          # orders most often of those left
    assert names[2] == "slipping"       # 96 days and drifting
    assert names[3] == "new or light"   # recent, but only GBP 353


def test_renumbering_clusters_does_not_change_names() -> None:
    """The reason this function exists.

    K-Means numbers clusters arbitrarily, so a retrain can turn cluster 1 into
    cluster 4 without anything about the customers changing. Names must follow
    behaviour, never the number.
    """
    renumbered = PROFILE.copy()
    renumbered["Cluster"] = [4, 3, 0, 1, 2]  # same rows, different ids

    original = name_clusters(PROFILE)
    shuffled = name_clusters(renumbered)

    # The row that was cluster 1 is now cluster 3, and must still be champions.
    assert original[1] == shuffled[3] == "champions"
    assert original[0] == shuffled[4] == "lapsed"
    assert original[4] == shuffled[2] == "loyal"


def test_every_cluster_gets_a_distinct_name() -> None:
    names = name_clusters(PROFILE)

    assert len(names) == len(PROFILE)
    assert len(set(names.values())) == len(names)


def test_fewer_clusters_than_rules() -> None:
    """With k=3 there are more naming rules than clusters, which must not fail."""
    names = name_clusters(PROFILE.head(3))

    assert len(names) == 3
    assert "champions" in names.values()
