"""Score new customers with the trained model.

Shared by the API and the dashboard so both apply the exact transform the model
was trained on: log1p, then the saved scaler, then predict.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["Recency", "Frequency", "Monetary"]


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their average behaviour.

    K-Means numbers clusters arbitrarily and a retrain can renumber them, so the
    names are derived from the profile rather than hardcoded.
    """
    remaining = profile.set_index("Cluster")
    names: dict[int, str] = {}

    rules = [
        ("champions", "Monetary", "max"),  # biggest spenders
        ("lapsed", "Recency", "max"),      # longest since last order
        ("loyal", "Frequency", "max"),     # order most often
        ("slipping", "Recency", "max"),    # drifting away
    ]
    for name, column, how in rules:
        if remaining.empty:
            break
        cluster = int(getattr(remaining[column], f"idx{how}")())
        names[cluster] = name
        remaining = remaining.drop(cluster)

    for cluster in remaining.index:
        names[int(cluster)] = "new or light"

    return names


class Segmenter:
    def __init__(self, root: Path = ROOT) -> None:
        self.model = joblib.load(root / "models" / "model.joblib")
        self.scaler = joblib.load(root / "models" / "scaler.joblib")
        self.names = name_clusters(pd.read_csv(root / "reports" / "cluster_profile.csv"))

    def predict(self, recency: float, frequency: float, monetary: float) -> tuple[int, str]:
        row = pd.DataFrame([[recency, frequency, monetary]], columns=COLUMNS)
        # Keep the column names: the model was fitted on a named DataFrame and
        # warns on every call when handed a bare array.
        scaled = pd.DataFrame(self.scaler.transform(np.log1p(row)), columns=COLUMNS)
        cluster = int(self.model.predict(scaled)[0])
        return cluster, self.names.get(cluster, f"cluster {cluster}")
