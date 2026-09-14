"""Rank every MLflow run, best first.

    uv run python src/compare_runs.py
    uv run python src/compare_runs.py --sort misfit_rate --ascending

Read-only: fetches runs from whatever MLFLOW_TRACKING_URI points at (DagsHub,
per .env) and prints them as a table. Not a DVC stage.

Blank rows are runs where fewer than two clusters were found, so the
separation metrics were skipped - usually a DBSCAN eps that was too large.
"""

from __future__ import annotations

import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

COLUMNS = {
    "tags.mlflow.runName": "run",
    "metrics.n_clusters_found": "clusters",
    "metrics.silhouette": "silhouette",
    "metrics.misfit_rate": "misfit",
    "metrics.stability_ari_mean": "stability",
    "metrics.noise_rate": "noise",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", default="silhouette", help="metric to rank by")
    parser.add_argument("--ascending", action="store_true", help="lowest first")
    args = parser.parse_args()

    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    print(f"tracking uri: {uri}\n")

    runs = mlflow.search_runs(search_all_experiments=True)
    if runs.empty:
        print("no runs found")
        return

    present = {src: name for src, name in COLUMNS.items() if src in runs.columns}
    table = runs[list(present)].rename(columns=present)

    if args.sort in table.columns:
        table = table.sort_values(args.sort, ascending=args.ascending)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(f"\n{len(table)} runs")


if __name__ == "__main__":
    main()
