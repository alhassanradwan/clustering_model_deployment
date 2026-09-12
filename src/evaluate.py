"""DVC stage 3: score the fitted model and profile the segments.

    uv run python src/evaluate.py

Writes metrics.json (tracked by `dvc metrics show`), the per-cluster profile,
and the customer table with its cluster label. Also logs the run to MLflow.

Inertia alone cannot tell you whether a clustering is real - it always falls as
k rises. The metrics here answer three separate questions: are the clusters
separated (silhouette, Davies-Bouldin, Calinski-Harabasz), does every cluster
hold together (per-cluster silhouette, misfit rate), and would you get the same
grouping again (ARI across random seeds).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import hydra
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

log = logging.getLogger(__name__)

# MLflow prints run links containing emoji. The Windows console defaults to
# cp1252, which cannot encode them, and the resulting UnicodeEncodeError kills
# the stage after the run has already been logged.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# Credentials and the tracking URI live in .env, which is gitignored.
load_dotenv()


def _resolve_tracking_uri(cfg: DictConfig, root: str) -> str:
    """Pick the MLflow tracking backend.

    MLFLOW_TRACKING_URI in the environment wins, so a .env file can point runs
    at a remote server (DagsHub) without editing the committed config. Falls
    back to the local sqlite database in conf/config.yaml.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri

    prefix = "sqlite:///"
    if uri.startswith(prefix):
        path = uri[len(prefix):]
        if not os.path.isabs(path):
            path = os.path.join(root, path)
        return prefix + path.replace("\\", "/")
    return uri


def score(cfg: DictConfig, X: pd.DataFrame, labels: np.ndarray) -> dict:
    metrics = {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "misfit_rate": float((silhouette_samples(X, labels) < 0).mean()),
    }

    # Stability: refit from different starting seeds and compare the grouping.
    # ARI is 1.0 for identical partitions and 0.0 for chance agreement, so a
    # high value means the segments are a feature of the data rather than an
    # artefact of where K-Means happened to start.
    seeds = int(cfg.evaluate.stability_seeds)
    if seeds > 1:
        aris = [
            adjusted_rand_score(
                labels,
                KMeans(
                    n_clusters=cfg.model.n_clusters,
                    random_state=seed,
                    n_init=cfg.model.n_init,
                ).fit_predict(X),
            )
            for seed in range(seeds)
        ]
        metrics["stability_ari_mean"] = float(np.mean(aris))
        metrics["stability_ari_min"] = float(np.min(aris))

    return metrics


def profile(target: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster summary in real pounds and days, for interpreting segments."""
    out = target.groupby("Cluster").agg(
        Customers=("CustomerID", "size"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Total=("Monetary", "sum"),
    ).round(1)

    out["pct_customers"] = (out["Customers"] / len(target) * 100).round(1)
    out["pct_sales"] = (out["Total"] / target["Monetary"].sum() * 100).round(1)
    return out


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = hydra.utils.get_original_cwd()

    X = pd.read_csv(os.path.join(root, cfg.paths.features))
    target = pd.read_csv(os.path.join(root, cfg.paths.target))
    model = joblib.load(os.path.join(root, cfg.paths.model))

    labels = model.predict(X)
    target["Cluster"] = labels

    metrics = score(cfg, X, labels)
    metrics["inertia"] = float(model.inertia_)
    summary = profile(target)

    log.info("cluster profile:\n%s", summary.to_string())
    for name, value in metrics.items():
        log.info("%-20s %.4f", name, value)

    with open(os.path.join(root, cfg.paths.metrics), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    profile_path = os.path.join(root, cfg.paths.profile)
    os.makedirs(os.path.dirname(profile_path) or ".", exist_ok=True)
    summary.to_csv(profile_path)
    target.to_csv(os.path.join(root, cfg.paths.segmented), index=False)

    if cfg.mlflow.enabled:
        import mlflow

        uri = _resolve_tracking_uri(cfg, root)
        log.info("mlflow tracking uri: %s", uri)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run(run_name=f"kmeans-k{cfg.model.n_clusters}"):
            mlflow.log_params({
                "n_clusters": cfg.model.n_clusters,
                "random_state": cfg.model.random_state,
                "log_transform": cfg.features.log_transform,
                "scaler": cfg.features.scaler,
                "country": cfg.data.country,
                "n_customers": len(target),
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dvc_stage", "evaluate")
            mlflow.log_artifact(os.path.join(root, cfg.paths.metrics))
            mlflow.log_artifact(os.path.join(root, cfg.paths.profile))


if __name__ == "__main__":
    main()
