"""DVC stage 3: score the fitted model and profile the segments.

    uv run python src/evaluate.py

Writes metrics.json (tracked by `dvc metrics show`), the per-cluster profile,
and the customer table with its cluster label. Also logs the run to MLflow.

Inertia alone cannot tell you whether a clustering is real - it always falls as
k rises. The metrics here answer three separate questions: are the clusters
separated (silhouette, Davies-Bouldin, Calinski-Harabasz), does every cluster
hold together (per-cluster silhouette, misfit rate), and would you get the same
grouping again (ARI across seeds, or across subsamples for DBSCAN).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, cast

import hydra
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import DBSCAN, KMeans
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
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

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


def _run_name(cfg: DictConfig) -> str:
    if str(cfg.model.algorithm).lower() == "kmeans":
        return f"kmeans-k{cfg.model.n_clusters}"
    return f"dbscan-eps{cfg.model.eps}-min{cfg.model.min_samples}"


def score(cfg: DictConfig, X: pd.DataFrame, labels: np.ndarray, model) -> dict:
    # Declared as float so the scores assigned below keep their decimals.
    metrics: dict[str, float] = {"n_clusters_found": len(set(labels) - {-1})}

    # DBSCAN labels outliers -1. That is not a cluster, so including those
    # points would make every separation metric meaningless. Score the
    # clustered points and report the noise share separately.
    keep = labels != -1
    metrics["noise_rate"] = float((~keep).mean())
    Xc, lc = X[keep], labels[keep]

    if metrics["n_clusters_found"] < 2:
        log.warning("fewer than 2 clusters found - separation metrics skipped")
        return metrics

    metrics["silhouette"] = float(silhouette_score(Xc, lc))
    metrics["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(Xc, lc))
    per_point = np.asarray(silhouette_samples(Xc, lc), dtype=float)
    metrics["misfit_rate"] = float((per_point < 0).mean())

    # Inertia only exists for centroid-based models.
    if hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)

    metrics.update(_stability(cfg, X, labels))
    return metrics


def _stability(cfg: DictConfig, X: pd.DataFrame, labels: np.ndarray) -> dict:
    """Would you get the same grouping again?

    ARI is 1.0 for identical partitions and 0.0 for chance agreement.

    K-Means is seeded, so refitting with different seeds tests whether the
    segments survive a different starting position. DBSCAN is deterministic -
    seeds would give 1.0 every time and measure nothing - so it is refitted on
    random 80% subsamples and compared on the rows they share.
    """
    n = int(cfg.evaluate.stability_seeds)
    if n < 2:
        return {}

    algo = str(cfg.model.algorithm).lower()
    rng = np.random.default_rng(0)
    aris = []

    for i in range(n):
        if algo == "kmeans":
            other = KMeans(
                n_clusters=cfg.model.n_clusters,
                random_state=i,
                n_init=cfg.model.n_init,
            ).fit_predict(X)
            aris.append(adjusted_rand_score(labels, other))
        else:
            idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
            other = DBSCAN(eps=cfg.model.eps, min_samples=cfg.model.min_samples).fit_predict(X.iloc[idx])
            aris.append(adjusted_rand_score(labels[idx], other))

    return {
        "stability_ari_mean": float(np.mean(aris)),
        "stability_ari_min": float(np.min(aris)),
    }


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

    # Labels come from train.py rather than model.predict(): DBSCAN has no
    # predict() and can only label the data it was fitted on.
    labels = pd.read_csv(os.path.join(root, cfg.paths.labels))["Cluster"].to_numpy()
    target["Cluster"] = labels

    metrics = score(cfg, X, labels, model)
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
        with mlflow.start_run(run_name=_run_name(cfg)):
            params = cast(
                dict[str, Any],
                OmegaConf.to_container(cfg.model, resolve=True),
            )
            params.update({
                "log_transform": cfg.features.log_transform,
                "scaler": cfg.features.scaler,
                "country": cfg.data.country,
                "n_customers": len(target),
            })
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dvc_stage", "evaluate")
            mlflow.log_artifact(os.path.join(root, cfg.paths.metrics))
            mlflow.log_artifact(os.path.join(root, cfg.paths.profile))


if __name__ == "__main__":
    main()
