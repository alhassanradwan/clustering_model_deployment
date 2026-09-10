"""K-Means customer segmentation, driven by Hydra config.

Single run:
    uv run python src/train.py

Override anything:
    uv run python src/train.py model.n_clusters=7 features.scaler=robust

Sweep k (each run gets its own output folder and MLflow entry):
    uv run python src/train.py --multirun model.n_clusters=3,4,5,6,7,8
"""

from __future__ import annotations

import logging
import os

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from preprocess import build_features, build_target, load

log = logging.getLogger(__name__)


def _resolve_tracking_uri(uri: str, root: str) -> str:
    """Point a relative sqlite URI at the project root.

    Keeps one database at the project root instead of depending on the
    process working directory.
    """
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        path = uri[len(prefix):]
        if not os.path.isabs(path):
            path = os.path.join(root, path)
        return prefix + path.replace("\\", "/")
    return uri


def evaluate(cfg: DictConfig, X: np.ndarray, labels: np.ndarray) -> dict:
    """Internal validity metrics plus a stability check across random seeds."""
    metrics = {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }

    samples = silhouette_samples(X, labels)
    metrics["misfit_rate"] = float((samples < 0).mean())

    # Stability: does a different starting seed find the same partition?
    # ARI of 1.0 means identical grouping, 0.0 means chance.
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
def main(cfg: DictConfig) -> float:
    # Resolve data paths against the directory the command was launched from.
    root = hydra.utils.get_original_cwd()

    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    df = load(cfg, root)
    target = build_target(cfg, df)
    X = build_features(cfg, target)
    log.info("transactions=%d customers=%d", len(df), len(target))

    model = KMeans(
        n_clusters=cfg.model.n_clusters,
        random_state=cfg.model.random_state,
        n_init=cfg.model.n_init,
    )
    target["Cluster"] = model.fit_predict(X)

    metrics = evaluate(cfg, X, target["Cluster"].to_numpy())
    metrics["inertia"] = float(model.inertia_)

    summary = profile(target)
    log.info("cluster profile:\n%s", summary.to_string())
    for name, value in metrics.items():
        log.info("%-20s %.4f", name, value)

    # Write into this run's own Hydra directory so sweep jobs never overwrite
    # each other. Resolved explicitly because with version_base=None Hydra does
    # not chdir into the run directory.
    out_dir = os.path.join(HydraConfig.get().runtime.output_dir, cfg.output.dir)
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "cluster_profile.csv"))
    if cfg.output.save_target:
        target.to_csv(os.path.join(out_dir, "segmented_customers.csv"), index=False)
    if cfg.output.save_model:
        joblib.dump(model, os.path.join(out_dir, "kmeans.joblib"))

    if cfg.mlflow.enabled:
        import mlflow

        mlflow.set_tracking_uri(_resolve_tracking_uri(cfg.mlflow.tracking_uri, root))
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run():
            mlflow.log_params({
                "n_clusters": cfg.model.n_clusters,
                "random_state": cfg.model.random_state,
                "log_transform": cfg.features.log_transform,
                "scaler": cfg.features.scaler,
                "country": cfg.data.country,
                "n_customers": len(target),
            })
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(out_dir)

    # Returned so Hydra sweepers can optimise against it.
    return metrics["silhouette"]


if __name__ == "__main__":
    main()
