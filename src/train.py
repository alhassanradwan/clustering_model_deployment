"""DVC stage 2: fit a clustering model on the scaled feature matrix.

    uv run python src/train.py
    uv run python src/train.py model.n_clusters=7
    uv run python src/train.py model=dbscan model.eps=0.4
    uv run python src/train.py --multirun model.n_clusters=3,4,5,6,7,8

Reads data/features.csv rather than the raw Excel file, so a sweep does not
re-run the 80-second spreadsheet load each time.

Labels are written alongside the model because DBSCAN has no predict() - it can
only label the data it was fitted on, so evaluate.py reads them from disk
instead of recomputing.
"""

from __future__ import annotations

import logging
import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import DBSCAN, KMeans

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig):
    algo = str(cfg.model.algorithm).lower()

    if algo == "kmeans":
        return KMeans(
            n_clusters=cfg.model.n_clusters,
            random_state=cfg.model.random_state,
            n_init=cfg.model.n_init,
        )
    if algo == "dbscan":
        return DBSCAN(eps=cfg.model.eps, min_samples=cfg.model.min_samples)

    raise ValueError(f"unknown algorithm {algo!r}; expected 'kmeans' or 'dbscan'")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = hydra.utils.get_original_cwd()

    X = pd.read_csv(os.path.join(root, cfg.paths.features))
    log.info("features: %s", tuple(X.shape))

    model = build_model(cfg)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    log.info("%s: %d clusters, %d noise points", cfg.model.algorithm, n_clusters, n_noise)

    model_path = os.path.join(root, cfg.paths.model)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    pd.DataFrame({"Cluster": labels}).to_csv(os.path.join(root, cfg.paths.labels), index=False)
    log.info("wrote %s and %s", cfg.paths.model, cfg.paths.labels)


if __name__ == "__main__":
    main()
