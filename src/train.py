"""DVC stage 2: fit K-Means on the scaled feature matrix.

    uv run python src/train.py
    uv run python src/train.py model.n_clusters=7
    uv run python src/train.py --multirun model.n_clusters=3,4,5,6,7,8

Reads data/features.csv rather than the raw Excel file, so a sweep over k does
not re-run the 80-second spreadsheet load each time.
"""

from __future__ import annotations

import logging
import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


def build_kmeans(cfg: DictConfig) -> KMeans:
    return KMeans(
        n_clusters=cfg.model.n_clusters,
        random_state=cfg.model.random_state,
        n_init=cfg.model.n_init,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    root = hydra.utils.get_original_cwd()

    X = pd.read_csv(os.path.join(root, cfg.paths.features))
    log.info("features: %s", tuple(X.shape))

    model = build_kmeans(cfg).fit(X)
    log.info("k=%d inertia=%.1f", cfg.model.n_clusters, model.inertia_)

    model_path = os.path.join(root, cfg.paths.model)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    log.info("wrote %s", cfg.paths.model)

    # Returned so Hydra sweepers can optimise against it.
    return float(model.inertia_)


if __name__ == "__main__":
    main()
