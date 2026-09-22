"""DVC stage 1: raw Excel -> customer-level RFM table and scaled feature matrix.

    uv run python src/featurize.py
    uv run python src/featurize.py features.scaler=robust

Follows Chen et al. (2012), pages 4-5. The rule that matters: cancellations are
never deleted from the transaction frame. Deleting a C invoice removes the
refund but keeps the original order, which invents revenue that never existed
(1,382 customers and GBP 528,823 in this dataset). Instead, Recency and
Frequency are computed from purchases only, while Monetary sums everything so
refunds subtract.
"""

from __future__ import annotations

import logging
import os

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(cfg: DictConfig, root: str) -> pd.DataFrame:
    """Read the raw Excel file and apply row-level cleaning."""
    df = pd.read_excel(os.path.join(root, cfg.paths.raw))
    log.info("raw rows: %d", len(df))

    if cfg.data.drop_duplicates:
        df = df.drop_duplicates()

    if cfg.data.drop_missing_customer:
        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = df["CustomerID"].astype(int)

    if cfg.data.country:
        df = df[df["Country"] == cfg.data.country].copy()

    if cfg.data.drop_zero_price:
        df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["Time"] = df["InvoiceDate"].dt.time
    df = df.drop(columns="InvoiceDate")

    df["Amount"] = df["Quantity"] * df["UnitPrice"]
    log.info("cleaned rows: %d", len(df))
    return df


def build_target(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one row per customer (paper Figure 1 columns)."""
    is_cancel = df["InvoiceNo"].astype(str).str.startswith(cfg.data.cancellation_prefix)
    purchases = df[~is_cancel]
    log.info("cancellations: %d | purchases: %d", is_cancel.sum(), len(purchases))

    snapshot = df["Date"].max() + pd.Timedelta(days=1)

    target = purchases.groupby("CustomerID").agg(
        First_Purchase=("Date", lambda x: (snapshot - x.min()).days),
        Recency=("Date", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Min=("Amount", "min"),
        Max=("Amount", "max"),
        Mean=("Amount", "mean"),
    )

    # Monetary uses the full frame so refunds net out against purchases.
    target["Monetary"] = df.groupby("CustomerID")["Amount"].sum()

    target = target.round(2)
    if cfg.data.drop_non_positive_monetary:
        dropped = (target["Monetary"] <= 0).sum()
        target = target[target["Monetary"] > 0]
        log.info("dropped %d customers with non-positive Monetary", dropped)

    target = target.reset_index()
    return target[[
        "CustomerID", "First_Purchase", "Recency",
        "Frequency", "Monetary", "Min", "Max", "Mean",
    ]]


def build_features(cfg: DictConfig, target: pd.DataFrame):
    """Log-transform and scale the RFM columns into the K-Means input matrix.

    Returns (features, scaler). The fitted scaler is handed back so it can be
    saved: scoring a new customer has to reuse the training set's mean and
    standard deviation. Refitting on incoming rows would make every value zero,
    and every customer would land in the same cluster.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    cols = list(cfg.features.columns)
    frame = target[cols]

    if cfg.features.log_transform:
        frame = pd.DataFrame(np.log1p(frame), columns=cols, index=frame.index)
        log.info("skew after log1p: %s", frame.skew().round(2).to_dict())

    name = str(cfg.features.scaler).lower()
    if name in ("none", "null", ""):
        return frame, None

    scalers = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    if name not in scalers:
        raise ValueError(f"unknown scaler {name!r}; expected one of {list(scalers)} or 'none'")

    scaler = scalers[name]()
    scaled = pd.DataFrame(scaler.fit_transform(frame), columns=cols)
    return scaled, scaler


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = hydra.utils.get_original_cwd()

    df = load(cfg, root)
    target = build_target(cfg, df)
    features, scaler = build_features(cfg, target)
    log.info("customers: %d | features: %s", len(target), tuple(features.shape))

    for rel in (cfg.paths.target, cfg.paths.features, cfg.paths.scaler):
        os.makedirs(os.path.dirname(os.path.join(root, rel)) or ".", exist_ok=True)

    target.to_csv(os.path.join(root, cfg.paths.target), index=False)
    features.to_csv(os.path.join(root, cfg.paths.features), index=False)
    log.info("wrote %s and %s", cfg.paths.target, cfg.paths.features)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(root, cfg.paths.scaler))
        log.info("wrote %s", cfg.paths.scaler)


if __name__ == "__main__":
    main()
