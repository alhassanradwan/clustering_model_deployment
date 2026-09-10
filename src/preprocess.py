"""RFM preprocessing for the Online Retail dataset.

Mirrors preprocessing.ipynb. Follows Chen et al. (2012), pages 4-5.

The one rule that matters here: cancellations are never deleted from `df`.
Deleting a C invoice removes the refund but keeps the original order, which
invents revenue that never existed (1,382 customers, GBP 528,823 in this data).
Instead, Recency/Frequency come from purchases only, while Monetary is summed
over everything so refunds subtract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def load(cfg: DictConfig, root: str) -> pd.DataFrame:
    """Read the raw Excel file and apply row-level cleaning."""
    import os

    df = pd.read_excel(os.path.join(root, cfg.data.path))

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
    return df


def build_target(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one row per customer (paper Figure 1 columns)."""
    prefix = cfg.data.cancellation_prefix
    is_cancel = df["InvoiceNo"].astype(str).str.startswith(prefix)
    purchases = df[~is_cancel]

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
        target = target[target["Monetary"] > 0]

    target = target.reset_index()
    return target[[
        "CustomerID", "First_Purchase", "Recency",
        "Frequency", "Monetary", "Min", "Max", "Mean",
    ]]


def build_features(cfg: DictConfig, target: pd.DataFrame) -> np.ndarray:
    """Log-transform and scale the RFM columns into the K-Means input matrix."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    cols = list(cfg.features.columns)
    frame = target[cols]

    if cfg.features.log_transform:
        frame = np.log1p(frame)

    scalers = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }

    name = str(cfg.features.scaler).lower()
    if name in ("none", "null", ""):
        return frame.to_numpy()
    if name not in scalers:
        raise ValueError(f"unknown scaler {name!r}; expected one of {list(scalers)} or 'none'")

    return scalers[name]().fit_transform(frame)
