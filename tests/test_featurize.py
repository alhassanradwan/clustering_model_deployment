"""Tests for the preprocessing decisions.

These need no model, no credentials and no Excel file - every test builds a
handful of rows by hand. They guard the rule the whole analysis rests on:
cancellations are never deleted, because deleting a C invoice removes the
refund but keeps the original order and invents revenue that never existed.
"""

from __future__ import annotations

import pandas as pd
import pytest
from typing import cast
from omegaconf import DictConfig, OmegaConf

from featurize import build_features, build_target


def make_cfg(scaler: str = "standard") -> DictConfig:
    """The slice of conf/config.yaml that featurize.py actually reads.

    Built fresh per call rather than merged onto a shared object: OmegaConf.merge
    is typed as returning DictConfig *or* ListConfig, which does not satisfy the
    DictConfig these functions ask for.
    """
    return OmegaConf.create(
        {
            "data": {
                "cancellation_prefix": "C",
                "drop_non_positive_monetary": True,
            },
            "features": {
                "columns": ["Recency", "Frequency", "Monetary"],
                "log_transform": True,
                "scaler": scaler,
            },
        }
    )


CFG = make_cfg()


def frame(rows: list[dict]) -> pd.DataFrame:
    """The four columns build_target actually reads."""
    df = pd.DataFrame(rows, columns=["InvoiceNo", "CustomerID", "Date", "Amount"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_a_fully_cancelled_order_is_worth_nothing() -> None:
    """Customer 12346, the real case: 74,215 jars ordered and cancelled 16
    minutes later. Deleting the C row would make them a GBP 77,183 customer.
    """
    target = build_target(
        CFG,
        frame(
            [
                {"InvoiceNo": "541431", "CustomerID": 12346,
                 "Date": "2011-01-18", "Amount": 77183.60},
                {"InvoiceNo": "C541433", "CustomerID": 12346,
                 "Date": "2011-01-18", "Amount": -77183.60},
                {"InvoiceNo": "600001", "CustomerID": 17850,
                 "Date": "2011-02-01", "Amount": 100.00},
            ]
        ),
    )

    # Nets to zero, so the customer is dropped rather than counted at 77,183.
    assert 12346 not in set(target["CustomerID"])
    assert 17850 in set(target["CustomerID"])


def test_a_refund_subtracts_from_monetary() -> None:
    target = build_target(
        CFG,
        frame(
            [
                {"InvoiceNo": "A1", "CustomerID": 1,
                 "Date": "2011-01-01", "Amount": 1000.0},
                {"InvoiceNo": "A2", "CustomerID": 1,
                 "Date": "2011-01-05", "Amount": 500.0},
                {"InvoiceNo": "CA3", "CustomerID": 1,
                 "Date": "2011-01-06", "Amount": -300.0},
            ]
        ),
    )

    row = target.set_index("CustomerID").loc[1]
    assert float(cast(float, row["Monetary"])) == pytest.approx(1200.0)  # 1000 + 500 - 300
    assert cast(int, row["Frequency"]) == 2  # the refund is not a third visit


def test_recency_ignores_refunds() -> None:
    """A refund is not a shop visit, so it must not make a customer look recent."""
    target = build_target(
        CFG,
        frame(
            [
                {"InvoiceNo": "A1", "CustomerID": 1,
                 "Date": "2011-01-01", "Amount": 500.0},
                {"InvoiceNo": "CA2", "CustomerID": 1,
                 "Date": "2011-01-10", "Amount": -100.0},
                {"InvoiceNo": "B1", "CustomerID": 2,
                 "Date": "2011-01-10", "Amount": 200.0},
            ]
        ),
    )

    recency = target.set_index("CustomerID")["Recency"]
    # Snapshot is the 11th. Customer 1 last *bought* on the 1st, not the 10th.
    assert recency[1] == 10
    assert recency[2] == 1


def test_frequency_counts_invoices_not_lines() -> None:
    target = build_target(
        CFG,
        frame(
            [
                {"InvoiceNo": "A1", "CustomerID": 1,
                 "Date": "2011-01-01", "Amount": 10.0},
                {"InvoiceNo": "A1", "CustomerID": 1,
                 "Date": "2011-01-01", "Amount": 20.0},
                {"InvoiceNo": "A2", "CustomerID": 1,
                 "Date": "2011-01-02", "Amount": 30.0},
            ]
        ),
    )

    assert target.set_index("CustomerID").loc[1, "Frequency"] == 2


def test_features_are_logged_scaled_and_the_scaler_comes_back() -> None:
    """The scaler must be returned so it can be saved.

    Refitting on a single incoming row would subtract that row's own mean and
    hand every customer [0, 0, 0] - the same cluster, every time, silently.
    """
    target = pd.DataFrame(
        {
            "Recency": [1.0, 50.0, 200.0, 10.0],
            "Frequency": [20.0, 3.0, 1.0, 8.0],
            "Monetary": [10000.0, 1200.0, 300.0, 2000.0],
        }
    )

    scaled, scaler = build_features(CFG, target)

    assert scaler is not None
    assert list(scaled.columns) == ["Recency", "Frequency", "Monetary"]
    assert len(scaled) == len(target)
    # StandardScaler centres each column on zero.
    assert scaled.mean().abs().max() == pytest.approx(0.0, abs=1e-9)
    # The scaler holds the *training* set's statistics, in log space.
    assert scaler.mean_.shape == (3,)


def test_scaler_none_returns_the_log_frame_unscaled() -> None:
    cfg = make_cfg(scaler="none")
    target = pd.DataFrame(
        {"Recency": [1.0, 50.0], "Frequency": [20.0, 3.0], "Monetary": [1000.0, 10.0]}
    )

    frame_out, scaler = build_features(cfg, target)

    assert scaler is None
    assert frame_out.mean().abs().max() > 0  # not centred, just log1p'd


def test_unknown_scaler_is_rejected() -> None:
    cfg = make_cfg(scaler="quantum")
    target = pd.DataFrame(
        {"Recency": [1.0], "Frequency": [1.0], "Monetary": [1.0]}
    )

    with pytest.raises(ValueError, match="unknown scaler"):
        build_features(cfg, target)
