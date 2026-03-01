from __future__ import annotations

import pandas as pd

from src.infrastructure.sector_features import SECTOR_FEATURE_COLUMNS, make_sector_feature_frame
from src.infrastructure.sector_forecast import run_sector_forecast


def _make_market_frame(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series([100.0 + i * 0.2 for i in range(n)])
    open_ = close.shift(1).fillna(close)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": close + 0.3,
            "low": open_ - 0.3,
            "close": close,
            "volume": 1_000_000.0,
            "amount": close * 1_000_000.0,
        }
    )


def _make_sector_frame(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series([100.0 + i * 0.35 for i in range(n)])
    open_ = close.shift(1).fillna(close)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": close + 0.4,
            "low": open_ - 0.4,
            "close": close,
            "volume": 2_000_000.0,
            "amount": close * 2_000_000.0,
            "coverage": 5.0,
        }
    )


def test_make_sector_feature_frame_contains_required_columns() -> None:
    feat = make_sector_feature_frame(_make_sector_frame(), _make_market_frame())

    assert set(SECTOR_FEATURE_COLUMNS).issubset(set(feat.columns))
    assert "sector_target_20d_excess" in feat.columns
    assert feat["date"].notna().all()


def test_run_sector_forecast_returns_ranked_records() -> None:
    records = run_sector_forecast(
        sector_frames={
            "强板块": _make_sector_frame(),
            "次强板块": _make_sector_frame().assign(close=lambda df: df["close"] * 0.98),
        },
        market_raw=_make_market_frame(),
        l2=0.8,
    )

    assert len(records) == 2
    assert records[0].up_20d_prob >= records[1].up_20d_prob
    assert all(0.0 <= item.excess_vs_market_prob <= 1.0 for item in records)
