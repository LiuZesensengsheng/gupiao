from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.entities import Security
from src.infrastructure.panel_dataset import build_stock_panel_dataset


def _make_market_frame(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100.0
    close = []
    for idx in range(n):
        base *= 1.0 + 0.001 + 0.002 * np.sin(idx / 8.0)
        close.append(base)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.full(n, 5_000_000.0, dtype=float)
    amount_arr = close_arr * volume_arr
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": close_arr * 0.999,
            "high": close_arr * 1.01,
            "low": close_arr * 0.99,
            "close": close_arr,
            "volume": volume_arr,
            "amount": amount_arr,
        }
    )
    from src.infrastructure.features import make_market_feature_frame

    return make_market_feature_frame(raw)


def _make_stock_frame(symbol_offset: float, n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 20.0 + symbol_offset
    close = []
    for idx in range(n):
        base *= 1.0 + 0.0015 + 0.0025 * np.sin((idx + symbol_offset) / 7.0)
        close.append(base)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.full(n, 800_000.0 + symbol_offset * 10_000.0, dtype=float)
    amount_arr = close_arr * volume_arr
    return pd.DataFrame(
        {
            "date": dates,
            "open": close_arr * 0.998,
            "high": close_arr * 1.01,
            "low": close_arr * 0.99,
            "close": close_arr,
            "volume": volume_arr,
            "amount": amount_arr,
        }
    )


def test_build_stock_panel_dataset_adds_cross_section_features(monkeypatch) -> None:
    market_frame = _make_market_frame()
    data_map = {
        "AAA.SZ": _make_stock_frame(1.0),
        "BBB.SZ": _make_stock_frame(2.0),
        "CCC.SZ": _make_stock_frame(3.0),
    }

    def fake_load_symbol_daily(*, symbol: str, source: str, data_dir: str, start: str, end: str) -> pd.DataFrame:
        return data_map[symbol].copy()

    def fake_margin(*, margin_stock_file: str, symbol: str, start: str, end: str):
        return pd.DataFrame(columns=["date"]), [], []

    monkeypatch.setattr("src.infrastructure.panel_dataset.load_symbol_daily", fake_load_symbol_daily)
    monkeypatch.setattr("src.infrastructure.panel_dataset.build_stock_margin_features", fake_margin)

    bundle = build_stock_panel_dataset(
        stock_securities=[
            Security("AAA.SZ", "A", "科技"),
            Security("BBB.SZ", "B", "科技"),
            Security("CCC.SZ", "C", "消费"),
        ],
        source="local",
        data_dir="data",
        start="2024-01-01",
        end="2024-12-31",
        market_frame=market_frame,
        extra_market_cols=[],
        use_margin_features=False,
    )

    assert not bundle.frame.empty
    assert "xs_ret_20_rank_pct" in bundle.feature_columns
    assert "sec_ret_20_rank_pct" in bundle.feature_columns
    assert "sector_ret_20_minus_mkt" in bundle.feature_columns
    assert set(bundle.frame["symbol"].unique()) == {"AAA.SZ", "BBB.SZ", "CCC.SZ"}
