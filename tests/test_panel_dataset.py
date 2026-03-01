from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.entities import Security
from src.infrastructure.panel_dataset import build_stock_panel_dataset
from src.infrastructure.market_data import DataError


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
    assert "target_1d_excess_mkt_up" in bundle.frame.columns
    assert "target_5d_excess_mkt_up" in bundle.frame.columns
    assert "target_20d_excess_sector_up" in bundle.frame.columns
    assert "excess_ret_1_vs_mkt" in bundle.frame.columns
    assert "excess_ret_5_vs_mkt" in bundle.frame.columns
    assert "excess_ret_20_vs_sector" in bundle.frame.columns
    assert set(bundle.frame["symbol"].unique()) == {"AAA.SZ", "BBB.SZ", "CCC.SZ"}


def test_build_stock_panel_dataset_handles_sparse_symbols_and_skips_missing_ones(monkeypatch) -> None:
    market_frame = _make_market_frame()
    sparse = _make_stock_frame(2.0).drop(index=list(range(0, 20, 2))).reset_index(drop=True)
    data_map = {
        "AAA.SZ": _make_stock_frame(1.0),
        "BBB.SZ": sparse,
    }

    def fake_load_symbol_daily(*, symbol: str, source: str, data_dir: str, start: str, end: str) -> pd.DataFrame:
        if symbol == "CCC.SZ":
            raise DataError("missing or delisted")
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
    assert set(bundle.frame["symbol"].unique()) == {"AAA.SZ", "BBB.SZ"}
    assert "CCC.SZ" not in set(bundle.frame["symbol"].unique())
    assert any("skip CCC.SZ" in note for note in bundle.notes)
    assert bundle.frame[["date", "symbol"]].duplicated().sum() == 0


def test_build_stock_panel_dataset_handles_single_symbol_universe(monkeypatch) -> None:
    market_frame = _make_market_frame()
    data_map = {
        "AAA.SZ": _make_stock_frame(1.0),
    }

    def fake_load_symbol_daily(*, symbol: str, source: str, data_dir: str, start: str, end: str) -> pd.DataFrame:
        return data_map[symbol].copy()

    def fake_margin(*, margin_stock_file: str, symbol: str, start: str, end: str):
        return pd.DataFrame(columns=["date"]), [], []

    monkeypatch.setattr("src.infrastructure.panel_dataset.load_symbol_daily", fake_load_symbol_daily)
    monkeypatch.setattr("src.infrastructure.panel_dataset.build_stock_margin_features", fake_margin)

    bundle = build_stock_panel_dataset(
        stock_securities=[Security("AAA.SZ", "A", "科技")],
        source="local",
        data_dir="data",
        start="2024-01-01",
        end="2024-12-31",
        market_frame=market_frame,
        extra_market_cols=[],
        use_margin_features=False,
    )

    assert not bundle.frame.empty
    assert set(bundle.frame["symbol"].unique()) == {"AAA.SZ"}
    assert float(bundle.frame["xs_ret_20_rank_pct"].min()) == 1.0
    assert float(bundle.frame["sec_ret_20_rank_pct"].min()) == 1.0


def test_build_stock_panel_dataset_returns_empty_when_all_symbols_unavailable(monkeypatch) -> None:
    market_frame = _make_market_frame()

    def fake_load_symbol_daily(*, symbol: str, source: str, data_dir: str, start: str, end: str) -> pd.DataFrame:
        raise DataError(f"{symbol} unavailable")

    def fake_margin(*, margin_stock_file: str, symbol: str, start: str, end: str):
        return pd.DataFrame(columns=["date"]), [], []

    monkeypatch.setattr("src.infrastructure.panel_dataset.load_symbol_daily", fake_load_symbol_daily)
    monkeypatch.setattr("src.infrastructure.panel_dataset.build_stock_margin_features", fake_margin)

    bundle = build_stock_panel_dataset(
        stock_securities=[
            Security("AAA.SZ", "A", "科技"),
            Security("BBB.SZ", "B", "消费"),
        ],
        source="local",
        data_dir="data",
        start="2024-01-01",
        end="2024-12-31",
        market_frame=market_frame,
        extra_market_cols=[],
        use_margin_features=False,
    )

    assert bundle.frame.empty
    assert bundle.feature_columns == []
    assert len(bundle.notes) == 2
    assert all(note.startswith("skip ") for note in bundle.notes)


def test_build_stock_panel_dataset_infers_symbol_statuses(monkeypatch) -> None:
    market_frame = _make_market_frame()
    normal = _make_stock_frame(1.0)
    halted = _make_stock_frame(2.0).iloc[:-5].reset_index(drop=True)
    delisted = _make_stock_frame(3.0).iloc[:-30].reset_index(drop=True)
    insufficient = _make_stock_frame(4.0, n=40)
    data_map = {
        "AAA.SZ": normal,
        "BBB.SZ": halted,
        "CCC.SZ": delisted,
        "DDD.SZ": insufficient,
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
            Security("DDD.SZ", "D", "消费"),
        ],
        source="local",
        data_dir="data",
        start="2024-01-01",
        end="2024-12-31",
        market_frame=market_frame,
        extra_market_cols=[],
        use_margin_features=False,
    )

    assert bundle.symbol_status["AAA.SZ"] == "normal"
    assert bundle.symbol_status["BBB.SZ"] == "halted"
    assert bundle.symbol_status["CCC.SZ"] == "delisted"
    assert bundle.symbol_status["DDD.SZ"] == "data_insufficient"
