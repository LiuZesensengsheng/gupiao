from __future__ import annotations

import pandas as pd
import pytest

from src.infrastructure.market_context import (
    _align_external_context_to_next_market_date,
    build_market_context_features,
)
from src.infrastructure.market_data import DataError, _normalize_external_index_columns


def _make_us_index_frame(start: str, periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    base = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0 + base,
            "high": 101.0 + base,
            "low": 99.0 + base,
            "close": 100.5 + base,
            "volume": 1_000_000.0 + base * 1000.0,
            "amount": (100.5 + base) * (1_000_000.0 + base * 1000.0),
            "symbol": ".INX",
        }
    )


def _make_cn_etf_frame(start: str, periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    base = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 1.0 + base * 0.01,
            "high": 1.01 + base * 0.01,
            "low": 0.99 + base * 0.01,
            "close": 1.005 + base * 0.01,
            "volume": 10_000_000.0 + base * 5000.0,
            "amount": (1.005 + base * 0.01) * (10_000_000.0 + base * 5000.0),
            "symbol": "510300",
        }
    )


def test_normalize_external_index_columns_supports_cn_headers() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": ["100", "101"],
            "最高": ["102", "103"],
            "最低": ["99", "100"],
            "收盘": ["101", "102"],
            "成交量": ["1000", "1100"],
            "成交额": ["101000", "112200"],
        }
    )

    got = _normalize_external_index_columns(raw, symbol=".INX")

    assert list(got.columns) == ["date", "open", "high", "low", "close", "volume", "amount", "symbol"]
    assert got["symbol"].tolist() == [".INX", ".INX"]
    assert got["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert got["close"].tolist() == [101.0, 102.0]


def test_align_external_context_maps_to_next_a_share_date_without_same_day_leakage() -> None:
    feature_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
            "ret_1": [0.1, 0.2, 0.3],
        }
    )
    market_dates = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"]))

    got = _align_external_context_to_next_market_date(
        feature_frame=feature_frame,
        market_dates=market_dates,
    )

    assert got["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04", "2024-01-08"]
    assert got["ret_1"].tolist() == [0.1, 0.2, 0.3]


def test_build_market_context_features_merges_us_index_context(monkeypatch: pytest.MonkeyPatch) -> None:
    market_dates = pd.Series(pd.date_range("2024-03-01", periods=8, freq="B"))

    monkeypatch.setattr(
        "src.infrastructure.market_context._build_index_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context._build_breadth_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )

    def fake_fetch_us_index_daily(symbol: str, **_: object) -> pd.DataFrame:
        frame = _make_us_index_frame("2023-11-01")
        frame["symbol"] = symbol
        return frame

    monkeypatch.setattr("src.infrastructure.market_context.fetch_us_index_daily", fake_fetch_us_index_daily)

    got = build_market_context_features(
        source="local",
        data_dir="data",
        start="2024-03-01",
        end="2024-03-31",
        market_dates=market_dates,
        use_margin_features=False,
        use_us_index_context=True,
        us_index_source="akshare",
        min_valid_ratio=0.0,
        min_valid_points=1,
    )

    assert any(col.startswith("us_inx_") for col in got.feature_columns)
    assert any(col.startswith("us_ndx_") for col in got.feature_columns)
    assert any(col.startswith("us_dji_") for col in got.feature_columns)
    assert len(got.frame) == len(market_dates)
    assert got.frame.filter(regex=r"^us_(inx|ndx|dji)_").notna().all().all()


def test_build_market_context_features_merges_us_sector_and_cn_etf_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_dates = pd.Series(pd.date_range("2024-03-01", periods=8, freq="B"))

    monkeypatch.setattr(
        "src.infrastructure.market_context._build_index_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context._build_breadth_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context.fetch_us_etf_daily",
        lambda symbol, **_: _make_us_index_frame("2023-11-01").assign(symbol=symbol),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context.fetch_cn_etf_daily",
        lambda symbol, **_: _make_cn_etf_frame("2024-01-01").assign(symbol=symbol),
    )

    got = build_market_context_features(
        source="local",
        data_dir="data",
        start="2024-03-01",
        end="2024-03-31",
        market_dates=market_dates,
        use_margin_features=False,
        use_us_sector_etf_context=True,
        use_cn_etf_context=True,
        cn_etf_source="akshare",
        min_valid_ratio=0.0,
        min_valid_points=1,
    )

    assert any(col.startswith("us_etf_xlk_") for col in got.feature_columns)
    assert any(col.startswith("cn_etf_300_") for col in got.feature_columns)
    assert got.frame.filter(regex=r"^(us_etf_xlk|cn_etf_300)_").notna().all().all()


def test_build_market_context_features_skips_us_index_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    market_dates = pd.Series(pd.date_range("2024-03-01", periods=5, freq="B"))

    monkeypatch.setattr(
        "src.infrastructure.market_context._build_index_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context._build_breadth_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )

    got = build_market_context_features(
        source="local",
        data_dir="data",
        start="2024-03-01",
        end="2024-03-31",
        market_dates=market_dates,
        use_margin_features=False,
        use_us_index_context=False,
        min_valid_ratio=0.0,
        min_valid_points=1,
    )

    assert not any(col.startswith("us_") for col in got.feature_columns)
    assert list(got.frame.columns) == ["date"]


def test_build_market_context_features_soft_skips_us_index_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    market_dates = pd.Series(pd.date_range("2024-03-01", periods=5, freq="B"))

    monkeypatch.setattr(
        "src.infrastructure.market_context._build_index_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context._build_breadth_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context.fetch_us_index_daily",
        lambda **_: (_ for _ in ()).throw(DataError("remote unavailable")),
    )

    got = build_market_context_features(
        source="local",
        data_dir="data",
        start="2024-03-01",
        end="2024-03-31",
        market_dates=market_dates,
        use_margin_features=False,
        use_us_index_context=True,
        us_index_source="akshare",
        min_valid_ratio=0.0,
        min_valid_points=1,
    )

    assert not any(col.startswith("us_") for col in got.feature_columns)
    assert any("US index context skipped" in note for note in got.notes)


def test_build_market_context_features_keeps_breadth_diagnostics_for_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_dates = pd.Series(pd.date_range("2024-03-01", periods=5, freq="B"))

    monkeypatch.setattr(
        "src.infrastructure.market_context._build_index_context",
        lambda **_: (pd.DataFrame(columns=["date"]), [], []),
    )
    monkeypatch.setattr(
        "src.infrastructure.market_context._build_breadth_context",
        lambda **_: (
            pd.DataFrame(
                {
                    "date": market_dates,
                    "breadth_up_ratio": [0.55, 0.56, 0.57, 0.58, 0.59],
                    "breadth_down_ratio": [0.30, 0.29, 0.28, 0.27, 0.26],
                    "breadth_up_down_diff": [0.25, 0.27, 0.29, 0.31, 0.33],
                    "breadth_limit_up_ratio": [0.01] * 5,
                    "breadth_limit_down_ratio": [0.00] * 5,
                    "breadth_limit_spread": [0.01] * 5,
                    "breadth_amount_z20": [0.2, 0.3, 0.4, 0.5, 0.6],
                    "breadth_coverage": [300.0] * 5,
                    "breadth_advancers": [180.0, 181.0, 182.0, 183.0, 184.0],
                    "breadth_decliners": [90.0, 89.0, 88.0, 87.0, 86.0],
                    "breadth_flats": [30.0] * 5,
                    "breadth_limit_up_count": [8.0, 9.0, 10.0, 11.0, 12.0],
                    "breadth_limit_down_count": [1.0, 1.0, 1.0, 1.0, 0.0],
                    "breadth_new_high_count": [20.0, 21.0, 22.0, 23.0, 24.0],
                    "breadth_new_low_count": [5.0, 4.0, 4.0, 3.0, 3.0],
                    "breadth_median_return": [0.003, 0.004, 0.005, 0.006, 0.007],
                    "breadth_sample_amount": [1.0e11, 1.1e11, 1.2e11, 1.3e11, 1.4e11],
                }
            ),
            ["breadth_up_ratio", "breadth_down_ratio", "breadth_up_down_diff", "breadth_coverage"],
            [],
        ),
    )

    got = build_market_context_features(
        source="local",
        data_dir="data",
        start="2024-03-01",
        end="2024-03-31",
        market_dates=market_dates,
        use_margin_features=False,
        min_valid_ratio=0.0,
        min_valid_points=1,
    )

    assert "breadth_advancers" in got.frame.columns
    assert "breadth_limit_up_count" in got.frame.columns
    assert "breadth_new_high_count" in got.frame.columns
    assert got.frame.iloc[-1]["breadth_advancers"] == 184.0
    assert got.frame.iloc[-1]["breadth_limit_up_count"] == 12.0
