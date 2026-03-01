from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.domain.entities import Security
from src.infrastructure.features import stock_feature_columns
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_data import DataError, load_symbol_daily
from src.infrastructure.features import make_stock_feature_frame


PANEL_UNIVERSE_RANK_BASE = [
    "ret_20",
    "trend_20_60",
    "vol_ratio_20",
    "price_pos_20",
    "obv_z_20",
    "amihud_20",
]

PANEL_SECTOR_RANK_BASE = [
    "ret_20",
    "trend_20_60",
    "vol_ratio_20",
    "price_pos_20",
]


@dataclass(frozen=True)
class StockPanelDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    notes: list[str]
    symbol_status: dict[str, str] = field(default_factory=dict)


STATUS_NORMAL = "normal"
STATUS_HALTED = "halted"
STATUS_DELISTED = "delisted"
STATUS_DATA_INSUFFICIENT = "data_insufficient"

MIN_HISTORY_ROWS = 80
HALT_GAP_DAYS = 3
DELIST_GAP_DAYS = 20


def _infer_symbol_status(
    *,
    stock_raw: pd.DataFrame,
) -> str:
    if stock_raw.empty:
        return STATUS_DATA_INSUFFICIENT

    clean = stock_raw.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean = clean.dropna(subset=["date"]).sort_values("date")
    if clean.empty or len(clean) < MIN_HISTORY_ROWS:
        return STATUS_DATA_INSUFFICIENT
    return STATUS_NORMAL


def _rank_pct(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    grouped = frame.groupby(group_cols, observed=True)[value_col]
    return grouped.rank(method="average", pct=True)


def _attach_cross_section_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    extra_cols: list[str] = []

    for col in PANEL_UNIVERSE_RANK_BASE:
        if col not in out.columns:
            continue
        new_col = f"xs_{col}_rank_pct"
        out[new_col] = _rank_pct(out, ["date"], col).astype(float)
        extra_cols.append(new_col)

    for col in PANEL_SECTOR_RANK_BASE:
        if col not in out.columns:
            continue
        new_col = f"sec_{col}_rank_pct"
        out[new_col] = _rank_pct(out, ["date", "sector"], col).astype(float)
        extra_cols.append(new_col)

    sector_agg = (
        out.groupby(["date", "sector"], as_index=False, observed=True)
        .agg(
            sector_ret_5_mean=("ret_5", "mean"),
            sector_ret_20_mean=("ret_20", "mean"),
            sector_trend_20_60_mean=("trend_20_60", "mean"),
            sector_vol_ratio_20_mean=("vol_ratio_20", "mean"),
            sector_price_pos_20_mean=("price_pos_20", "mean"),
            sector_amihud_20_mean=("amihud_20", "mean"),
            sector_member_count=("symbol", "count"),
        )
    )
    out = out.merge(sector_agg, on=["date", "sector"], how="left", validate="m:1")
    out["sector_ret_20_minus_mkt"] = out["sector_ret_20_mean"] - out["mkt_ret_20"]
    out["sector_trend_20_60_minus_mkt"] = out["sector_trend_20_60_mean"] - out["mkt_trend_20_60"]
    extra_cols.extend(
        [
            "sector_ret_5_mean",
            "sector_ret_20_mean",
            "sector_trend_20_60_mean",
            "sector_vol_ratio_20_mean",
            "sector_price_pos_20_mean",
            "sector_amihud_20_mean",
            "sector_member_count",
            "sector_ret_20_minus_mkt",
            "sector_trend_20_60_minus_mkt",
        ]
    )

    for col in extra_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out.groupby("symbol", observed=True)[col].ffill()
        if out[col].isna().any():
            median = out[col].median(skipna=True)
            out[col] = out[col].fillna(0.0 if pd.isna(median) else float(median))
    return out, extra_cols


def _attach_relative_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["excess_ret_1_vs_mkt"] = out["fwd_ret_1"] - out["mkt_fwd_ret_1"]
    out["excess_ret_5_vs_mkt"] = out["fwd_ret_5"] - out["mkt_fwd_ret_5"]
    out["excess_ret_20_vs_mkt"] = out["fwd_ret_20"] - out["mkt_fwd_ret_20"]

    sector_forward = (
        out.groupby(["date", "sector"], as_index=False, observed=True)
        .agg(
            sector_fwd_ret_1_mean=("fwd_ret_1", "mean"),
            sector_fwd_ret_5_mean=("fwd_ret_5", "mean"),
            sector_fwd_ret_20_mean=("fwd_ret_20", "mean"),
        )
    )
    out = out.merge(sector_forward, on=["date", "sector"], how="left", validate="m:1")
    out["excess_ret_1_vs_sector"] = out["fwd_ret_1"] - out["sector_fwd_ret_1_mean"]
    out["excess_ret_5_vs_sector"] = out["fwd_ret_5"] - out["sector_fwd_ret_5_mean"]
    out["excess_ret_20_vs_sector"] = out["fwd_ret_20"] - out["sector_fwd_ret_20_mean"]

    out["target_1d_excess_mkt_up"] = np.where(
        out["excess_ret_1_vs_mkt"].notna(),
        (out["excess_ret_1_vs_mkt"] > 0).astype(float),
        np.nan,
    )
    out["target_5d_excess_mkt_up"] = np.where(
        out["excess_ret_5_vs_mkt"].notna(),
        (out["excess_ret_5_vs_mkt"] > 0).astype(float),
        np.nan,
    )
    out["target_20d_excess_mkt_up"] = np.where(
        out["excess_ret_20_vs_mkt"].notna(),
        (out["excess_ret_20_vs_mkt"] > 0).astype(float),
        np.nan,
    )
    out["target_1d_excess_sector_up"] = np.where(
        out["excess_ret_1_vs_sector"].notna(),
        (out["excess_ret_1_vs_sector"] > 0).astype(float),
        np.nan,
    )
    out["target_5d_excess_sector_up"] = np.where(
        out["excess_ret_5_vs_sector"].notna(),
        (out["excess_ret_5_vs_sector"] > 0).astype(float),
        np.nan,
    )
    out["target_20d_excess_sector_up"] = np.where(
        out["excess_ret_20_vs_sector"].notna(),
        (out["excess_ret_20_vs_sector"] > 0).astype(float),
        np.nan,
    )
    return out


def build_stock_panel_dataset(
    *,
    stock_securities: Sequence[Security],
    source: str,
    data_dir: str,
    start: str,
    end: str,
    market_frame: pd.DataFrame,
    extra_market_cols: list[str] | None = None,
    use_margin_features: bool = True,
    margin_stock_file: str = "input/margin_stock.csv",
) -> StockPanelDataset:
    notes: list[str] = []
    parts: list[pd.DataFrame] = []
    stock_margin_cols_union: list[str] = []
    symbol_status: dict[str, str] = {}
    raw_latest_dates: dict[str, pd.Timestamp] = {}

    for security in stock_securities:
        try:
            stock_raw = load_symbol_daily(
                symbol=security.symbol,
                source=source,
                data_dir=data_dir,
                start=start,
                end=end,
            )
        except DataError as exc:
            notes.append(f"skip {security.symbol}: {exc}")
            symbol_status[security.symbol] = STATUS_DATA_INSUFFICIENT
            continue

        raw_dates = pd.to_datetime(stock_raw.get("date"), errors="coerce")
        raw_dates = raw_dates.dropna().sort_values()
        if not raw_dates.empty:
            raw_latest_dates[security.symbol] = pd.Timestamp(raw_dates.iloc[-1])
        symbol_status[security.symbol] = _infer_symbol_status(stock_raw=stock_raw)
        frame = make_stock_feature_frame(stock_raw, market_frame)
        stock_margin_cols: list[str] = []
        if use_margin_features:
            margin_frame, margin_cols, margin_notes = build_stock_margin_features(
                margin_stock_file=margin_stock_file,
                symbol=security.symbol,
                start=start,
                end=end,
            )
            notes.extend(margin_notes)
            if margin_cols:
                frame = frame.merge(margin_frame, on="date", how="left", validate="1:1")
                stock_margin_cols = list(margin_cols)
                for col in stock_margin_cols:
                    if col not in stock_margin_cols_union:
                        stock_margin_cols_union.append(col)

        frame["symbol"] = security.symbol
        frame["name"] = security.name
        frame["sector"] = security.sector or "其他"
        frame["tradability_status"] = symbol_status.get(security.symbol, STATUS_NORMAL)
        parts.append(frame)

    if not parts:
        return StockPanelDataset(frame=pd.DataFrame(), feature_columns=[], notes=notes, symbol_status=symbol_status)

    universe_latest = max(raw_latest_dates.values()) if raw_latest_dates else None
    market_dates = pd.to_datetime(market_frame["date"], errors="coerce").dropna().sort_values()
    for symbol, status in list(symbol_status.items()):
        if status != STATUS_NORMAL:
            continue
        last_trade_date = raw_latest_dates.get(symbol)
        if universe_latest is None or last_trade_date is None or market_dates.empty:
            continue
        missing_peer_days = int(((market_dates > last_trade_date) & (market_dates <= universe_latest)).sum())
        if missing_peer_days >= DELIST_GAP_DAYS:
            symbol_status[symbol] = STATUS_DELISTED
        elif missing_peer_days >= HALT_GAP_DAYS:
            symbol_status[symbol] = STATUS_HALTED

    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    panel, panel_extra_cols = _attach_cross_section_features(panel)
    panel = _attach_relative_targets(panel)

    feature_cols = stock_feature_columns(
        extra_market_cols=extra_market_cols,
        extra_stock_cols=stock_margin_cols_union + panel_extra_cols,
    )
    feature_cols = [col for col in feature_cols if col in panel.columns]

    required_cols = feature_cols + [
        "target_1d_excess_mkt_up",
        "target_5d_excess_mkt_up",
        "target_20d_excess_sector_up",
        "excess_ret_1_vs_mkt",
        "excess_ret_5_vs_mkt",
        "excess_ret_20_vs_sector",
    ]
    pre_filter_symbols = set(panel["symbol"].astype(str).unique())
    panel = panel.dropna(subset=[col for col in required_cols if col in panel.columns]).copy()
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    post_filter_symbols = set(panel["symbol"].astype(str).unique())
    for symbol in sorted(pre_filter_symbols - post_filter_symbols):
        if symbol_status.get(symbol) == STATUS_NORMAL:
            symbol_status[symbol] = STATUS_DATA_INSUFFICIENT
        notes.append(f"filter {symbol}: insufficient complete rows after panel assembly")
    return StockPanelDataset(
        frame=panel,
        feature_columns=feature_cols,
        notes=notes,
        symbol_status=symbol_status,
    )
