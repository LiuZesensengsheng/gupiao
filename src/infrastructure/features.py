from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_20",
    "trend_5_20",
    "trend_20_60",
    "volatility_20",
    "volatility_60",
    "drawdown_20",
    "price_pos_20",
    "vol_ratio_20",
    "vol_conc_5_20",
    "obv_z_20",
    "amihud_20",
    "atr_14",
    "gap_1",
]

MARKET_FEATURE_COLUMNS = [f"mkt_{c}" for c in BASE_FEATURE_COLUMNS]


def _safe_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std().replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    close = out["close"]
    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].replace(0, np.nan)
    amount = out["amount"].replace(0, np.nan)

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    out["trend_5_20"] = ma5 / ma20 - 1.0
    out["trend_20_60"] = ma20 / ma60 - 1.0

    out["volatility_20"] = out["ret_1"].rolling(20).std()
    out["volatility_60"] = out["ret_1"].rolling(60).std()

    high20 = close.rolling(20).max()
    low20 = close.rolling(20).min()
    out["drawdown_20"] = close / high20 - 1.0
    out["price_pos_20"] = (close - low20) / (high20 - low20 + 1e-9)

    out["vol_ratio_20"] = volume / (volume.rolling(20).mean() + 1e-9)
    out["vol_conc_5_20"] = volume.rolling(5).sum() / (volume.rolling(20).sum() + 1e-9)

    signed_volume = np.sign(out["ret_1"].fillna(0.0)) * volume.fillna(0.0)
    obv = signed_volume.cumsum()
    out["obv_z_20"] = (obv - obv.rolling(20).mean()) / (_safe_std(obv, 20) + 1e-9)

    out["amihud_20"] = (out["ret_1"].abs() / (amount + 1e-9)).rolling(20).mean() * 1e8

    tr = pd.concat(
        [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / (close + 1e-9)

    out["gap_1"] = open_ / close.shift(1) - 1.0

    out["fwd_ret_1"] = close.shift(-1) / close - 1.0
    out["fwd_ret_20"] = close.shift(-20) / close - 1.0
    out["target_1d_up"] = np.where(out["fwd_ret_1"].notna(), (out["fwd_ret_1"] > 0).astype(float), np.nan)
    out["target_20d_up"] = np.where(out["fwd_ret_20"].notna(), (out["fwd_ret_20"] > 0).astype(float), np.nan)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def make_market_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = build_features(df)
    keep_cols = ["date"] + BASE_FEATURE_COLUMNS + ["target_1d_up", "target_20d_up"]
    mkt = feat[keep_cols].copy()
    rename_map = {c: f"mkt_{c}" for c in BASE_FEATURE_COLUMNS}
    rename_map.update({"target_1d_up": "mkt_target_1d_up", "target_20d_up": "mkt_target_20d_up"})
    return mkt.rename(columns=rename_map)


def make_stock_feature_frame(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    stock_feat = build_features(stock_df)
    merged = stock_feat.merge(market_df, on="date", how="left", validate="m:1")
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged


def stock_feature_columns() -> List[str]:
    return BASE_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS

