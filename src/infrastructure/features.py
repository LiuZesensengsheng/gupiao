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
    "bear_body_1",
    "upper_shadow_ratio_1",
    "lower_shadow_ratio_1",
    "body_ratio_1",
    "up_streak_3",
    "down_streak_3",
    "narrow_range_rank_20",
    "range_contraction_5",
    "breakout_above_20_high",
    "breakdown_below_20_low",
    "distance_to_20d_high",
    "distance_to_20d_low",
    "volume_breakout_ratio",
    "squeeze_breakout_score",
    "breakout_quality_score",
    "exhaustion_reversal_risk",
    "pullback_reclaim_score",
    "hvbd_recent_5",
]

MARKET_FEATURE_COLUMNS = [f"mkt_{c}" for c in BASE_FEATURE_COLUMNS]

SHORT_RETURN_BUCKET_EDGES = [-float("inf"), -0.02, -0.005, 0.005, 0.02, float("inf")]
MID_RETURN_BUCKET_EDGES = [-float("inf"), -0.10, -0.03, 0.03, 0.10, float("inf")]


def _safe_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std().replace(0, np.nan)


def _rolling_last_rank_pct(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(
        lambda values: float(pd.Series(values).rank(pct=True).iloc[-1]),
        raw=False,
    )


def _capped_run_length(flags: pd.Series, cap: int) -> pd.Series:
    out = np.zeros(len(flags), dtype=float)
    run = 0
    arr = flags.fillna(False).astype(bool).to_numpy()
    for idx, flag in enumerate(arr):
        if flag:
            run = min(cap, run + 1)
        else:
            run = 0
        out[idx] = float(run)
    return pd.Series(out, index=flags.index, dtype=float)


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

    candle_range = (high - low).abs()
    candle_range_safe = candle_range.replace(0, np.nan)
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0.0)
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low).clip(lower=0.0)
    body_abs = (close - open_).abs()
    out["upper_shadow_ratio_1"] = upper_shadow / (candle_range_safe + 1e-9)
    out["lower_shadow_ratio_1"] = lower_shadow / (candle_range_safe + 1e-9)
    out["body_ratio_1"] = body_abs / (candle_range_safe + 1e-9)

    up_close = close > close.shift(1)
    down_close = close < close.shift(1)
    out["up_streak_3"] = _capped_run_length(up_close, 3) / 3.0
    out["down_streak_3"] = _capped_run_length(down_close, 3) / 3.0

    out["narrow_range_rank_20"] = _rolling_last_rank_pct(candle_range_safe, 20)
    out["range_contraction_5"] = candle_range_safe.rolling(5).mean() / (candle_range_safe.rolling(20).mean() + 1e-9)

    prev_high_20 = close.shift(1).rolling(20).max()
    prev_low_20 = close.shift(1).rolling(20).min()
    out["breakout_above_20_high"] = np.where(
        prev_high_20.notna(),
        (close > prev_high_20).astype(float),
        np.nan,
    )
    out["breakdown_below_20_low"] = np.where(
        prev_low_20.notna(),
        (close < prev_low_20).astype(float),
        np.nan,
    )
    out["distance_to_20d_high"] = (close - prev_high_20) / (prev_high_20 + 1e-9)
    out["distance_to_20d_low"] = (close - prev_low_20) / (prev_low_20 + 1e-9)
    out["volume_breakout_ratio"] = (
        out["vol_ratio_20"] * np.where(out["breakout_above_20_high"].fillna(0.0) > 0.5, 1.0, 0.0)
    )
    close_near_high = (close - low) / (candle_range_safe + 1e-9)
    close_near_low = (high - close) / (candle_range_safe + 1e-9)
    squeeze_component = np.clip((1.20 - out["range_contraction_5"]) / 0.60, 0.0, 1.5)
    volume_component = np.clip(out["vol_ratio_20"] / 2.0, 0.0, 2.0)
    out["squeeze_breakout_score"] = (
        out["breakout_above_20_high"].fillna(0.0)
        * squeeze_component
        * volume_component
    )
    out["breakout_quality_score"] = (
        out["breakout_above_20_high"].fillna(0.0)
        * (
            0.40 * np.clip(out["volume_breakout_ratio"], 0.0, 3.0)
            + 0.35 * np.clip(out["body_ratio_1"], 0.0, 1.0)
            + 0.25 * np.clip(close_near_high, 0.0, 1.0)
        )
    )
    out["exhaustion_reversal_risk"] = (
        np.clip(out["upper_shadow_ratio_1"], 0.0, 1.0)
        * np.clip(out["price_pos_20"], 0.0, 1.0)
        * np.clip(out["vol_ratio_20"], 0.0, 3.0)
        * np.clip(0.4 + 0.6 * close_near_low, 0.0, 1.0)
    )
    out["pullback_reclaim_score"] = (
        np.clip(out["lower_shadow_ratio_1"], 0.0, 1.0)
        * np.clip(out["body_ratio_1"], 0.0, 1.0)
        * np.clip(1.0 - out["drawdown_20"].abs(), 0.0, 1.0)
        * np.clip(0.5 + 0.5 * out["up_streak_3"] - 0.3 * out["down_streak_3"], 0.0, 1.0)
    )

    out["gap_1"] = open_ / close.shift(1) - 1.0
    out["bear_body_1"] = np.clip((open_ - close) / (open_ + 1e-9), -0.25, 0.25)
    hvbd_today = (
        (out["ret_1"] <= -0.04)
        & (out["vol_ratio_20"] >= 2.0)
        & (out["price_pos_20"].shift(1) >= 0.75)
    ).astype(float)
    out["hvbd_recent_5"] = hvbd_today.rolling(5).max()

    out["fwd_ret_1"] = close.shift(-1) / close - 1.0
    out["fwd_ret_2"] = close.shift(-2) / close - 1.0
    out["fwd_ret_3"] = close.shift(-3) / close - 1.0
    out["fwd_ret_5"] = close.shift(-5) / close - 1.0
    out["fwd_ret_20"] = close.shift(-20) / close - 1.0
    out["target_1d_up"] = np.where(out["fwd_ret_1"].notna(), (out["fwd_ret_1"] > 0).astype(float), np.nan)
    out["target_2d_up"] = np.where(out["fwd_ret_2"].notna(), (out["fwd_ret_2"] > 0).astype(float), np.nan)
    out["target_3d_up"] = np.where(out["fwd_ret_3"].notna(), (out["fwd_ret_3"] > 0).astype(float), np.nan)
    out["target_5d_up"] = np.where(out["fwd_ret_5"].notna(), (out["fwd_ret_5"] > 0).astype(float), np.nan)
    out["target_20d_up"] = np.where(out["fwd_ret_20"].notna(), (out["fwd_ret_20"] > 0).astype(float), np.nan)
    out["target_1d_bucket"] = pd.cut(
        out["fwd_ret_1"],
        bins=SHORT_RETURN_BUCKET_EDGES,
        labels=False,
        include_lowest=True,
    ).astype("Float64")
    out["target_20d_bucket"] = pd.cut(
        out["fwd_ret_20"],
        bins=MID_RETURN_BUCKET_EDGES,
        labels=False,
        include_lowest=True,
    ).astype("Float64")

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def make_market_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = build_features(df)
    keep_cols = [
        "date",
        *BASE_FEATURE_COLUMNS,
        "fwd_ret_1",
        "fwd_ret_2",
        "fwd_ret_3",
        "fwd_ret_5",
        "fwd_ret_20",
        "target_1d_up",
        "target_2d_up",
        "target_3d_up",
        "target_5d_up",
        "target_20d_up",
        "target_1d_bucket",
        "target_20d_bucket",
    ]
    mkt = feat[keep_cols].copy()
    rename_map = {c: f"mkt_{c}" for c in BASE_FEATURE_COLUMNS}
    rename_map.update(
        {
            "fwd_ret_1": "mkt_fwd_ret_1",
            "fwd_ret_2": "mkt_fwd_ret_2",
            "fwd_ret_3": "mkt_fwd_ret_3",
            "fwd_ret_5": "mkt_fwd_ret_5",
            "fwd_ret_20": "mkt_fwd_ret_20",
            "target_1d_up": "mkt_target_1d_up",
            "target_2d_up": "mkt_target_2d_up",
            "target_3d_up": "mkt_target_3d_up",
            "target_5d_up": "mkt_target_5d_up",
            "target_20d_up": "mkt_target_20d_up",
            "target_1d_bucket": "mkt_target_1d_bucket",
            "target_20d_bucket": "mkt_target_20d_bucket",
        }
    )
    return mkt.rename(columns=rename_map)


def make_stock_feature_frame(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    stock_feat = build_features(stock_df)
    merged = stock_feat.merge(market_df, on="date", how="left", validate="m:1")
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged


def stock_feature_columns(
    extra_market_cols: List[str] | None = None,
    extra_stock_cols: List[str] | None = None,
) -> List[str]:
    cols = BASE_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS
    if extra_market_cols:
        cols = cols + [str(c) for c in extra_market_cols]
    if extra_stock_cols:
        cols = cols + [str(c) for c in extra_stock_cols]
    return cols
