from __future__ import annotations

import pandas as pd

from src.infrastructure.features import build_features


SECTOR_FEATURE_COLUMNS = [
    "sec_ret_1",
    "sec_ret_5",
    "sec_ret_20",
    "sec_trend_5_20",
    "sec_trend_20_60",
    "sec_volatility_20",
    "sec_drawdown_20",
    "sec_price_pos_20",
    "sec_vol_ratio_20",
    "sec_amihud_20",
    "sec_coverage_z20",
    "sec_ret_20_minus_mkt",
    "sec_trend_20_60_minus_mkt",
    "sec_corr_mkt_20",
]


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).corr(b)


def make_sector_feature_frame(
    sector_df: pd.DataFrame,
    market_raw: pd.DataFrame,
) -> pd.DataFrame:
    sec_feat = build_features(sector_df)
    mkt_feat = build_features(market_raw)
    market_keep = mkt_feat[
        ["date", "ret_1", "ret_20", "trend_20_60", "fwd_ret_20"]
    ].copy()
    market_keep = market_keep.rename(
        columns={
            "ret_1": "mkt_ret_1",
            "ret_20": "mkt_ret_20",
            "trend_20_60": "mkt_trend_20_60",
            "fwd_ret_20": "mkt_fwd_ret_20",
        }
    )
    merged = sec_feat.merge(market_keep, on="date", how="left", validate="m:1")

    coverage = pd.to_numeric(merged.get("coverage", 0.0), errors="coerce").astype(float)
    coverage_std = coverage.rolling(20).std().replace(0.0, float("nan"))
    merged["sec_coverage_z20"] = (coverage - coverage.rolling(20).mean()) / (coverage_std + 1e-9)
    merged["sec_ret_20_minus_mkt"] = merged["ret_20"] - merged["mkt_ret_20"]
    merged["sec_trend_20_60_minus_mkt"] = merged["trend_20_60"] - merged["mkt_trend_20_60"]
    merged["sec_corr_mkt_20"] = _rolling_corr(merged["ret_1"], merged["mkt_ret_1"], 20)

    rename_map = {
        "ret_1": "sec_ret_1",
        "ret_5": "sec_ret_5",
        "ret_20": "sec_ret_20",
        "trend_5_20": "sec_trend_5_20",
        "trend_20_60": "sec_trend_20_60",
        "volatility_20": "sec_volatility_20",
        "drawdown_20": "sec_drawdown_20",
        "price_pos_20": "sec_price_pos_20",
        "vol_ratio_20": "sec_vol_ratio_20",
        "amihud_20": "sec_amihud_20",
    }
    merged = merged.rename(columns=rename_map)
    merged["sector_target_5d_up"] = (merged["fwd_ret_1"].rolling(5).sum().shift(-4) > 0.0).astype(float)
    merged.loc[merged["fwd_ret_1"].rolling(5).sum().shift(-4).isna(), "sector_target_5d_up"] = pd.NA
    merged["sector_target_20d_up"] = (merged["fwd_ret_20"] > 0.0).astype(float)
    merged.loc[merged["fwd_ret_20"].isna(), "sector_target_20d_up"] = pd.NA
    merged["sector_target_20d_excess"] = (merged["fwd_ret_20"] > merged["mkt_fwd_ret_20"]).astype(float)
    merged.loc[
        merged["fwd_ret_20"].isna() | merged["mkt_fwd_ret_20"].isna(),
        "sector_target_20d_excess",
    ] = pd.NA
    merged.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    for col in SECTOR_FEATURE_COLUMNS:
        if col not in merged.columns:
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill()
        if merged[col].isna().any():
            median = merged[col].median(skipna=True)
            merged[col] = merged[col].fillna(0.0 if pd.isna(median) else float(median))
    keep_cols = ["date"] + SECTOR_FEATURE_COLUMNS + [
        "sector_target_5d_up",
        "sector_target_20d_up",
        "sector_target_20d_excess",
    ]
    return merged[keep_cols].copy()
