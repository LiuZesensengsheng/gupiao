from __future__ import annotations

import numpy as np
import pandas as pd


def build_date_slice_index(
    frame: pd.DataFrame,
    *,
    sort_cols: list[str],
) -> tuple[pd.DataFrame, dict[pd.Timestamp, tuple[int, int]]]:
    if frame.empty:
        return frame.copy(), {}
    work = frame.sort_values(sort_cols).reset_index(drop=True).copy()
    dates = pd.to_datetime(work["date"]).to_numpy()
    bounds: dict[pd.Timestamp, tuple[int, int]] = {}
    start = 0
    n = len(work)
    while start < n:
        date = pd.Timestamp(dates[start])
        end = start + 1
        while end < n and pd.Timestamp(dates[end]) == date:
            end += 1
        bounds[date] = (start, end)
        start = end
    return work, bounds


def distributional_score(
    *,
    short_prob: float,
    two_prob: float,
    three_prob: float,
    five_prob: float,
    mid_prob: float,
    short_expected_ret: float,
    mid_expected_ret: float,
) -> float:
    base_score = float(
        0.14 * float(short_prob)
        + 0.18 * float(two_prob)
        + 0.22 * float(three_prob)
        + 0.28 * float(five_prob)
        + 0.18 * float(mid_prob)
    )
    short_ret_score = float(np.clip(0.5 + float(short_expected_ret) / 0.06, 0.0, 1.0))
    two_ret_score = float(
        np.clip(0.5 + (0.75 * float(short_expected_ret) + 0.25 * float(mid_expected_ret)) / 0.08, 0.0, 1.0)
    )
    three_ret_score = float(
        np.clip(0.5 + (0.60 * float(short_expected_ret) + 0.40 * float(mid_expected_ret)) / 0.10, 0.0, 1.0)
    )
    five_ret_score = float(
        np.clip(0.5 + (0.35 * float(short_expected_ret) + 0.65 * float(mid_expected_ret)) / 0.12, 0.0, 1.0)
    )
    mid_ret_score = float(np.clip(0.5 + float(mid_expected_ret) / 0.20, 0.0, 1.0))
    dist_score = float(
        0.14 * short_ret_score
        + 0.18 * two_ret_score
        + 0.22 * three_ret_score
        + 0.24 * five_ret_score
        + 0.22 * mid_ret_score
    )
    return float(0.4 * base_score + 0.6 * dist_score)


def panel_slice_metrics(
    scored_rows: pd.DataFrame,
    *,
    realized_col: str = "realized_ret_20d",
    top_k: int = 3,
) -> tuple[float, float, float, float]:
    if scored_rows.empty:
        return 0.0, 0.0, 0.0, 0.0
    frame = scored_rows.dropna(subset=["score", realized_col]).copy()
    if len(frame) < 2:
        return 0.0, 0.0, 0.0, 0.0
    rank_ic = float(frame["score"].corr(frame[realized_col], method="spearman"))
    if rank_ic != rank_ic:
        rank_ic = 0.0
    bucket_n = max(1, int(np.ceil(len(frame) * 0.1)))
    ranked = frame.sort_values("score", ascending=False).reset_index(drop=True)
    top_bucket = ranked.head(bucket_n)
    bottom_bucket = ranked.tail(bucket_n)
    top_decile_return = float(top_bucket[realized_col].mean()) if not top_bucket.empty else 0.0
    bottom_return = float(bottom_bucket[realized_col].mean()) if not bottom_bucket.empty else 0.0
    top_bottom_spread = float(top_decile_return - bottom_return)
    top_k_n = max(1, min(int(top_k), len(ranked)))
    top_k_hit_rate = float((ranked.head(top_k_n)[realized_col] > 0.0).mean())
    return rank_ic, top_decile_return, top_bottom_spread, top_k_hit_rate


def panel_horizon_metrics(scored_rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping = {
        "1d": "realized_ret_1d",
        "2d": "realized_ret_2d",
        "3d": "realized_ret_3d",
        "5d": "realized_ret_5d",
        "20d": "realized_ret_20d",
    }
    out: dict[str, dict[str, float]] = {}
    for horizon, realized_col in mapping.items():
        rank_ic, top_decile_ret, top_bottom_spread, top_k_hit_rate = panel_slice_metrics(
            scored_rows,
            realized_col=realized_col,
        )
        out[horizon] = {
            "rank_ic": float(rank_ic),
            "top_decile_return": float(top_decile_ret),
            "top_bottom_spread": float(top_bottom_spread),
            "top_k_hit_rate": float(top_k_hit_rate),
        }
    return out
