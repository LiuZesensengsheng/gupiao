from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.infrastructure.modeling import LogisticBinaryModel
from src.infrastructure.sector_features import SECTOR_FEATURE_COLUMNS, make_sector_feature_frame


@dataclass(frozen=True)
class SectorForecastRecord:
    sector: str
    latest_date: pd.Timestamp
    up_5d_prob: float
    up_20d_prob: float
    excess_vs_market_prob: float
    rotation_speed: float
    crowding_score: float
    tradeability_score: float


def _fit_prob(
    frame: pd.DataFrame,
    *,
    target_col: str,
    l2: float,
) -> float:
    valid = frame.dropna(subset=SECTOR_FEATURE_COLUMNS + [target_col]).sort_values("date").copy()
    if len(valid) < 60:
        latest = frame.sort_values("date").iloc[-1]
        if target_col == "sector_target_20d_excess":
            raw = 0.5 + 0.5 * float(latest.get("sec_ret_20_minus_mkt", 0.0))
        elif target_col == "sector_target_20d_up":
            raw = 0.5 + 0.5 * float(latest.get("sec_ret_20", 0.0))
        else:
            raw = 0.5 + 0.5 * float(latest.get("sec_ret_5", 0.0))
        return float(np.clip(raw, 0.05, 0.95))

    model = LogisticBinaryModel(l2=float(l2)).fit(valid, SECTOR_FEATURE_COLUMNS, target_col)
    latest = valid.dropna(subset=SECTOR_FEATURE_COLUMNS).sort_values("date").iloc[[-1]]
    return float(model.predict_proba(latest, SECTOR_FEATURE_COLUMNS)[0])


def run_sector_forecast(
    *,
    sector_frames: dict[str, pd.DataFrame],
    market_raw: pd.DataFrame,
    l2: float,
) -> list[SectorForecastRecord]:
    out: list[SectorForecastRecord] = []
    for sector, frame in sector_frames.items():
        if frame is None or frame.empty:
            continue
        feat = make_sector_feature_frame(frame, market_raw)
        valid = feat.dropna(subset=SECTOR_FEATURE_COLUMNS).sort_values("date")
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        up_5d_prob = _fit_prob(feat, target_col="sector_target_5d_up", l2=l2)
        up_20d_prob = _fit_prob(feat, target_col="sector_target_20d_up", l2=l2)
        excess_prob = _fit_prob(feat, target_col="sector_target_20d_excess", l2=l2)
        rotation_speed = float(np.clip(abs(float(latest.get("sec_ret_5", 0.0)) - float(latest.get("sec_ret_20", 0.0))), 0.0, 1.0))
        crowding_score = float(np.clip(float(latest.get("sec_price_pos_20", 0.0)), 0.0, 1.0))
        tradeability_score = float(np.clip(1.0 - abs(float(latest.get("sec_amihud_20", 0.0))) / 10.0, 0.0, 1.0))
        out.append(
            SectorForecastRecord(
                sector=str(sector),
                latest_date=pd.Timestamp(latest["date"]),
                up_5d_prob=float(up_5d_prob),
                up_20d_prob=float(up_20d_prob),
                excess_vs_market_prob=float(excess_prob),
                rotation_speed=rotation_speed,
                crowding_score=crowding_score,
                tradeability_score=tradeability_score,
            )
        )
    out.sort(key=lambda item: (item.up_20d_prob, item.excess_vs_market_prob), reverse=True)
    return out
