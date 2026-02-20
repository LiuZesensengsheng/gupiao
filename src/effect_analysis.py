from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .data import DataError, load_symbol_daily, normalize_symbol
from .features import build_features
from .pipeline import Security


@dataclass
class EffectSummary:
    sample_size: int
    win_rate_1d: float
    win_rate_5d: float
    strong_rate_5d: float
    median_ret_1d: float
    median_ret_5d: float
    loss_rate_1d: float
    deep_loss_rate: float
    median_drawdown_20: float
    avg_price_pos_20: float
    avg_vol_ratio_20: float
    avg_obv_z_20: float
    avg_vol_conc_5_20: float
    money_score: float
    chip_score: float
    money_label: str
    chip_label: str
    pnl_label: str
    risk_label: str


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _level_label(score: float) -> str:
    if score >= 0.35:
        return "偏强"
    if score <= -0.35:
        return "偏弱"
    return "中性"


def build_latest_snapshot(
    source: str,
    data_dir: str,
    start: str,
    end: str,
    stocks: Sequence[Security],
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    needed = [
        "ret_1",
        "ret_5",
        "ret_20",
        "drawdown_20",
        "price_pos_20",
        "vol_ratio_20",
        "vol_conc_5_20",
        "obv_z_20",
        "trend_5_20",
        "trend_20_60",
        "amihud_20",
    ]

    for sec in stocks:
        symbol = normalize_symbol(sec.symbol).symbol
        try:
            raw = load_symbol_daily(symbol=symbol, source=source, data_dir=data_dir, start=start, end=end)
        except DataError:
            continue
        feat = build_features(raw)
        valid = feat.dropna(subset=needed).sort_values("date")
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        rows.append(
            {
                "symbol": symbol,
                "name": sec.name,
                "sector": sector_map.get(symbol, "其他"),
                "date": pd.Timestamp(latest["date"]),
                "ret_1": float(latest["ret_1"]),
                "ret_5": float(latest["ret_5"]),
                "ret_20": float(latest["ret_20"]),
                "drawdown_20": float(latest["drawdown_20"]),
                "price_pos_20": float(latest["price_pos_20"]),
                "vol_ratio_20": float(latest["vol_ratio_20"]),
                "vol_conc_5_20": float(latest["vol_conc_5_20"]),
                "obv_z_20": float(latest["obv_z_20"]),
                "trend_5_20": float(latest["trend_5_20"]),
                "trend_20_60": float(latest["trend_20_60"]),
                "amihud_20": float(latest["amihud_20"]),
            }
        )
    return pd.DataFrame(rows)


def compute_effect_summary(snapshot: pd.DataFrame) -> EffectSummary:
    if snapshot.empty:
        return EffectSummary(
            sample_size=0,
            win_rate_1d=np.nan,
            win_rate_5d=np.nan,
            strong_rate_5d=np.nan,
            median_ret_1d=np.nan,
            median_ret_5d=np.nan,
            loss_rate_1d=np.nan,
            deep_loss_rate=np.nan,
            median_drawdown_20=np.nan,
            avg_price_pos_20=np.nan,
            avg_vol_ratio_20=np.nan,
            avg_obv_z_20=np.nan,
            avg_vol_conc_5_20=np.nan,
            money_score=np.nan,
            chip_score=np.nan,
            money_label="NA",
            chip_label="NA",
            pnl_label="NA",
            risk_label="NA",
        )

    ret1 = snapshot["ret_1"]
    ret5 = snapshot["ret_5"]
    drawdown = snapshot["drawdown_20"]
    price_pos = snapshot["price_pos_20"]
    vol_ratio = snapshot["vol_ratio_20"]
    obv_z = snapshot["obv_z_20"]
    vol_conc = snapshot["vol_conc_5_20"]

    win_rate_1d = float((ret1 > 0).mean())
    win_rate_5d = float((ret5 > 0).mean())
    strong_rate_5d = float((ret5 > 0.03).mean())
    loss_rate_1d = float((ret1 < 0).mean())
    deep_loss_rate = float(((ret1 < -0.03) | (drawdown < -0.10)).mean())

    money_score = _clip(0.75 * (vol_ratio.mean() - 1.0) + 0.25 * (obv_z.mean() / 2.0), -1.0, 1.0)
    chip_score = _clip(
        0.55 * ((price_pos.mean() - 0.5) * 2.0)
        + 0.30 * (vol_conc.mean() - 0.25) * 3.0
        + 0.15 * (drawdown.mean() + 0.08) * 3.0,
        -1.0,
        1.0,
    )

    pnl_score = _clip(0.6 * (win_rate_1d - 0.5) * 2 + 0.4 * (win_rate_5d - 0.5) * 2, -1.0, 1.0)
    risk_score = _clip(0.7 * deep_loss_rate + 0.3 * max(0.0, -drawdown.median()), 0.0, 1.0)

    if risk_score >= 0.45:
        risk_label = "高"
    elif risk_score >= 0.25:
        risk_label = "中"
    else:
        risk_label = "低"

    return EffectSummary(
        sample_size=int(len(snapshot)),
        win_rate_1d=win_rate_1d,
        win_rate_5d=win_rate_5d,
        strong_rate_5d=strong_rate_5d,
        median_ret_1d=float(ret1.median()),
        median_ret_5d=float(ret5.median()),
        loss_rate_1d=loss_rate_1d,
        deep_loss_rate=deep_loss_rate,
        median_drawdown_20=float(drawdown.median()),
        avg_price_pos_20=float(price_pos.mean()),
        avg_vol_ratio_20=float(vol_ratio.mean()),
        avg_obv_z_20=float(obv_z.mean()),
        avg_vol_conc_5_20=float(vol_conc.mean()),
        money_score=money_score,
        chip_score=chip_score,
        money_label=_level_label(money_score),
        chip_label=_level_label(chip_score),
        pnl_label=_level_label(pnl_score),
        risk_label=risk_label,
    )


def compute_sector_table(snapshot: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=["sector", "heat_score", "win_rate_1d", "median_ret_5d", "money_score", "chip_score", "count"])

    grouped = snapshot.groupby("sector", dropna=False)
    rows: List[Dict[str, object]] = []
    for sector, g in grouped:
        heat = _clip(
            0.35 * g["ret_5"].mean()
            + 0.25 * g["trend_5_20"].mean()
            + 0.20 * (g["vol_ratio_20"].mean() - 1.0)
            + 0.20 * (g["obv_z_20"].mean() / 2.0),
            -1.0,
            1.0,
        )
        money = _clip(0.7 * (g["vol_ratio_20"].mean() - 1.0) + 0.3 * (g["obv_z_20"].mean() / 2.0), -1.0, 1.0)
        chip = _clip(0.7 * ((g["price_pos_20"].mean() - 0.5) * 2) + 0.3 * (g["vol_conc_5_20"].mean() - 0.25) * 3, -1.0, 1.0)
        rows.append(
            {
                "sector": str(sector),
                "heat_score": float(heat),
                "win_rate_1d": float((g["ret_1"] > 0).mean()),
                "median_ret_5d": float(g["ret_5"].median()),
                "money_score": float(money),
                "chip_score": float(chip),
                "count": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values("heat_score", ascending=False).reset_index(drop=True)
    return out.head(top_k)

