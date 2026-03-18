from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.application.v2_contracts import (
    CapitalFlowState,
    CrossSectionForecastState,
    HorizonForecast,
    InfoAggregateState,
    MacroContextState,
    MarketForecastState,
    MarketSentimentState,
    StockForecastState,
)

_HORIZON_SCALE = {
    "1d": 0.035,
    "2d": 0.050,
    "3d": 0.065,
    "5d": 0.095,
    "10d": 0.145,
    "20d": 0.220,
}


@dataclass(frozen=True)
class ForecastSupportDependencies:
    clip: Callable[[float, float, float], float]
    return_quantile_profile_cls: type


def blend_quantile_profiles(
    left: object,
    right: object,
    *,
    left_weight: float,
    deps: ForecastSupportDependencies,
) -> object:
    w = float(np.clip(left_weight, 0.0, 1.0))
    r = 1.0 - w
    cls = deps.return_quantile_profile_cls
    return cls(
        expected_return=float(w * float(getattr(left, "expected_return")) + r * float(getattr(right, "expected_return"))),
        q10=float(w * float(getattr(left, "q10")) + r * float(getattr(right, "q10"))),
        q30=float(w * float(getattr(left, "q30")) + r * float(getattr(right, "q30"))),
        q20=float(w * float(getattr(left, "q20")) + r * float(getattr(right, "q20"))),
        q50=float(w * float(getattr(left, "q50")) + r * float(getattr(right, "q50"))),
        q70=float(w * float(getattr(left, "q70")) + r * float(getattr(right, "q70"))),
        q80=float(w * float(getattr(left, "q80")) + r * float(getattr(right, "q80"))),
        q90=float(w * float(getattr(left, "q90")) + r * float(getattr(right, "q90"))),
    )


def synthetic_quantile_profile(
    *,
    prob: float,
    horizon_key: str,
    deps: ForecastSupportDependencies,
) -> object:
    scale = float(_HORIZON_SCALE.get(str(horizon_key), 0.08))
    expected = float(np.clip((float(prob) - 0.5) * 1.2 * scale, -scale, scale))
    spread = max(scale * 0.55, 0.01)
    q50 = expected
    q10 = expected - 0.85 * spread
    q30 = expected - 0.40 * spread
    q70 = expected + 0.40 * spread
    q90 = expected + 0.85 * spread
    cls = deps.return_quantile_profile_cls
    return cls(
        expected_return=float(expected),
        q10=float(q10),
        q30=float(q30),
        q20=float(0.5 * (q10 + q30)),
        q50=float(q50),
        q70=float(q70),
        q80=float(0.5 * (q70 + q90)),
        q90=float(q90),
    )


def intrinsic_confidence(
    *,
    up_prob: float,
    horizon_probs: dict[str, float],
    info_state: InfoAggregateState | None = None,
    calibration_prior: dict[str, float] | None = None,
    tradability_status: str = "normal",
) -> tuple[float, str]:
    p = float(up_prob)
    probability_edge = float(np.clip(abs(p - 0.5) / 0.22, 0.0, 1.0))
    local_probs = [float(v) for v in horizon_probs.values() if v == v]
    if local_probs:
        dispersion = float(np.std(local_probs))
        consistency = float(np.clip(1.0 - dispersion / 0.12, 0.0, 1.0))
    else:
        consistency = 0.5
    coverage = float(getattr(info_state, "coverage_confidence", 0.55) if info_state is not None else 0.55)
    calibration = 0.58
    if calibration_prior:
        hit_rate = float(calibration_prior.get("top_k_hit_rate", calibration_prior.get("hit_rate", 0.55)))
        rank_ic = float(calibration_prior.get("rank_ic", 0.05))
        calibration = float(np.clip(0.55 * hit_rate + 0.45 * (0.5 + rank_ic), 0.0, 1.0))
    status_penalty = 0.0
    if tradability_status == "data_insufficient":
        status_penalty = 0.18
    elif tradability_status in {"halted", "delisted"}:
        status_penalty = 0.40
    confidence = float(
        np.clip(
            0.34 * calibration
            + 0.28 * consistency
            + 0.20 * coverage
            + 0.18 * probability_edge
            - status_penalty,
            0.05,
            0.98,
        )
    )
    if confidence >= 0.78:
        tone = "高"
    elif confidence >= 0.62:
        tone = "中高"
    elif confidence >= 0.46:
        tone = "中"
    else:
        tone = "偏低"
    return confidence, f"{tone}置信度，综合了校准稳定性、信号一致性和样本覆盖。"


def build_horizon_forecasts(
    *,
    latest_close: float,
    horizon_probs: dict[str, float],
    short_profile: object | None,
    mid_profile: object | None,
    info_state: InfoAggregateState | None = None,
    calibration_priors: dict[str, dict[str, float]] | None = None,
    tradability_status: str = "normal",
    deps: ForecastSupportDependencies,
) -> dict[str, HorizonForecast]:
    base_short = short_profile or synthetic_quantile_profile(prob=float(horizon_probs.get("1d", 0.5)), horizon_key="1d", deps=deps)
    base_mid = mid_profile or synthetic_quantile_profile(prob=float(horizon_probs.get("20d", 0.5)), horizon_key="20d", deps=deps)
    five_profile = blend_quantile_profiles(base_short, base_mid, left_weight=0.35, deps=deps)
    profile_map = {
        "1d": base_short,
        "2d": blend_quantile_profiles(base_short, five_profile, left_weight=0.75, deps=deps),
        "3d": blend_quantile_profiles(base_short, five_profile, left_weight=0.60, deps=deps),
        "5d": five_profile,
        "10d": blend_quantile_profiles(five_profile, base_mid, left_weight=0.45, deps=deps),
        "20d": base_mid,
    }
    labels = {
        "1d": "未来1日",
        "2d": "未来2日",
        "3d": "未来3日",
        "5d": "未来5日",
        "10d": "未来10日",
        "20d": "未来20日",
    }
    out: dict[str, HorizonForecast] = {}
    for key, label in labels.items():
        profile = profile_map[key]
        up_prob = float(horizon_probs.get(key, 0.5))
        confidence, reason = intrinsic_confidence(
            up_prob=up_prob,
            horizon_probs=horizon_probs,
            info_state=info_state,
            calibration_prior=None if calibration_priors is None else calibration_priors.get(key),
            tradability_status=tradability_status,
        )
        base_price = float(latest_close)
        price_low = np.nan if base_price != base_price else float(base_price * (1.0 + float(getattr(profile, "q10"))))
        price_mid = np.nan if base_price != base_price else float(base_price * (1.0 + float(getattr(profile, "q50"))))
        price_high = np.nan if base_price != base_price else float(base_price * (1.0 + float(getattr(profile, "q90"))))
        out[key] = HorizonForecast(
            horizon_days=int(key.rstrip("d")),
            label=label,
            up_prob=up_prob,
            expected_return=float(getattr(profile, "expected_return")),
            q10=float(getattr(profile, "q10")),
            q50=float(getattr(profile, "q50")),
            q90=float(getattr(profile, "q90")),
            price_low=price_low,
            price_mid=price_mid,
            price_high=price_high,
            confidence=float(confidence),
            confidence_reason=str(reason),
        )
    return out


def sentiment_stage(score: float) -> str:
    if score >= 78.0:
        return "过热"
    if score >= 64.0:
        return "亢奋"
    if score >= 54.0:
        return "回暖"
    if score >= 42.0:
        return "修复"
    return "冰点"


def pct_text(value: float, *, signed: bool = False) -> str:
    return f"{float(value) * 100:+.1f}%" if signed else f"{float(value) * 100:.1f}%"


def num_text(value: float, digits: int = 2, *, signed: bool = False) -> str:
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def build_market_sentiment_state(
    *,
    market: MarketForecastState,
    cross_section: CrossSectionForecastState,
    capital_flow: CapitalFlowState,
    macro: MacroContextState,
) -> MarketSentimentState:
    facts = market.market_facts
    advance_balance = float(np.clip((facts.advancers - facts.decliners) / max(1, facts.sample_coverage), -1.0, 1.0))
    limit_balance = float(
        np.clip((facts.limit_up_count - facts.limit_down_count) / max(1, facts.sample_coverage), -1.0, 1.0)
    )
    high_low_balance = float(
        np.clip((facts.new_high_count - facts.new_low_count) / max(1, facts.sample_coverage), -1.0, 1.0)
    )
    score = float(
        np.clip(
            50.0
            + 14.0 * float(cross_section.breadth_strength)
            + 12.0 * advance_balance
            + 8.0 * limit_balance
            + 8.0 * high_low_balance
            + 10.0 * (float(capital_flow.turnover_heat) - 0.5)
            + 6.0 * float(capital_flow.margin_balance_change)
            + 4.0 * float(capital_flow.northbound_net_flow)
            + 6.0 * (float(market.up_5d_prob) - 0.5)
            - 10.0 * float(market.drawdown_risk)
            - 6.0 * max(0.0, float(macro.fx_pressure))
            - 5.0 * max(0.0, float(macro.commodity_pressure)),
            0.0,
            100.0,
        )
    )
    drivers: list[tuple[float, str]] = [
        (abs(float(cross_section.breadth_strength)), f"市场宽度强度 {pct_text(float(cross_section.breadth_strength), signed=True)}"),
        (abs(advance_balance), f"涨跌家数差 {facts.advancers}/{facts.decliners}"),
        (abs(limit_balance), f"涨跌停差 {facts.limit_up_count}/{facts.limit_down_count}"),
        (abs(high_low_balance), f"新高/新低 {facts.new_high_count}/{facts.new_low_count}"),
        (abs(float(capital_flow.turnover_heat) - 0.5), f"成交热度 {pct_text(float(capital_flow.turnover_heat))}"),
        (abs(float(capital_flow.margin_balance_change)), f"两融变化 {pct_text(float(capital_flow.margin_balance_change), signed=True)}"),
        (abs(float(capital_flow.northbound_net_flow)), f"北向强度 {num_text(float(capital_flow.northbound_net_flow), 2, signed=True)}"),
    ]
    ordered = [text for _, text in sorted(drivers, key=lambda item: item[0], reverse=True)[:4]]
    stage = sentiment_stage(score)
    summary = f"{stage}阶段，下一交易日情绪分 {score:.0f}/100。"
    return MarketSentimentState(score=score, stage=stage, drivers=ordered, summary=summary)


def status_score_penalty(status: str) -> float:
    status = str(status)
    if status == "halted":
        return 1.0
    if status == "delisted":
        return 1.5
    if status == "data_insufficient":
        return 0.08
    return 0.0


def alpha_score_components(
    stock: StockForecastState,
    *,
    deps: ForecastSupportDependencies,
) -> dict[str, float]:
    base_alpha_score = float(getattr(stock, "alpha_score", 0.0))
    up_1d = float(getattr(stock, "up_1d_prob", 0.5))
    up_5d = float(getattr(stock, "up_5d_prob", 0.5))
    up_20d = float(getattr(stock, "up_20d_prob", 0.5))
    up_2d = float(getattr(stock, "up_2d_prob", 0.65 * up_1d + 0.35 * up_5d))
    up_3d = float(getattr(stock, "up_3d_prob", 0.35 * up_1d + 0.65 * up_5d))
    excess_vs_sector = float(getattr(stock, "excess_vs_sector_prob", 0.5))
    event_impact = float(getattr(stock, "event_impact_score", 0.5))
    tradeability_score = float(getattr(stock, "tradeability_score", 0.5))
    if abs(base_alpha_score) <= 1e-12:
        base_components = {
            "short": 0.08 * up_1d,
            "two": 0.10 * up_2d,
            "three": 0.13 * up_3d,
            "five": 0.20 * up_5d,
            "mid": 0.20 * up_20d,
            "excess": 0.18 * excess_vs_sector,
            "tradeability": 0.07 * tradeability_score,
            "event": 0.04 * event_impact,
        }
        base_alpha_score = float(sum(base_components.values()))
    else:
        base_components = {}

    horizon_dispersion = float(
        deps.clip(
            0.20 * abs(up_1d - up_2d)
            + 0.25 * abs(up_2d - up_3d)
            + 0.25 * abs(up_3d - up_5d)
            + 0.30 * abs(up_5d - up_20d),
            0.0,
            1.0,
        )
    )
    execution_risk = float(deps.clip(1.0 - tradeability_score, 0.0, 1.0))
    event_risk = float(deps.clip((0.55 - event_impact) / 0.55, 0.0, 1.0))
    medium_edge = float(deps.clip(0.58 * (up_20d - 0.50) + 0.42 * (up_5d - 0.50), 0.0, 0.35))
    sector_edge = float(deps.clip(excess_vs_sector - 0.50, 0.0, 0.30))
    swing_edge = float(
        deps.clip(
            0.28 * max(0.0, up_2d - 0.50)
            + 0.40 * max(0.0, up_3d - 0.50)
            + 0.70 * max(0.0, up_5d - 0.50)
            + 0.32 * max(0.0, excess_vs_sector - 0.50),
            0.0,
            0.42,
        )
    )
    trend_alignment = float(
        deps.clip(
            0.55 * max(0.0, up_3d - up_1d)
            + 0.75 * max(0.0, up_5d - up_2d)
            + 0.90 * max(0.0, up_20d - up_5d),
            0.0,
            1.0,
        )
    )
    continuation_bonus = float(
        deps.clip(
            0.75 * max(0.0, up_3d - up_1d)
            + 1.05 * max(0.0, up_5d - up_3d)
            + 0.45 * max(0.0, excess_vs_sector - 0.52),
            0.0,
            0.38,
        )
    )
    stability_bonus = float(deps.clip((0.16 - horizon_dispersion) / 0.16, 0.0, 1.0))
    quality_bonus = float(deps.clip(0.65 * tradeability_score + 0.35 * event_impact - 0.55, 0.0, 0.35))
    reversal_penalty = float(deps.clip(up_1d - max(up_5d, up_20d), 0.0, 0.35))
    swing_fade_penalty = float(
        deps.clip(
            0.75 * max(0.0, up_1d - up_3d)
            + 0.95 * max(0.0, up_3d - up_5d)
            + 0.55 * max(0.0, 0.53 - excess_vs_sector),
            0.0,
            0.32,
        )
    )
    weak_mid_penalty = float(deps.clip(0.52 - up_20d, 0.0, 0.20))
    risk_penalty = float(
        0.16 * horizon_dispersion
        + 0.12 * execution_risk
        + 0.08 * event_risk
        + 0.10 * reversal_penalty
        + 0.12 * swing_fade_penalty
        + 0.08 * weak_mid_penalty
    )
    selection_bonus = float(
        0.14 * medium_edge
        + 0.16 * sector_edge
        + 0.18 * swing_edge
        + 0.12 * trend_alignment
        + 0.12 * continuation_bonus
        + 0.07 * stability_bonus
        + 0.06 * quality_bonus
    )
    status_penalty = float(status_score_penalty(getattr(stock, "tradability_status", "normal")))
    raw = dict(base_components)
    raw["base_alpha_score"] = float(base_alpha_score)
    raw["medium_edge"] = medium_edge
    raw["sector_edge"] = sector_edge
    raw["swing_edge"] = swing_edge
    raw["trend_alignment"] = trend_alignment
    raw["continuation_bonus"] = continuation_bonus
    raw["stability_bonus"] = stability_bonus
    raw["quality_bonus"] = quality_bonus
    raw["selection_bonus"] = selection_bonus
    raw["horizon_dispersion"] = horizon_dispersion
    raw["execution_risk"] = execution_risk
    raw["event_risk"] = event_risk
    raw["reversal_penalty"] = reversal_penalty
    raw["swing_fade_penalty"] = swing_fade_penalty
    raw["weak_mid_penalty"] = weak_mid_penalty
    raw["risk_penalty"] = risk_penalty
    raw["status_penalty"] = status_penalty
    raw["alpha_score"] = float(base_alpha_score + selection_bonus - risk_penalty - status_penalty)
    return raw
