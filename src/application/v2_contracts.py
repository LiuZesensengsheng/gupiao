from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.domain.entities import TradeAction


@dataclass(frozen=True)
class MarketForecastState:
    as_of_date: str
    up_1d_prob: float
    up_5d_prob: float
    up_20d_prob: float
    trend_state: str
    drawdown_risk: float
    volatility_regime: str
    liquidity_stress: float


@dataclass(frozen=True)
class SectorForecastState:
    sector: str
    up_5d_prob: float
    up_20d_prob: float
    relative_strength: float
    rotation_speed: float
    crowding_score: float


@dataclass(frozen=True)
class StockForecastState:
    symbol: str
    sector: str
    up_1d_prob: float
    up_5d_prob: float
    up_20d_prob: float
    excess_vs_sector_prob: float
    event_impact_score: float
    tradeability_score: float
    alpha_score: float = 0.0
    tradability_status: str = "normal"


@dataclass(frozen=True)
class CrossSectionForecastState:
    as_of_date: str
    large_vs_small_bias: float
    growth_vs_value_bias: float
    fund_flow_strength: float
    margin_risk_on_score: float
    breadth_strength: float
    leader_participation: float
    weak_stock_ratio: float


@dataclass(frozen=True)
class CompositeState:
    market: MarketForecastState
    cross_section: CrossSectionForecastState
    sectors: List[SectorForecastState]
    stocks: List[StockForecastState]
    strategy_mode: str
    risk_regime: str


@dataclass(frozen=True)
class PolicyInput:
    composite_state: CompositeState
    current_weights: Dict[str, float]
    current_cash: float
    total_equity: float


@dataclass(frozen=True)
class PolicySpec:
    risk_on_exposure: float = 0.85
    cautious_exposure: float = 0.60
    risk_off_exposure: float = 0.35
    risk_on_positions: int = 4
    cautious_positions: int = 3
    risk_off_positions: int = 2
    risk_on_turnover_cap: float = 0.40
    cautious_turnover_cap: float = 0.28
    risk_off_turnover_cap: float = 0.20


@dataclass(frozen=True)
class PolicyDecision:
    target_exposure: float
    target_position_count: int
    rebalance_now: bool
    rebalance_intensity: float
    intraday_t_allowed: bool
    turnover_cap: float
    sector_budgets: Dict[str, float] = field(default_factory=dict)
    desired_sector_budgets: Dict[str, float] = field(default_factory=dict)
    symbol_target_weights: Dict[str, float] = field(default_factory=dict)
    desired_symbol_target_weights: Dict[str, float] = field(default_factory=dict)
    execution_notes: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategySnapshot:
    strategy_id: str
    universe_id: str
    feature_set_version: str
    market_model_id: str
    sector_model_id: str
    stock_model_id: str
    cross_section_model_id: str
    policy_version: str
    execution_version: str


@dataclass(frozen=True)
class DailyRunResult:
    snapshot: StrategySnapshot
    composite_state: CompositeState
    policy_decision: PolicyDecision
    trade_actions: List[TradeAction]


@dataclass(frozen=True)
class V2BacktestSummary:
    start_date: str
    end_date: str
    n_days: int
    total_return: float
    annual_return: float
    max_drawdown: float
    avg_turnover: float
    total_cost: float
    gross_total_return: float = 0.0
    annual_vol: float = 0.0
    win_rate: float = 0.0
    trade_days: int = 0
    avg_fill_ratio: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_rank_ic: float = 0.0
    avg_top_decile_return: float = 0.0
    avg_top_bottom_spread: float = 0.0
    avg_top_k_hit_rate: float = 0.0
    horizon_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nav_curve: List[float] = field(default_factory=list)
    curve_dates: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class V2CalibrationResult:
    best_policy: PolicySpec
    best_score: float
    baseline: V2BacktestSummary
    calibrated: V2BacktestSummary
    trials: List[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class LearnedPolicyModel:
    feature_names: List[str]
    exposure_intercept: float
    exposure_coef: List[float]
    position_intercept: float
    position_coef: List[float]
    turnover_intercept: float
    turnover_coef: List[float]
    train_rows: int
    train_r2_exposure: float
    train_r2_positions: float
    train_r2_turnover: float


@dataclass(frozen=True)
class V2PolicyLearningResult:
    model: LearnedPolicyModel
    baseline: V2BacktestSummary
    learned: V2BacktestSummary
