from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.domain.entities import TradeAction


@dataclass(frozen=True)
class HorizonForecast:
    horizon_days: int
    label: str
    up_prob: float = 0.5
    expected_return: float = 0.0
    q10: float = 0.0
    q50: float = 0.0
    q90: float = 0.0
    price_low: float = float("nan")
    price_mid: float = float("nan")
    price_high: float = float("nan")
    confidence: float = 0.5
    confidence_reason: str = ""


@dataclass(frozen=True)
class MarketFactsState:
    sample_coverage: int = 0
    advancers: int = 0
    decliners: int = 0
    flats: int = 0
    limit_up_count: int = 0
    limit_down_count: int = 0
    new_high_count: int = 0
    new_low_count: int = 0
    median_return: float = 0.0
    sample_amount: float = 0.0
    amount_z20: float = 0.0


@dataclass(frozen=True)
class MarketSentimentState:
    score: float = 50.0
    stage: str = "中性"
    drivers: List[str] = field(default_factory=list)
    summary: str = ""


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
    up_2d_prob: float = 0.5
    up_3d_prob: float = 0.5
    up_10d_prob: float = 0.5
    latest_close: float = float("nan")
    horizon_forecasts: Dict[str, HorizonForecast] = field(default_factory=dict)
    market_facts: MarketFactsState = field(default_factory=MarketFactsState)
    sentiment: MarketSentimentState = field(default_factory=MarketSentimentState)


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
    up_2d_prob: float = 0.5
    up_3d_prob: float = 0.5
    up_10d_prob: float = 0.5
    latest_close: float = float("nan")
    horizon_forecasts: Dict[str, HorizonForecast] = field(default_factory=dict)
    selection_reasons: List[str] = field(default_factory=list)
    ranking_reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    invalidation_rule: str = ""
    action_reason: str = ""
    weight_reason: str = ""
    blocked_reason: str = ""


@dataclass(frozen=True)
class InfoAggregateState:
    short_score: float = 0.0
    mid_score: float = 0.0
    item_count: int = 0
    announcement_count: int = 0
    research_count: int = 0
    negative_event_risk: float = 0.0
    coverage_ratio: float = 0.0
    info_prob_1d: float = 0.5
    info_prob_5d: float = 0.5
    info_prob_20d: float = 0.5
    shadow_prob_1d: float = 0.5
    shadow_prob_5d: float = 0.5
    shadow_prob_20d: float = 0.5
    event_risk_level: float = 0.0
    catalyst_strength: float = 0.0
    coverage_confidence: float = 0.0
    source_diversity: float = 0.0


@dataclass(frozen=True)
class CapitalFlowState:
    northbound_net_flow: float = 0.0
    margin_balance_change: float = 0.0
    turnover_heat: float = 0.5
    large_order_bias: float = 0.0
    flow_regime: str = "neutral"


@dataclass(frozen=True)
class MacroContextState:
    style_regime: str = "balanced"
    commodity_pressure: float = 0.0
    fx_pressure: float = 0.0
    index_breadth_proxy: float = 0.5
    macro_risk_level: str = "neutral"


@dataclass(frozen=True)
class CandidateSelectionState:
    shortlisted_symbols: List[str] = field(default_factory=list)
    shortlisted_sectors: List[str] = field(default_factory=list)
    sector_slots: Dict[str, int] = field(default_factory=dict)
    total_scored: int = 0
    shortlist_size: int = 0
    shortlist_ratio: float = 0.0
    selection_mode: str = "full_universe"
    selection_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MainlineState:
    name: str
    driver: str = "sector_leadership"
    conviction: float = 0.0
    breadth: float = 0.0
    leadership: float = 0.0
    catalyst_strength: float = 0.0
    event_risk_level: float = 0.0
    capital_support: float = 0.0
    macro_alignment: float = 0.0
    sectors: List[str] = field(default_factory=list)
    representative_symbols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InfoItem:
    date: str
    target_type: str
    target: str
    horizon: str
    direction: str
    info_type: str
    title: str
    source_subset: str = "market_news"
    source_url: str = ""
    strength: float = 3.0
    confidence: float = 0.7
    source_weight: float = 0.0
    publisher: str = ""
    event_tag: str = ""
    event_id: str = ""


@dataclass(frozen=True)
class InfoSignalRecord:
    target: str
    target_name: str
    title: str
    info_type: str
    direction: str
    horizon: str
    event_tag: str = ""
    score: float = 0.0
    negative_event_risk: float = 0.0
    source_url: str = ""


@dataclass(frozen=True)
class InfoDivergenceRecord:
    symbol: str
    name: str
    quant_prob_20d: float
    info_prob_20d: float
    shadow_prob_20d: float
    gap: float


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
    candidate_selection: CandidateSelectionState = field(default_factory=CandidateSelectionState)
    mainlines: List[MainlineState] = field(default_factory=list)
    market_info_state: InfoAggregateState = field(default_factory=InfoAggregateState)
    sector_info_states: Dict[str, InfoAggregateState] = field(default_factory=dict)
    stock_info_states: Dict[str, InfoAggregateState] = field(default_factory=dict)
    capital_flow_state: CapitalFlowState = field(default_factory=CapitalFlowState)
    macro_context_state: MacroContextState = field(default_factory=MacroContextState)


@dataclass(frozen=True)
class PolicyInput:
    composite_state: CompositeState
    current_weights: Dict[str, float]
    current_cash: float
    total_equity: float
    current_holding_days: Dict[str, int] = field(default_factory=dict)


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
    event_risk_cutoff: float = 0.55
    catalyst_boost_cap: float = 0.12
    flow_exposure_cap: float = 0.08


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
    universe_size: int = 0
    universe_generation_rule: str = ""
    source_universe_manifest_path: str = ""
    info_manifest_path: str = ""
    info_hash: str = ""
    info_shadow_enabled: bool = False
    external_signal_manifest_path: str = ""
    external_signal_version: str = ""
    external_signal_enabled: bool = False
    capital_flow_snapshot: Dict[str, object] = field(default_factory=dict)
    macro_context_snapshot: Dict[str, object] = field(default_factory=dict)
    generator_manifest_path: str = ""
    generator_version: str = ""
    generator_hash: str = ""
    coarse_pool_size: int = 0
    refined_pool_size: int = 0
    selected_pool_size: int = 0
    theme_allocations: List[Dict[str, object]] = field(default_factory=list)
    run_id: str = ""
    data_window: str = ""
    model_hashes: Dict[str, str] = field(default_factory=dict)
    policy_hash: str = ""
    universe_hash: str = ""
    created_at: str = ""
    snapshot_hash: str = ""
    config_hash: str = ""
    manifest_path: str = ""
    use_us_index_context: bool = False
    us_index_source: str = ""


@dataclass(frozen=True)
class DailyRunResult:
    snapshot: StrategySnapshot
    composite_state: CompositeState
    policy_decision: PolicyDecision
    trade_actions: List[TradeAction]
    symbol_names: Dict[str, str] = field(default_factory=dict)
    info_hash: str = ""
    info_manifest_path: str = ""
    info_shadow_enabled: bool = False
    info_item_count: int = 0
    external_signal_manifest_path: str = ""
    external_signal_version: str = ""
    external_signal_enabled: bool = False
    capital_flow_snapshot: Dict[str, object] = field(default_factory=dict)
    macro_context_snapshot: Dict[str, object] = field(default_factory=dict)
    generator_manifest_path: str = ""
    generator_version: str = ""
    generator_hash: str = ""
    coarse_pool_size: int = 0
    refined_pool_size: int = 0
    selected_pool_size: int = 0
    theme_allocations: List[Dict[str, object]] = field(default_factory=list)
    top_negative_info_events: List[InfoSignalRecord] = field(default_factory=list)
    top_positive_info_signals: List[InfoSignalRecord] = field(default_factory=list)
    quant_info_divergence: List[InfoDivergenceRecord] = field(default_factory=list)
    run_id: str = ""
    snapshot_hash: str = ""
    config_hash: str = ""
    manifest_path: str = ""
    memory_path: str = ""
    memory_recall: "StrategyMemoryRecall" = field(default_factory=lambda: StrategyMemoryRecall())
    prediction_review: "PredictionReviewState" = field(default_factory=lambda: PredictionReviewState())


@dataclass(frozen=True)
class StrategyMemoryRecall:
    memory_path: str = ""
    updated_at: str = ""
    latest_research_run_id: str = ""
    latest_research_end_date: str = ""
    latest_research_release_gate_passed: bool = False
    latest_research_excess_annual_return: float = 0.0
    latest_research_information_ratio: float = 0.0
    recent_daily_run_count: int = 0
    average_target_exposure: float = 0.0
    exposure_trend: float = 0.0
    rebalance_ratio: float = 0.0
    recurring_symbols: List[str] = field(default_factory=list)
    recurring_sectors: List[str] = field(default_factory=list)
    recurring_risk_tags: List[str] = field(default_factory=list)
    recurring_positive_tags: List[str] = field(default_factory=list)
    recurring_event_risk_tags: List[str] = field(default_factory=list)
    recurring_catalyst_tags: List[str] = field(default_factory=list)
    recent_flow_regimes: List[str] = field(default_factory=list)
    recurring_macro_risk_levels: List[str] = field(default_factory=list)
    narrative: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionReviewWindow:
    window_days: int
    label: str
    hit_rate: float = 0.0
    avg_edge: float = 0.0
    realized_return: float = 0.0
    sample_size: int = 0
    note: str = ""


@dataclass(frozen=True)
class PredictionReviewState:
    windows: Dict[str, PredictionReviewWindow] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ThemeAllocationRecord:
    theme: str
    selected_count: int = 0
    refined_count: int = 0
    coarse_count: int = 0
    eligible_count: int = 0
    theme_strength: float = 0.0
    cap: int = 0
    floor: int = 0


@dataclass(frozen=True)
class UniverseGeneratorManifest:
    generator_version: str
    source_universe_path: str
    source_universe_size: int
    eligible_size: int
    coarse_pool_size: int
    refined_pool_size: int
    selected_pool_size: int
    generator_hash: str = ""
    manifest_path: str = ""
    theme_allocations: List[ThemeAllocationRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicUniverseResult:
    eligible_symbols: List[str] = field(default_factory=list)
    coarse_pool: List[Dict[str, object]] = field(default_factory=list)
    refined_pool: List[Dict[str, object]] = field(default_factory=list)
    selected_300: List[Dict[str, object]] = field(default_factory=list)
    theme_allocations: List[ThemeAllocationRecord] = field(default_factory=list)
    generator_manifest: UniverseGeneratorManifest = field(
        default_factory=lambda: UniverseGeneratorManifest(
            generator_version="",
            source_universe_path="",
            source_universe_size=0,
            eligible_size=0,
            coarse_pool_size=0,
            refined_pool_size=0,
            selected_pool_size=0,
        )
    )


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
    benchmark_total_return: float = 0.0
    benchmark_annual_return: float = 0.0
    excess_total_return: float = 0.0
    excess_annual_return: float = 0.0
    excess_max_drawdown: float = 0.0
    information_ratio: float = 0.0
    nav_curve: List[float] = field(default_factory=list)
    benchmark_nav_curve: List[float] = field(default_factory=list)
    excess_nav_curve: List[float] = field(default_factory=list)
    curve_dates: List[str] = field(default_factory=list)
    run_id: str = ""
    snapshot_hash: str = ""
    config_hash: str = ""


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
