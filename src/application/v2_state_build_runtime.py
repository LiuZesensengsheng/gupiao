from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    MarketForecastState,
    StockForecastState,
)
from src.infrastructure.modeling import (
    LogisticBinaryModel,
    MLPBinaryModel,
    MLPQuantileModel,
    QuantileLinearModel,
)


@dataclass(frozen=True)
class StateBuildRuntimeDependencies:
    predict_quantile_profiles: Callable[..., pd.DataFrame]
    distributional_score: Callable[..., float]
    status_tradeability_limit: Callable[[str], float]
    build_horizon_forecasts: Callable[..., dict[str, object]]
    return_quantile_profile_cls: type
    safe_float: Callable[[object, float], float]
    clip: Callable[[float, float, float], float]
    stock_policy_score: Callable[[StockForecastState], float]
    market_facts_from_row: Callable[[pd.Series], object]
    load_symbol_daily: Callable[..., pd.DataFrame]
    make_market_feature_frame: Callable[[pd.DataFrame], pd.DataFrame]
    build_market_context_features: Callable[..., object]
    decide_market_state: Callable[[float, float], object]
    forecast_cross_section_state: Callable[[pd.DataFrame], object]
    build_mainline_states: Callable[..., list[object]]
    build_candidate_selection_state: Callable[..., object]
    apply_leader_candidate_overlay: Callable[..., CompositeState]


def build_stock_states_from_panel_slice(
    *,
    panel_row: pd.DataFrame,
    feature_cols: list[str],
    short_model: LogisticBinaryModel | MLPBinaryModel,
    two_model: LogisticBinaryModel | MLPBinaryModel,
    three_model: LogisticBinaryModel | MLPBinaryModel,
    five_model: LogisticBinaryModel | MLPBinaryModel,
    mid_model: LogisticBinaryModel | MLPBinaryModel,
    short_q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
    mid_q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
    deps: StateBuildRuntimeDependencies,
) -> tuple[list[StockForecastState], pd.DataFrame]:
    if panel_row.empty:
        return [], pd.DataFrame()

    short_probs = short_model.predict_proba(panel_row, feature_cols)
    two_probs = two_model.predict_proba(panel_row, feature_cols)
    three_probs = three_model.predict_proba(panel_row, feature_cols)
    five_probs = five_model.predict_proba(panel_row, feature_cols)
    mid_probs = mid_model.predict_proba(panel_row, feature_cols)
    short_profiles = deps.predict_quantile_profiles(
        panel_row,
        feature_cols=feature_cols,
        q_models=short_q_models,
    )
    mid_profiles = deps.predict_quantile_profiles(
        panel_row,
        feature_cols=feature_cols,
        q_models=mid_q_models,
    )

    symbols = panel_row["symbol"].astype(str).to_numpy()
    sectors = panel_row.get("sector", pd.Series(["鍏朵粬"] * len(panel_row), index=panel_row.index)).fillna("鍏朵粬").astype(str).to_numpy()
    if "tradability_status" in panel_row.columns:
        statuses = panel_row["tradability_status"].fillna("normal").astype(str).to_numpy()
    else:
        statuses = np.full(len(panel_row), "normal", dtype=object)
    realized_1d_arr = panel_row.get("excess_ret_1_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_2d_arr = panel_row.get("excess_ret_2_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_3d_arr = panel_row.get("excess_ret_3_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_5d_arr = panel_row.get("excess_ret_5_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_20d_arr = panel_row.get("excess_ret_20_vs_sector", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    latest_close_arr = panel_row.get("close", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    short_expected_arr = short_profiles["expected_return"].to_numpy(dtype=float)
    mid_expected_arr = mid_profiles["expected_return"].to_numpy(dtype=float)

    interim_rows: list[dict[str, float | str]] = []
    sector_mid_total: dict[str, float] = {}
    sector_counts: dict[str, int] = {}

    for idx, symbol in enumerate(symbols):
        sector = sectors[idx]
        short_prob = float(short_probs[idx])
        two_prob = float(two_probs[idx])
        three_prob = float(three_probs[idx])
        five_prob = float(five_probs[idx])
        mid_prob = float(mid_probs[idx])
        short_expected_ret = float(short_expected_arr[idx])
        mid_expected_ret = float(mid_expected_arr[idx])
        interim_rows.append(
            {
                "idx": idx,
                "symbol": symbol,
                "sector": sector,
                "short_prob": short_prob,
                "two_prob": two_prob,
                "three_prob": three_prob,
                "five_prob": five_prob,
                "mid_prob": mid_prob,
                "short_expected_ret": short_expected_ret,
                "mid_expected_ret": mid_expected_ret,
                "score": deps.distributional_score(
                    short_prob=short_prob,
                    two_prob=two_prob,
                    three_prob=three_prob,
                    five_prob=five_prob,
                    mid_prob=mid_prob,
                    short_expected_ret=short_expected_ret,
                    mid_expected_ret=mid_expected_ret,
                ),
            }
        )
        sector_mid_total[sector] = sector_mid_total.get(sector, 0.0) + mid_prob
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    sector_avg_mid = {
        sector: sector_mid_total[sector] / max(1, sector_counts[sector])
        for sector in sector_mid_total
    }
    scored_rows: list[dict[str, float | str]] = []
    out: list[StockForecastState] = []
    for item in interim_rows:
        idx = int(item["idx"])
        short_prob = float(item["short_prob"])
        two_prob = float(item["two_prob"])
        three_prob = float(item["three_prob"])
        five_prob = float(item["five_prob"])
        mid_prob = float(item["mid_prob"])
        short_expected_ret = float(item["short_expected_ret"])
        mid_expected_ret = float(item["mid_expected_ret"])
        sector = str(item["sector"])
        status = str(statuses[idx] or "normal")
        tradeability = deps.clip(
            1.0
            - (
                0.18 * abs(short_prob - two_prob)
                + 0.22 * abs(two_prob - three_prob)
                + 0.25 * abs(three_prob - five_prob)
                + 0.35 * abs(five_prob - mid_prob)
            ),
            0.0,
            1.0,
        )
        tradeability = min(tradeability, deps.status_tradeability_limit(status))
        expected_anchor = float(np.clip(mid_expected_ret / 0.20, -0.5, 0.5))
        event_impact = float(deps.clip(0.5 + short_expected_ret / 0.06, 0.0, 1.0))
        if status in {"halted", "delisted"}:
            event_impact = 0.0
        out.append(
            StockForecastState(
                symbol=str(item["symbol"]),
                sector=sector,
                up_1d_prob=short_prob,
                up_2d_prob=two_prob,
                up_3d_prob=three_prob,
                up_5d_prob=five_prob,
                up_10d_prob=float(0.45 * five_prob + 0.55 * mid_prob),
                up_20d_prob=mid_prob,
                excess_vs_sector_prob=float(
                    deps.clip(
                        mid_prob
                        - sector_avg_mid.get(sector, mid_prob)
                        + 0.5
                        + 0.1 * expected_anchor,
                        0.0,
                        1.0,
                    )
                ),
                event_impact_score=event_impact,
                tradeability_score=float(tradeability),
                alpha_score=float(item["score"]),
                tradability_status=status,
                latest_close=float(latest_close_arr[idx]),
                horizon_forecasts=deps.build_horizon_forecasts(
                    latest_close=float(latest_close_arr[idx]),
                    horizon_probs={
                        "1d": short_prob,
                        "2d": two_prob,
                        "3d": three_prob,
                        "5d": five_prob,
                        "10d": float(0.45 * five_prob + 0.55 * mid_prob),
                        "20d": mid_prob,
                    },
                    short_profile=deps.return_quantile_profile_cls(
                        expected_return=float(short_profiles["expected_return"].iloc[idx]),
                        q10=float(short_profiles["q10"].iloc[idx]),
                        q30=float(short_profiles["q30"].iloc[idx]),
                        q20=float(short_profiles["q20"].iloc[idx]),
                        q50=float(short_profiles["q50"].iloc[idx]),
                        q70=float(short_profiles["q70"].iloc[idx]),
                        q80=float(short_profiles["q80"].iloc[idx]),
                        q90=float(short_profiles["q90"].iloc[idx]),
                    ),
                    mid_profile=deps.return_quantile_profile_cls(
                        expected_return=float(mid_profiles["expected_return"].iloc[idx]),
                        q10=float(mid_profiles["q10"].iloc[idx]),
                        q30=float(mid_profiles["q30"].iloc[idx]),
                        q20=float(mid_profiles["q20"].iloc[idx]),
                        q50=float(mid_profiles["q50"].iloc[idx]),
                        q70=float(mid_profiles["q70"].iloc[idx]),
                        q80=float(mid_profiles["q80"].iloc[idx]),
                        q90=float(mid_profiles["q90"].iloc[idx]),
                    ),
                    tradability_status=status,
                ),
            )
        )
        scored_rows.append(
            {
                "symbol": str(item["symbol"]),
                "score": float(item["score"]),
                "realized_ret_1d": float(realized_1d_arr[idx]),
                "realized_ret_2d": float(realized_2d_arr[idx]),
                "realized_ret_3d": float(realized_3d_arr[idx]),
                "realized_ret_5d": float(realized_5d_arr[idx]),
                "realized_ret_20d": float(realized_20d_arr[idx]),
            }
        )
    out.sort(
        key=lambda stock: (
            deps.distributional_score(
                short_prob=stock.up_1d_prob,
                two_prob=getattr(stock, "up_2d_prob", 0.5),
                three_prob=getattr(stock, "up_3d_prob", 0.5),
                five_prob=stock.up_5d_prob,
                mid_prob=stock.up_20d_prob,
                short_expected_ret=float(stock.event_impact_score - 0.5) * 0.06,
                mid_expected_ret=float(stock.excess_vs_sector_prob - 0.5) * 0.20,
            ),
            stock.up_20d_prob,
            stock.tradeability_score,
        ),
        reverse=True,
    )
    return out, pd.DataFrame(scored_rows)


def build_market_and_cross_section_states(
    *,
    market_symbol: str,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    use_margin_features: bool,
    margin_market_file: str,
    use_us_index_context: bool,
    us_index_source: str,
    use_us_sector_etf_context: bool = False,
    use_cn_etf_context: bool = False,
    cn_etf_source: str = "akshare",
    market_short_prob: float,
    market_two_prob: float | None,
    market_three_prob: float | None,
    market_five_prob: float | None,
    market_mid_prob: float,
    market_short_profile: object | None = None,
    market_mid_profile: object | None = None,
    deps: StateBuildRuntimeDependencies,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    market_raw = deps.load_symbol_daily(
        symbol=market_symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_feat_base = deps.make_market_feature_frame(market_raw)
    market_context = deps.build_market_context_features(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_dates=market_feat_base["date"],
        use_margin_features=use_margin_features,
        margin_market_file=margin_market_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        use_us_sector_etf_context=use_us_sector_etf_context,
        use_cn_etf_context=use_cn_etf_context,
        cn_etf_source=cn_etf_source,
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left")
    return build_market_and_cross_section_from_prebuilt_frame(
        market_frame=market_frame,
        market_short_prob=market_short_prob,
        market_two_prob=market_two_prob,
        market_three_prob=market_three_prob,
        market_five_prob=market_five_prob,
        market_mid_prob=market_mid_prob,
        market_short_profile=market_short_profile,
        market_mid_profile=market_mid_profile,
        deps=deps,
    )


def build_stock_states_from_rows(
    rows: list[object],
    sector_map: dict[str, str],
    *,
    sector_strength_map: dict[str, float] | None = None,
    deps: StateBuildRuntimeDependencies,
) -> list[StockForecastState]:
    out: list[StockForecastState] = []
    sector_mid_map: dict[str, float] = {}
    sector_counts: dict[str, int] = {}
    for row in rows:
        sector = sector_map.get(row.symbol, "鍏朵粬")
        sector_mid_map[sector] = sector_mid_map.get(sector, 0.0) + float(row.mid_prob)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    sector_avg_mid = {
        sector: sector_mid_map[sector] / max(1, sector_counts[sector])
        for sector in sector_mid_map
    }

    for row in rows:
        sector = sector_map.get(row.symbol, "鍏朵粬")
        status = str(getattr(row, "tradability_status", "normal") or "normal")
        two_prob = deps.safe_float(
            getattr(row, "two_prob", 0.65 * float(row.short_prob) + 0.35 * float(getattr(row, "five_prob", row.mid_prob))),
            0.5,
        )
        three_prob = deps.safe_float(
            getattr(row, "three_prob", 0.35 * float(row.short_prob) + 0.65 * float(getattr(row, "five_prob", row.mid_prob))),
            0.5,
        )
        five_prob = deps.safe_float(getattr(row, "five_prob", 0.6 * float(row.short_prob) + 0.4 * float(row.mid_prob)), 0.5)
        tradeability = deps.clip(
            1.0
            - (
                0.18 * abs(float(row.short_prob) - two_prob)
                + 0.22 * abs(two_prob - three_prob)
                + 0.25 * abs(three_prob - five_prob)
                + 0.35 * abs(five_prob - float(row.mid_prob))
            ),
            0.0,
            1.0,
        )
        tradeability = min(tradeability, deps.status_tradeability_limit(status))
        short_expected_ret = deps.safe_float(getattr(row, "short_expected_ret", 0.0), 0.0)
        mid_expected_ret = deps.safe_float(getattr(row, "mid_expected_ret", 0.0), 0.0)
        sector_excess_anchor = 0.0 if sector_strength_map is None else float(sector_strength_map.get(sector, 0.0))
        expected_anchor = float(np.clip(mid_expected_ret / 0.20, -0.5, 0.5))
        event_impact = float(deps.clip(0.5 + short_expected_ret / 0.06, 0.0, 1.0))
        if status in {"halted", "delisted"}:
            event_impact = 0.0
        latest_close = float(deps.safe_float(getattr(row, "latest_close", getattr(row, "close", np.nan)), np.nan))
        short_profile = deps.return_quantile_profile_cls(
            expected_return=float(short_expected_ret),
            q10=float(deps.safe_float(getattr(row, "short_q10", np.nan), np.nan)),
            q30=float(deps.safe_float(getattr(row, "short_q30", np.nan), np.nan)),
            q20=float(deps.safe_float(getattr(row, "short_q20", np.nan), np.nan)),
            q50=float(deps.safe_float(getattr(row, "short_q50", np.nan), np.nan)),
            q70=float(deps.safe_float(getattr(row, "short_q70", np.nan), np.nan)),
            q80=float(deps.safe_float(getattr(row, "short_q80", np.nan), np.nan)),
            q90=float(deps.safe_float(getattr(row, "short_q90", np.nan), np.nan)),
        )
        mid_profile = deps.return_quantile_profile_cls(
            expected_return=float(mid_expected_ret),
            q10=float(deps.safe_float(getattr(row, "mid_q10", np.nan), np.nan)),
            q30=float(deps.safe_float(getattr(row, "mid_q30", np.nan), np.nan)),
            q20=float(deps.safe_float(getattr(row, "mid_q20", np.nan), np.nan)),
            q50=float(deps.safe_float(getattr(row, "mid_q50", np.nan), np.nan)),
            q70=float(deps.safe_float(getattr(row, "mid_q70", np.nan), np.nan)),
            q80=float(deps.safe_float(getattr(row, "mid_q80", np.nan), np.nan)),
            q90=float(deps.safe_float(getattr(row, "mid_q90", np.nan), np.nan)),
        )
        horizon_probs = {
            "1d": float(row.short_prob),
            "2d": float(two_prob),
            "3d": float(three_prob),
            "5d": float(five_prob),
            "10d": float(0.45 * five_prob + 0.55 * float(row.mid_prob)),
            "20d": float(row.mid_prob),
        }
        out.append(
            StockForecastState(
                symbol=row.symbol,
                sector=sector,
                up_1d_prob=float(row.short_prob),
                up_2d_prob=float(two_prob),
                up_3d_prob=float(three_prob),
                up_5d_prob=float(five_prob),
                up_10d_prob=float(horizon_probs["10d"]),
                up_20d_prob=float(row.mid_prob),
                excess_vs_sector_prob=float(
                    deps.clip(
                        float(row.mid_prob)
                        - sector_avg_mid.get(sector, float(row.mid_prob))
                        + 0.5
                        + 0.2 * sector_excess_anchor
                        + 0.1 * expected_anchor,
                        0.0,
                        1.0,
                    )
                ),
                event_impact_score=event_impact,
                tradeability_score=float(tradeability),
                alpha_score=float(
                    deps.distributional_score(
                        short_prob=float(row.short_prob),
                        two_prob=float(two_prob),
                        three_prob=float(three_prob),
                        five_prob=float(five_prob),
                        mid_prob=float(row.mid_prob),
                        short_expected_ret=float(short_expected_ret),
                        mid_expected_ret=float(mid_expected_ret),
                    )
                ),
                tradability_status=status,
                latest_close=latest_close,
                horizon_forecasts=deps.build_horizon_forecasts(
                    latest_close=latest_close,
                    horizon_probs=horizon_probs,
                    short_profile=short_profile,
                    mid_profile=mid_profile,
                    tradability_status=status,
                ),
            )
        )
    out.sort(
        key=lambda item: (deps.stock_policy_score(item), item.up_20d_prob, item.excess_vs_sector_prob),
        reverse=True,
    )
    return out


def compose_state(
    *,
    market: MarketForecastState,
    sectors: list[object],
    stocks: list[StockForecastState],
    cross_section: CrossSectionForecastState,
    deps: StateBuildRuntimeDependencies,
) -> CompositeState:
    risk_score = max(
        float(market.drawdown_risk),
        float(market.liquidity_stress),
        float(cross_section.weak_stock_ratio),
    )
    if risk_score >= 0.60:
        risk_regime = "risk_off"
    elif risk_score >= 0.40:
        risk_regime = "cautious"
    else:
        risk_regime = "risk_on"

    if market.trend_state == "trend" and cross_section.breadth_strength >= 0.10:
        strategy_mode = "trend_follow"
    elif market.trend_state == "range":
        strategy_mode = "range_rotation"
    else:
        strategy_mode = "defensive"

    ordered_sectors = sorted(sectors, key=lambda item: (item.up_20d_prob, item.relative_strength), reverse=True)
    ordered_stocks = sorted(
        stocks,
        key=lambda item: (deps.stock_policy_score(item), item.up_20d_prob, item.excess_vs_sector_prob),
        reverse=True,
    )
    mainlines = deps.build_mainline_states(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        stock_score_fn=deps.stock_policy_score,
    )
    candidate_selection = deps.build_candidate_selection_state(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        mainlines=mainlines,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
        stock_score_fn=deps.stock_policy_score,
    )
    composite_state = CompositeState(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
        candidate_selection=candidate_selection,
        mainlines=mainlines,
    )
    return deps.apply_leader_candidate_overlay(
        state=composite_state,
    )


def build_market_and_cross_section_from_prebuilt_frame(
    *,
    market_frame: pd.DataFrame,
    market_short_prob: float,
    market_two_prob: float | None,
    market_three_prob: float | None,
    market_five_prob: float | None,
    market_mid_prob: float,
    market_short_profile: object | None = None,
    market_mid_profile: object | None = None,
    deps: StateBuildRuntimeDependencies,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    if market_frame["date"].is_monotonic_increasing:
        latest = market_frame.iloc[-1]
    else:
        latest = market_frame.sort_values("date").iloc[-1]
    state = deps.decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = deps.forecast_cross_section_state(market_frame)

    mkt_vol_20 = float(latest.get("mkt_volatility_20", 0.0))
    if mkt_vol_20 != mkt_vol_20:
        mkt_vol_20 = 0.0
    mkt_vol_60 = float(latest.get("mkt_volatility_60", mkt_vol_20))
    if mkt_vol_60 != mkt_vol_60:
        mkt_vol_60 = mkt_vol_20
    mkt_vol_60 = max(1e-6, mkt_vol_60)
    if mkt_vol_20 >= mkt_vol_60 * 1.15:
        volatility_regime = "high"
    elif mkt_vol_20 <= mkt_vol_60 * 0.85:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    drawdown_raw = float(latest.get("mkt_drawdown_20", 0.0))
    if drawdown_raw != drawdown_raw:
        drawdown_raw = 0.0
    latest_close = float(deps.safe_float(latest.get("close", np.nan), np.nan))
    default_five_prob = market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob)
    horizon_probs = {
        "1d": float(market_short_prob),
        "2d": float(
            market_two_prob
            if market_two_prob is not None
            else (0.65 * market_short_prob + 0.35 * (market_five_prob if market_five_prob is not None else market_mid_prob))
        ),
        "3d": float(
            market_three_prob
            if market_three_prob is not None
            else (0.35 * market_short_prob + 0.65 * (market_five_prob if market_five_prob is not None else market_mid_prob))
        ),
        "5d": float(default_five_prob),
        "10d": float(0.45 * default_five_prob + 0.55 * market_mid_prob),
        "20d": float(market_mid_prob),
    }
    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=horizon_probs["1d"],
        up_2d_prob=horizon_probs["2d"],
        up_3d_prob=horizon_probs["3d"],
        up_5d_prob=horizon_probs["5d"],
        up_10d_prob=horizon_probs["10d"],
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=deps.clip(abs(drawdown_raw), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=deps.clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
        latest_close=latest_close,
        horizon_forecasts=deps.build_horizon_forecasts(
            latest_close=latest_close,
            horizon_probs=horizon_probs,
            short_profile=market_short_profile,
            mid_profile=market_mid_profile,
        ),
        market_facts=deps.market_facts_from_row(latest),
    )
    cross_section = CrossSectionForecastState(
        as_of_date=str(cross_section_record.as_of_date.date()),
        large_vs_small_bias=float(cross_section_record.large_vs_small_bias),
        growth_vs_value_bias=float(cross_section_record.growth_vs_value_bias),
        fund_flow_strength=float(cross_section_record.fund_flow_strength),
        margin_risk_on_score=float(cross_section_record.margin_risk_on_score),
        breadth_strength=float(cross_section_record.breadth_strength),
        leader_participation=float(cross_section_record.leader_participation),
        weak_stock_ratio=float(cross_section_record.weak_stock_ratio),
    )
    return market, cross_section
