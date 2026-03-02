from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
    DailyRunResult,
    LearnedPolicyModel,
    MarketForecastState,
    PolicyDecision,
    PolicyInput,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    StrategySnapshot,
)
from src.application.watchlist import load_watchlist
from src.domain.entities import TradeAction
from src.domain.policies import blend_horizon_score, decide_market_state
from src.infrastructure.discovery import build_candidate_universe
from src.infrastructure.cross_section_forecast import forecast_cross_section_state
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
)
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import load_symbol_daily
from src.infrastructure.modeling import (
    LogisticBinaryModel,
    MLPBinaryModel,
    MLPQuantileModel,
    QuantileLinearModel,
)
from src.infrastructure.panel_dataset import build_stock_panel_dataset
from src.infrastructure.sector_data import build_sector_daily_frames
from src.infrastructure.sector_forecast import run_sector_forecast


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _coalesce(primary: object, secondary: object, default: object) -> object:
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def _policy_feature_names() -> list[str]:
    return [
        "mkt_up_1d",
        "mkt_up_20d",
        "mkt_drawdown_risk",
        "mkt_liquidity_stress",
        "cross_fund_flow",
        "cross_margin_risk_on",
        "cross_breadth",
        "cross_leader_participation",
        "cross_weak_ratio",
        "top_sector_up_20d",
        "top_sector_relative_strength",
        "top_stock_up_20d",
        "top_stock_tradeability",
        "top_stock_excess_vs_sector",
    ]


def _policy_feature_vector(state: CompositeState) -> np.ndarray:
    top_sector = state.sectors[0] if state.sectors else None
    top_stock = state.stocks[0] if state.stocks else None
    return np.asarray(
        [
            float(state.market.up_1d_prob),
            float(state.market.up_20d_prob),
            float(state.market.drawdown_risk),
            float(state.market.liquidity_stress),
            float(state.cross_section.fund_flow_strength),
            float(state.cross_section.margin_risk_on_score),
            float(state.cross_section.breadth_strength),
            float(state.cross_section.leader_participation),
            float(state.cross_section.weak_stock_ratio),
            0.0 if top_sector is None else float(top_sector.up_20d_prob),
            0.0 if top_sector is None else float(top_sector.relative_strength),
            0.0 if top_stock is None else float(top_stock.up_20d_prob),
            0.0 if top_stock is None else float(top_stock.tradeability_score),
            0.0 if top_stock is None else float(top_stock.excess_vs_sector_prob),
        ],
        dtype=float,
    )


def _fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    if X.size == 0 or y.size == 0:
        return 0.0, np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    X_aug = np.hstack([ones, X])
    reg = np.eye(X_aug.shape[1], dtype=float) * float(max(0.0, l2))
    reg[0, 0] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return float(coef[0]), np.asarray(coef[1:], dtype=float)


def _predict_ridge(features: np.ndarray, intercept: float, coef: np.ndarray) -> float:
    return float(intercept + np.dot(np.asarray(features, dtype=float), np.asarray(coef, dtype=float)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass(frozen=True)
class _ReturnQuantileProfile:
    expected_return: float
    q10: float
    q30: float
    q20: float
    q50: float
    q70: float
    q80: float
    q90: float


def _fit_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel]:
    return (
        QuantileLinearModel(quantile=0.10, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.30, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.70, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.90, l2=l2).fit(df, feature_cols, target_col),
    )


def _fit_mlp_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[MLPQuantileModel, MLPQuantileModel, MLPQuantileModel, MLPQuantileModel, MLPQuantileModel]:
    return (
        MLPQuantileModel(quantile=0.10, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.30, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.70, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.90, l2=l2).fit(df, feature_cols, target_col),
    )


def _predict_quantile_profile(
    row: pd.DataFrame,
    *,
    feature_cols: list[str],
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> _ReturnQuantileProfile:
    raw = [float(model.predict(row, feature_cols)[0]) for model in q_models]
    q10, q30, q50, q70, q90 = [float(x) for x in np.maximum.accumulate(np.asarray(raw, dtype=float))]
    q20 = float(0.5 * (q10 + q30))
    q80 = float(0.5 * (q70 + q90))
    return _ReturnQuantileProfile(
        expected_return=float(0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90),
        q10=q10,
        q30=q30,
        q20=q20,
        q50=q50,
        q70=q70,
        q80=q80,
        q90=q90,
    )


def _distributional_score(
    *,
    short_prob: float,
    five_prob: float,
    mid_prob: float,
    short_expected_ret: float,
    mid_expected_ret: float,
) -> float:
    base_score = float(
        0.25 * float(short_prob)
        + 0.35 * float(five_prob)
        + 0.40 * float(mid_prob)
    )
    short_ret_score = float(np.clip(0.5 + float(short_expected_ret) / 0.06, 0.0, 1.0))
    five_ret_score = float(np.clip(0.5 + (0.35 * float(short_expected_ret) + 0.65 * float(mid_expected_ret)) / 0.12, 0.0, 1.0))
    mid_ret_score = float(np.clip(0.5 + float(mid_expected_ret) / 0.20, 0.0, 1.0))
    dist_score = float(0.20 * short_ret_score + 0.35 * five_ret_score + 0.45 * mid_ret_score)
    return float(0.4 * base_score + 0.6 * dist_score)


def _is_actionable_status(status: str) -> bool:
    return str(status) not in {"halted", "delisted"}


def _status_tradeability_limit(status: str) -> float:
    status = str(status)
    if status in {"halted", "delisted"}:
        return 0.0
    if status == "data_insufficient":
        return 0.35
    return 1.0


def _status_score_penalty(status: str) -> float:
    status = str(status)
    if status == "halted":
        return 1.0
    if status == "delisted":
        return 1.5
    if status == "data_insufficient":
        return 0.08
    return 0.0


def _alpha_score_components(stock: StockForecastState) -> dict[str, float]:
    raw = {
        "short": 0.20 * float(stock.up_1d_prob),
        "five": 0.30 * float(stock.up_5d_prob),
        "mid": 0.25 * float(stock.up_20d_prob),
        "excess": 0.15 * float(stock.excess_vs_sector_prob),
        "tradeability": 0.10 * float(stock.tradeability_score),
        "event": 0.05 * float(stock.event_impact_score),
    }
    raw_score = float(sum(raw.values()))
    penalty = float(_status_score_penalty(getattr(stock, "tradability_status", "normal")))
    raw["status_penalty"] = penalty
    raw["alpha_score"] = float(raw_score - penalty)
    return raw


def _build_stock_states_from_panel_slice(
    *,
    panel_row: pd.DataFrame,
    feature_cols: list[str],
    short_model: LogisticBinaryModel | MLPBinaryModel,
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
) -> tuple[list[StockForecastState], pd.DataFrame]:
    if panel_row.empty:
        return [], pd.DataFrame()

    short_probs = short_model.predict_proba(panel_row, feature_cols)
    five_probs = five_model.predict_proba(panel_row, feature_cols)
    mid_probs = mid_model.predict_proba(panel_row, feature_cols)

    interim_rows: list[dict[str, float | str]] = []
    sector_mid_total: dict[str, float] = {}
    sector_counts: dict[str, int] = {}

    for idx, (_, latest_row) in enumerate(panel_row.iterrows()):
        latest_df = pd.DataFrame([latest_row])
        short_profile = _predict_quantile_profile(
            latest_df,
            feature_cols=feature_cols,
            q_models=short_q_models,
        )
        mid_profile = _predict_quantile_profile(
            latest_df,
            feature_cols=feature_cols,
            q_models=mid_q_models,
        )
        symbol = str(latest_row["symbol"])
        sector = str(latest_row.get("sector", "其他") or "其他")
        short_prob = float(short_probs[idx])
        five_prob = float(five_probs[idx])
        mid_prob = float(mid_probs[idx])
        interim_rows.append(
            {
                "symbol": symbol,
                "sector": sector,
                "short_prob": short_prob,
                "five_prob": five_prob,
                "mid_prob": mid_prob,
                "short_expected_ret": float(short_profile.expected_return),
                "mid_expected_ret": float(mid_profile.expected_return),
                "score": _distributional_score(
                    short_prob=short_prob,
                    five_prob=five_prob,
                    mid_prob=mid_prob,
                    short_expected_ret=float(short_profile.expected_return),
                    mid_expected_ret=float(mid_profile.expected_return),
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
        short_prob = float(item["short_prob"])
        five_prob = float(item["five_prob"])
        mid_prob = float(item["mid_prob"])
        short_expected_ret = float(item["short_expected_ret"])
        mid_expected_ret = float(item["mid_expected_ret"])
        sector = str(item["sector"])
        status = str(panel_row.loc[panel_row["symbol"] == str(item["symbol"]), "tradability_status"].iloc[0]) if "tradability_status" in panel_row.columns and not panel_row.loc[panel_row["symbol"] == str(item["symbol"]), "tradability_status"].empty else "normal"
        tradeability = _clip(1.0 - abs(short_prob - mid_prob), 0.0, 1.0)
        tradeability = min(tradeability, _status_tradeability_limit(status))
        expected_anchor = float(np.clip(mid_expected_ret / 0.20, -0.5, 0.5))
        event_impact = float(_clip(0.5 + short_expected_ret / 0.06, 0.0, 1.0))
        if status in {"halted", "delisted"}:
            event_impact = 0.0
        out.append(
            StockForecastState(
                symbol=str(item["symbol"]),
                sector=sector,
                up_1d_prob=short_prob,
                up_5d_prob=five_prob,
                up_20d_prob=mid_prob,
                excess_vs_sector_prob=float(
                    _clip(
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
            )
        )
        realized_1d = np.nan
        realized_5d = np.nan
        realized_20d = np.nan
        matched = panel_row[panel_row["symbol"] == str(item["symbol"])]
        if not matched.empty:
            realized_1d = float(matched.iloc[0].get("excess_ret_1_vs_mkt", np.nan))
            realized_5d = float(matched.iloc[0].get("excess_ret_5_vs_mkt", np.nan))
            realized_20d = float(matched.iloc[0].get("excess_ret_20_vs_sector", np.nan))
        scored_rows.append(
            {
                "symbol": str(item["symbol"]),
                "score": float(item["score"]),
                "realized_ret_1d": float(realized_1d),
                "realized_ret_5d": float(realized_5d),
                "realized_ret_20d": float(realized_20d),
            }
        )
    out.sort(
        key=lambda stock: (
            _distributional_score(
                short_prob=stock.up_1d_prob,
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


def _panel_slice_metrics(
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


def _panel_horizon_metrics(scored_rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping = {
        "1d": "realized_ret_1d",
        "5d": "realized_ret_5d",
        "20d": "realized_ret_20d",
    }
    out: dict[str, dict[str, float]] = {}
    for horizon, realized_col in mapping.items():
        rank_ic, top_decile_ret, top_bottom_spread, top_k_hit_rate = _panel_slice_metrics(
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


def build_strategy_snapshot(
    *,
    strategy_id: str,
    universe_id: str = "top_liquid_200",
) -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id=str(strategy_id).strip() or "swing_v2",
        universe_id=str(universe_id).strip() or "top_liquid_200",
        feature_set_version="fset_v2_core",
        market_model_id="mkt_lr_v2",
        sector_model_id="sector_lr_v2",
        stock_model_id="stock_lr_v2",
        cross_section_model_id="cross_section_v2",
        policy_version="policy_v2_rules",
        execution_version="exec_t1_v2",
    )


def _load_v2_runtime_settings(
    *,
    config_path: str,
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            payload = raw

    common = payload.get("common", {}) if isinstance(payload.get("common"), dict) else {}
    daily = payload.get("daily", {}) if isinstance(payload.get("daily"), dict) else {}

    def pick(key: str, default: object) -> object:
        return _coalesce(daily.get(key), common.get(key), default)

    return {
        "config_path": str(config_path),
        "watchlist": str(pick("watchlist", "config/watchlist.json")),
        "source": str(source).strip() if source is not None and str(source).strip() else str(pick("source", "auto")),
        "data_dir": str(pick("data_dir", "data")),
        "start": str(pick("start", "2018-01-01")),
        "end": str(pick("end", "2099-12-31")),
        "min_train_days": int(pick("min_train_days", 240)),
        "step_days": int(pick("step_days", 20)),
        "l2": float(pick("l2", 0.8)),
        "max_positions": int(pick("max_positions", 5)),
        "use_margin_features": bool(pick("use_margin_features", True)),
        "margin_market_file": str(pick("margin_market_file", "input/margin_market.csv")),
        "margin_stock_file": str(pick("margin_stock_file", "input/margin_stock.csv")),
        "universe_file": (
            str(universe_file).strip()
            if universe_file is not None and str(universe_file).strip()
            else str(pick("universe_file", "config/universe_smoke_5.json"))
        ),
        "universe_limit": int(
            universe_limit
            if universe_limit is not None
            else int(pick("universe_limit", 5))
        ),
    }


def _build_market_and_cross_section_states(
    *,
    market_symbol: str,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    use_margin_features: bool,
    margin_market_file: str,
    market_short_prob: float,
    market_five_prob: float | None,
    market_mid_prob: float,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    market_raw = load_symbol_daily(
        symbol=market_symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_dates=market_feat_base["date"],
        use_margin_features=use_margin_features,
        margin_market_file=margin_market_file,
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left")
    latest = market_frame.sort_values("date").iloc[-1]
    state = decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = forecast_cross_section_state(market_frame)

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

    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=float(market_short_prob),
        up_5d_prob=float(
            market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob)
        ),
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=_clip(abs(float(latest.get("mkt_drawdown_20", 0.0) or 0.0)), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=_clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
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


def _build_stock_states_from_rows(
    rows: list[object],
    sector_map: dict[str, str],
    *,
    sector_strength_map: dict[str, float] | None = None,
) -> list[StockForecastState]:
    out: list[StockForecastState] = []
    sector_mid_map: dict[str, float] = {}
    sector_counts: dict[str, int] = {}
    for row in rows:
        sector = sector_map.get(row.symbol, "其他")
        sector_mid_map[sector] = sector_mid_map.get(sector, 0.0) + float(row.mid_prob)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    sector_avg_mid = {
        sector: sector_mid_map[sector] / max(1, sector_counts[sector])
        for sector in sector_mid_map
    }

    for row in rows:
        sector = sector_map.get(row.symbol, "其他")
        status = str(getattr(row, "tradability_status", "normal") or "normal")
        five_prob = _safe_float(getattr(row, "five_prob", 0.6 * float(row.short_prob) + 0.4 * float(row.mid_prob)), 0.5)
        tradeability = _clip(
            1.0 - (0.55 * abs(float(row.short_prob) - five_prob) + 0.45 * abs(five_prob - float(row.mid_prob))),
            0.0,
            1.0,
        )
        tradeability = min(tradeability, _status_tradeability_limit(status))
        short_expected_ret = _safe_float(getattr(row, "short_expected_ret", 0.0), 0.0)
        mid_expected_ret = _safe_float(getattr(row, "mid_expected_ret", 0.0), 0.0)
        sector_excess_anchor = 0.0 if sector_strength_map is None else float(sector_strength_map.get(sector, 0.0))
        expected_anchor = float(np.clip(mid_expected_ret / 0.20, -0.5, 0.5))
        event_impact = float(_clip(0.5 + short_expected_ret / 0.06, 0.0, 1.0))
        if status in {"halted", "delisted"}:
            event_impact = 0.0
        out.append(
            StockForecastState(
                symbol=row.symbol,
                sector=sector,
                up_1d_prob=float(row.short_prob),
                up_5d_prob=float(five_prob),
                up_20d_prob=float(row.mid_prob),
                excess_vs_sector_prob=float(
                    _clip(
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
                    _distributional_score(
                        short_prob=float(row.short_prob),
                        five_prob=float(five_prob),
                        mid_prob=float(row.mid_prob),
                        short_expected_ret=float(short_expected_ret),
                        mid_expected_ret=float(mid_expected_ret),
                    )
                ),
                tradability_status=status,
            )
        )
    out.sort(
        key=lambda item: (_stock_policy_score(item), item.up_20d_prob, item.excess_vs_sector_prob),
        reverse=True,
    )
    return out


def compose_state(
    *,
    market: MarketForecastState,
    sectors: list[SectorForecastState],
    stocks: list[StockForecastState],
    cross_section: CrossSectionForecastState,
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
        key=lambda item: (_stock_policy_score(item), item.up_20d_prob, item.excess_vs_sector_prob),
        reverse=True,
    )
    return CompositeState(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
    )


def _ranked_sector_budgets(sectors: Iterable[SectorForecastState], *, target_exposure: float) -> dict[str, float]:
    rows = list(sectors)
    if not rows:
        return {}
    raw = [max(0.0, float(item.up_20d_prob) - 0.50) + max(0.0, float(item.relative_strength)) for item in rows]
    total = sum(raw)
    if total <= 1e-9:
        equal = float(target_exposure) / float(len(rows))
        return {item.sector: equal for item in rows}
    return {item.sector: float(target_exposure) * score / total for item, score in zip(rows, raw)}


def _stock_policy_score(stock: StockForecastState) -> float:
    alpha_score = float(getattr(stock, "alpha_score", 0.0))
    if abs(alpha_score) > 1e-12:
        penalty = float(_status_score_penalty(getattr(stock, "tradability_status", "normal")))
        return float(alpha_score - penalty)
    return float(_alpha_score_components(stock)["alpha_score"])


def _allocate_sector_slots(
    *,
    sector_budgets: dict[str, float],
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]],
    total_slots: int,
) -> dict[str, int]:
    active_sectors = [
        sector for sector, budget in sector_budgets.items()
        if float(budget) > 1e-9 and available_by_sector.get(sector)
    ]
    if not active_sectors or total_slots <= 0:
        return {}

    slots = {sector: 0 for sector in active_sectors}
    ordered = sorted(active_sectors, key=lambda sector: float(sector_budgets.get(sector, 0.0)), reverse=True)

    for sector in ordered:
        if total_slots <= 0:
            break
        slots[sector] = 1
        total_slots -= 1

    while total_slots > 0:
        expandable = [
            sector for sector in ordered
            if slots[sector] < len(available_by_sector.get(sector, []))
        ]
        if not expandable:
            break
        best = max(
            expandable,
            key=lambda sector: (
                float(sector_budgets.get(sector, 0.0)) / float(max(1, slots[sector] + 1)),
                len(available_by_sector.get(sector, [])),
            ),
        )
        slots[best] += 1
        total_slots -= 1
    return slots


def _allocate_with_sector_budgets(
    *,
    stocks: list[StockForecastState],
    sector_budgets: dict[str, float],
    target_position_count: int,
) -> dict[str, float]:
    max_single_position = 0.35
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]] = {}
    for stock in stocks:
        if not _is_actionable_status(getattr(stock, "tradability_status", "normal")):
            continue
        score = _stock_policy_score(stock)
        available_by_sector.setdefault(stock.sector, []).append((stock, score))
    for sector in list(available_by_sector):
        available_by_sector[sector].sort(key=lambda item: item[1], reverse=True)

    slots_by_sector = _allocate_sector_slots(
        sector_budgets=sector_budgets,
        available_by_sector=available_by_sector,
        total_slots=max(1, int(target_position_count)),
    )
    symbol_target_weights: dict[str, float] = {}
    for sector, slots in slots_by_sector.items():
        sector_budget = float(sector_budgets.get(sector, 0.0))
        picks = available_by_sector.get(sector, [])[: max(0, int(slots))]
        if not picks or sector_budget <= 1e-9:
            continue
        cap = min(float(max_single_position), float(sector_budget))
        remaining = float(sector_budget)
        uncapped = list(picks)
        while uncapped and remaining > 1e-9:
            sector_scores = [max(0.0, score - 0.50) for _, score in uncapped]
            sector_total = float(sum(sector_scores))
            if sector_total <= 1e-9:
                provisional = [remaining / float(len(uncapped))] * len(uncapped)
            else:
                provisional = [remaining * float(score) / sector_total for score in sector_scores]
            over_limit = [
                idx for idx, weight in enumerate(provisional)
                if float(weight) > cap + 1e-9
            ]
            if not over_limit:
                for (stock, _), weight in zip(uncapped, provisional):
                    symbol_target_weights[stock.symbol] = float(weight)
                remaining = 0.0
                break
            next_uncapped: list[tuple[StockForecastState, float]] = []
            for idx, pair in enumerate(uncapped):
                stock, score = pair
                if idx in over_limit:
                    symbol_target_weights[stock.symbol] = float(cap)
                    remaining -= float(cap)
                else:
                    next_uncapped.append((stock, score))
            uncapped = next_uncapped
    return symbol_target_weights


def _finalize_target_weights(
    *,
    desired_weights: dict[str, float],
    current_weights: dict[str, float],
    stocks: list[StockForecastState],
    target_exposure: float,
    min_trade_delta: float,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {symbol: max(0.0, float(weight)) for symbol, weight in desired_weights.items()}
    state_map = {item.symbol: item for item in stocks}
    notes: list[str] = []
    locked_symbols: set[str] = set()

    all_symbols = sorted(set(adjusted) | set(current_weights))
    for symbol in all_symbols:
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        state = state_map.get(symbol)
        status = "data_insufficient" if state is None else str(getattr(state, "tradability_status", "normal") or "normal")
        target = max(0.0, float(adjusted.get(symbol, 0.0)))

        if state is None and current > 1e-9:
            adjusted[symbol] = current
            locked_symbols.add(symbol)
            notes.append(f"{symbol}: missing state, holding frozen.")
            continue
        if not _is_actionable_status(status):
            if current > 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(f"{symbol}: {status}, holding frozen.")
            else:
                adjusted.pop(symbol, None)
                notes.append(f"{symbol}: {status}, new entry blocked.")
            continue
        if status == "data_insufficient":
            if current <= 1e-9 and target > 1e-9:
                adjusted.pop(symbol, None)
                notes.append(f"{symbol}: data insufficient, new entry blocked.")
                continue
            if target > current + 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(f"{symbol}: data insufficient, add-on blocked.")
                continue

    for symbol in sorted(set(adjusted) | set(current_weights)):
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        target = max(0.0, float(adjusted.get(symbol, 0.0)))
        if abs(target - current) < float(min_trade_delta):
            if abs(target - current) > 1e-9:
                notes.append(f"{symbol}: rebalance gap below threshold.")
            if current > 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
            else:
                adjusted.pop(symbol, None)

    locked_total = float(sum(max(0.0, float(adjusted.get(symbol, 0.0))) for symbol in locked_symbols))
    free_symbols = [
        symbol for symbol, weight in adjusted.items()
        if symbol not in locked_symbols and float(weight) > 1e-9
    ]
    free_total = float(sum(float(adjusted[symbol]) for symbol in free_symbols))
    free_budget = max(0.0, float(target_exposure) - locked_total)
    if free_total > free_budget + 1e-9 and free_total > 1e-9:
        scale = float(free_budget / free_total) if free_budget > 1e-9 else 0.0
        for symbol in free_symbols:
            adjusted[symbol] = float(adjusted[symbol]) * scale
        notes.append("Actionable targets scaled down to respect target exposure after frozen holdings.")

    adjusted = {
        symbol: float(weight)
        for symbol, weight in adjusted.items()
        if float(weight) > 1e-6
    }
    return adjusted, notes


def _sector_budgets_from_weights(
    *,
    symbol_weights: dict[str, float],
    stocks: list[StockForecastState],
) -> dict[str, float]:
    state_map = {item.symbol: item for item in stocks}
    out: dict[str, float] = {}
    for symbol, weight in symbol_weights.items():
        if float(weight) <= 1e-9:
            continue
        sector = state_map.get(symbol).sector if state_map.get(symbol) is not None else "其他"
        out[sector] = out.get(sector, 0.0) + float(weight)
    return out


def apply_policy(
    policy_input: PolicyInput,
    *,
    policy_spec: PolicySpec | None = None,
) -> PolicyDecision:
    policy_spec = policy_spec or PolicySpec()
    state = policy_input.composite_state
    market = state.market
    cross = state.cross_section

    target_exposure = float(policy_spec.risk_off_exposure)
    target_position_count = int(policy_spec.risk_off_positions)
    turnover_cap = float(policy_spec.risk_off_turnover_cap)
    intraday_t_allowed = False
    risk_notes: list[str] = []

    if state.risk_regime == "risk_on":
        target_exposure = float(policy_spec.risk_on_exposure)
        target_position_count = int(policy_spec.risk_on_positions)
        turnover_cap = float(policy_spec.risk_on_turnover_cap)
        intraday_t_allowed = state.strategy_mode == "range_rotation"
    elif state.risk_regime == "cautious":
        target_exposure = float(policy_spec.cautious_exposure)
        target_position_count = int(policy_spec.cautious_positions)
        turnover_cap = float(policy_spec.cautious_turnover_cap)
    else:
        risk_notes.append("Risk-off regime: exposure capped aggressively.")

    if market.up_20d_prob < 0.50:
        target_exposure = min(target_exposure, 0.45)
        risk_notes.append("Market mid-term probability below 0.50.")
    if market.drawdown_risk >= 0.50:
        target_exposure = min(target_exposure, 0.35)
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Drawdown risk elevated.")
    if cross.fund_flow_strength < 0.0:
        target_exposure *= 0.85
        risk_notes.append("Fund flow weak.")

    target_exposure = _clip(target_exposure, 0.0, 1.0)
    desired_sector_budgets = _ranked_sector_budgets(state.sectors[: max(1, target_position_count)], target_exposure=target_exposure)
    desired_symbol_target_weights = _allocate_with_sector_budgets(
        stocks=state.stocks,
        sector_budgets=desired_sector_budgets,
        target_position_count=int(target_position_count),
    )
    symbol_target_weights, execution_notes = _finalize_target_weights(
        desired_weights=desired_symbol_target_weights,
        current_weights=policy_input.current_weights,
        stocks=state.stocks,
        target_exposure=target_exposure,
        min_trade_delta=min(0.02, 0.25 * float(turnover_cap)),
    )
    risk_notes.extend(execution_notes)
    sector_budgets = _sector_budgets_from_weights(
        symbol_weights=symbol_target_weights,
        stocks=state.stocks,
    )

    current_total = sum(max(0.0, float(v)) for v in policy_input.current_weights.values())
    rebalance_gap = abs(float(target_exposure) - float(current_total))
    rebalance_now = rebalance_gap >= 0.05
    rebalance_intensity = _clip(rebalance_gap / max(0.05, turnover_cap), 0.0, 1.0)

    return PolicyDecision(
        target_exposure=float(target_exposure),
        target_position_count=int(target_position_count),
        rebalance_now=bool(rebalance_now),
        rebalance_intensity=float(rebalance_intensity),
        intraday_t_allowed=bool(intraday_t_allowed),
        turnover_cap=float(turnover_cap),
        sector_budgets=sector_budgets,
        desired_sector_budgets=desired_sector_budgets,
        symbol_target_weights=symbol_target_weights,
        desired_symbol_target_weights=desired_symbol_target_weights,
        execution_notes=execution_notes,
        risk_notes=risk_notes,
    )


def build_trade_actions(
    *,
    decision: PolicyDecision,
    current_weights: dict[str, float],
) -> list[TradeAction]:
    all_symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    actions: list[TradeAction] = []
    for symbol in all_symbols:
        current_weight = max(0.0, float(current_weights.get(symbol, 0.0)))
        target_weight = max(0.0, float(decision.symbol_target_weights.get(symbol, 0.0)))
        delta_weight = float(target_weight - current_weight)
        if delta_weight > 0.02:
            action = "BUY"
        elif delta_weight < -0.02:
            action = "SELL"
        else:
            action = "HOLD"

        note = ""
        if action == "HOLD" and abs(delta_weight) > 1e-9:
            note = "below_rebalance_threshold"

        actions.append(
            TradeAction(
                symbol=symbol,
                name=symbol,
                action=action,
                current_weight=float(current_weight),
                target_weight=float(target_weight),
                delta_weight=float(delta_weight),
                note=note,
            )
        )
    actions.sort(key=lambda item: (abs(float(item.delta_weight)), float(item.target_weight)), reverse=True)
    return actions


def _policy_spec_from_model(
    *,
    state: CompositeState,
    model: LearnedPolicyModel,
) -> PolicySpec:
    features = _policy_feature_vector(state)
    exposure = _clip(
        _predict_ridge(features, model.exposure_intercept, np.asarray(model.exposure_coef, dtype=float)),
        0.20,
        0.95,
    )
    positions = int(
        round(
            _clip(
                _predict_ridge(features, model.position_intercept, np.asarray(model.position_coef, dtype=float)),
                1.0,
                6.0,
            )
        )
    )
    turnover_cap = _clip(
        _predict_ridge(features, model.turnover_intercept, np.asarray(model.turnover_coef, dtype=float)),
        0.10,
        0.45,
    )
    cautious_exposure = _clip(0.5 * (exposure + 0.35), 0.30, exposure)
    risk_off_exposure = _clip(0.5 * cautious_exposure, 0.20, 0.40)
    cautious_positions = min(positions, max(1, positions - 1))
    risk_off_positions = max(1, positions - 2)
    cautious_turnover = _clip(min(turnover_cap, 0.85 * turnover_cap), 0.10, turnover_cap)
    risk_off_turnover = _clip(min(cautious_turnover, 0.70 * turnover_cap), 0.08, cautious_turnover)
    return PolicySpec(
        risk_on_exposure=float(exposure),
        cautious_exposure=float(cautious_exposure),
        risk_off_exposure=float(risk_off_exposure),
        risk_on_positions=int(positions),
        cautious_positions=int(cautious_positions),
        risk_off_positions=int(risk_off_positions),
        risk_on_turnover_cap=float(turnover_cap),
        cautious_turnover_cap=float(cautious_turnover),
        risk_off_turnover_cap=float(risk_off_turnover),
    )


def _simulate_execution_day(
    *,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    decision: PolicyDecision,
    current_weights: dict[str, float],
    current_cash: float,
    stock_states: list[StockForecastState],
    stock_frames: dict[str, pd.DataFrame],
    total_commission_rate: float,
    base_slippage_rate: float,
) -> tuple[float, float, float, float, float, dict[str, float], float]:
    state_map = {item.symbol: item for item in stock_states}
    symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    raw_deltas = {
        symbol: float(decision.symbol_target_weights.get(symbol, 0.0)) - float(current_weights.get(symbol, 0.0))
        for symbol in symbols
    }

    executed_deltas: dict[str, float] = {}
    fill_ratios: list[float] = []
    slippage_rates: list[float] = []
    slippage_amounts: list[float] = []
    total_turnover_budget = float(max(0.0, decision.turnover_cap))
    used_turnover = 0.0

    ordered_symbols = sorted(symbols, key=lambda sym: abs(raw_deltas.get(sym, 0.0)), reverse=True)
    for symbol in ordered_symbols:
        delta = float(raw_deltas.get(symbol, 0.0))
        if abs(delta) <= 1e-4:
            continue
        state = state_map.get(symbol)
        frame = stock_frames.get(symbol)
        day_row = None
        if frame is not None:
            day_row = frame[frame["date"] == date]
        status = "normal"
        if state is not None:
            status = str(getattr(state, "tradability_status", "normal") or "normal")
        elif frame is None or day_row is None or day_row.empty:
            status = "halted"
        if not _is_actionable_status(status):
            continue
        if day_row is not None and not day_row.empty:
            latest = day_row.iloc[0]
            close_px = _safe_float(latest.get("close"), np.nan)
            low_px = _safe_float(latest.get("low"), np.nan)
            high_px = _safe_float(latest.get("high"), np.nan)
            ret_1 = _safe_float(latest.get("ret_1"), np.nan)
            if close_px == close_px and ret_1 == ret_1:
                prev_close = close_px / max(1e-9, 1.0 + ret_1)
                limit_up_px = prev_close * 1.098
                limit_down_px = prev_close * 0.902
                # Single rebalance per trading day is already T+1 compatible; this only blocks orders
                # when the instrument appears pinned at the daily price limit for the full session.
                if delta > 0.0 and low_px == low_px and low_px >= limit_up_px:
                    continue
                if delta < 0.0 and high_px == high_px and high_px <= limit_down_px:
                    continue
        tradeability = 0.45 if state is None else _clip(float(state.tradeability_score), 0.10, 1.0)
        tradeability = min(tradeability, _status_tradeability_limit(status))
        liquidity_cap = 0.03 + 0.12 * tradeability
        remaining_turnover = max(0.0, total_turnover_budget - used_turnover)
        max_abs_trade = min(abs(delta), liquidity_cap, remaining_turnover)
        if max_abs_trade <= 1e-6:
            continue
        fill_ratio = max_abs_trade / max(abs(delta), 1e-9)
        executed = float(np.sign(delta) * max_abs_trade)
        executed_deltas[symbol] = executed
        fill_ratios.append(float(fill_ratio))
        impact = max_abs_trade / max(liquidity_cap, 1e-6)
        slippage_rate = float(base_slippage_rate * (0.65 + 0.7 * impact + 0.35 * (1.0 - tradeability)))
        slippage_rates.append(slippage_rate)
        slippage_amounts.append(float(abs(executed) * slippage_rate))
        used_turnover += abs(executed)

    executed_weights = {symbol: max(0.0, float(weight)) for symbol, weight in current_weights.items()}
    for symbol, delta in executed_deltas.items():
        executed_weights[symbol] = max(0.0, float(executed_weights.get(symbol, 0.0)) + float(delta))
    executed_weights = {
        symbol: float(weight)
        for symbol, weight in executed_weights.items()
        if float(weight) > 1e-6
    }

    invested_after_trade = float(sum(executed_weights.values()))
    if invested_after_trade > 1.0:
        scale = 1.0 / invested_after_trade
        executed_weights = {symbol: float(weight) * scale for symbol, weight in executed_weights.items()}
        invested_after_trade = 1.0
    cash_after_trade = max(0.0, 1.0 - invested_after_trade)

    gross_end_value = float(cash_after_trade)
    position_values: dict[str, float] = {}
    for symbol, weight in executed_weights.items():
        frame = stock_frames.get(symbol)
        if frame is None:
            realized_ret = 0.0
        else:
            row = frame[frame["date"] == date]
            realized_ret = 0.0 if row.empty else float(row.iloc[0]["fwd_ret_1"])
        value = float(weight) * (1.0 + realized_ret)
        position_values[symbol] = value
        gross_end_value += value

    commission_cost = float(used_turnover * total_commission_rate)
    slippage_cost = float(sum(slippage_amounts)) if slippage_amounts else 0.0
    total_cost = float(commission_cost + slippage_cost)
    net_end_value = max(1e-9, gross_end_value - total_cost)

    next_weights = {
        symbol: float(value) / net_end_value
        for symbol, value in position_values.items()
        if float(value) > 1e-9
    }
    next_cash = max(0.0, float(cash_after_trade - total_cost) / net_end_value)
    daily_return = float(net_end_value - 1.0)
    avg_fill_ratio = float(np.mean(fill_ratios)) if fill_ratios else 0.0
    avg_slippage_bps = float(np.mean(slippage_rates) * 10000.0) if slippage_rates else 0.0
    return (
        daily_return,
        float(used_turnover),
        float(total_cost),
        avg_fill_ratio,
        avg_slippage_bps,
        next_weights,
        next_cash,
    )


def _to_v2_backtest_summary(
    *,
    returns: list[float],
    benchmark_returns: list[float] | None = None,
    turnovers: list[float],
    costs: list[float],
    gross_returns: list[float],
    fill_ratios: list[float],
    slippage_bps: list[float],
    rank_ics: list[float] | None = None,
    top_decile_returns: list[float] | None = None,
    top_bottom_spreads: list[float] | None = None,
    top_k_hit_rates: list[float] | None = None,
    horizon_metrics: dict[str, dict[str, list[float]]] | None = None,
    dates: list[pd.Timestamp],
) -> V2BacktestSummary:
    if not returns or not dates:
        return V2BacktestSummary(
            start_date="",
            end_date="",
            n_days=0,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            avg_turnover=0.0,
            total_cost=0.0,
            avg_rank_ic=0.0,
            avg_top_decile_return=0.0,
            avg_top_bottom_spread=0.0,
            avg_top_k_hit_rate=0.0,
            horizon_metrics={},
        )
    ret_arr = np.asarray(returns, dtype=float)
    nav = np.cumprod(1.0 + ret_arr)
    bench_arr = np.asarray(benchmark_returns if benchmark_returns is not None else np.zeros_like(ret_arr), dtype=float)
    if bench_arr.shape != ret_arr.shape:
        bench_arr = np.resize(bench_arr, ret_arr.shape)
    benchmark_nav = np.cumprod(1.0 + bench_arr)
    excess_ret_arr = (1.0 + ret_arr) / np.maximum(1.0 + bench_arr, 1e-9) - 1.0
    excess_nav = np.cumprod(1.0 + excess_ret_arr)
    gross_nav = np.cumprod(1.0 + np.asarray(gross_returns, dtype=float)) if gross_returns else nav
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.maximum(peak, 1e-12) - 1.0
    excess_peak = np.maximum.accumulate(excess_nav)
    total_return = float(nav[-1] - 1.0)
    benchmark_total_return = float(benchmark_nav[-1] - 1.0)
    excess_total_return = float(excess_nav[-1] - 1.0)
    gross_total_return = float(gross_nav[-1] - 1.0)
    n_days = len(returns)
    annual_return = float((1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0)
    benchmark_annual_return = float((1.0 + benchmark_total_return) ** (252.0 / max(1, n_days)) - 1.0)
    excess_annual_return = float((1.0 + excess_total_return) ** (252.0 / max(1, n_days)) - 1.0)
    annual_vol = float(np.std(ret_arr, ddof=0) * np.sqrt(252.0))
    win_rate = float(np.mean(ret_arr > 0.0))
    excess_drawdown = excess_nav / np.maximum(excess_peak, 1e-12) - 1.0
    excess_vol = float(np.std(excess_ret_arr, ddof=0))
    information_ratio = 0.0 if excess_vol <= 1e-12 else float(np.mean(excess_ret_arr) / excess_vol * np.sqrt(252.0))
    horizon_summary: dict[str, dict[str, float]] = {}
    if horizon_metrics:
        for horizon, metric_map in horizon_metrics.items():
            horizon_summary[horizon] = {
                key: float(np.mean(values)) if values else 0.0
                for key, values in metric_map.items()
            }
    return V2BacktestSummary(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        n_days=int(n_days),
        total_return=float(total_return),
        annual_return=float(annual_return),
        max_drawdown=float(np.min(drawdown)),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        total_cost=float(np.sum(costs)) if costs else 0.0,
        gross_total_return=float(gross_total_return),
        annual_vol=float(annual_vol),
        win_rate=float(win_rate),
        trade_days=int(sum(1 for item in turnovers if float(item) > 1e-9)),
        avg_fill_ratio=float(np.mean(fill_ratios)) if fill_ratios else 0.0,
        avg_slippage_bps=float(np.mean(slippage_bps)) if slippage_bps else 0.0,
        avg_rank_ic=float(np.mean(rank_ics)) if rank_ics else 0.0,
        avg_top_decile_return=float(np.mean(top_decile_returns)) if top_decile_returns else 0.0,
        avg_top_bottom_spread=float(np.mean(top_bottom_spreads)) if top_bottom_spreads else 0.0,
        avg_top_k_hit_rate=float(np.mean(top_k_hit_rates)) if top_k_hit_rates else 0.0,
        horizon_metrics=horizon_summary,
        benchmark_total_return=float(benchmark_total_return),
        benchmark_annual_return=float(benchmark_annual_return),
        excess_total_return=float(excess_total_return),
        excess_annual_return=float(excess_annual_return),
        excess_max_drawdown=float(np.min(excess_drawdown)),
        information_ratio=float(information_ratio),
        nav_curve=[float(x) for x in nav.tolist()],
        benchmark_nav_curve=[float(x) for x in benchmark_nav.tolist()],
        excess_nav_curve=[float(x) for x in excess_nav.tolist()],
        curve_dates=[str(item.date()) for item in dates],
    )


def _build_market_and_cross_section_from_prebuilt_frame(
    *,
    market_frame: pd.DataFrame,
    market_short_prob: float,
    market_five_prob: float | None,
    market_mid_prob: float,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    latest = market_frame.sort_values("date").iloc[-1]
    state = decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = forecast_cross_section_state(market_frame)

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
    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=float(market_short_prob),
        up_5d_prob=float(
            market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob)
        ),
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=_clip(abs(drawdown_raw), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=_clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
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


def _derive_learning_targets(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> tuple[float, float, float]:
    ranked = sorted(state.stocks, key=_stock_policy_score, reverse=True)
    realized: list[float] = []
    for stock in ranked[:4]:
        frame = stock_frames.get(stock.symbol)
        if frame is None:
            continue
        row = frame[frame["date"] == date]
        if row.empty:
            continue
        realized.append(float(row.iloc[0]["fwd_ret_1"]))
    lead_ret = float(np.mean(realized)) if realized else 0.0

    if lead_ret >= 0.008:
        exposure = 0.85
    elif lead_ret >= 0.0:
        exposure = 0.60
    else:
        exposure = 0.35

    breadth_bonus = 1 if float(state.cross_section.breadth_strength) > 0.05 else 0
    weakness_penalty = 1 if float(state.cross_section.weak_stock_ratio) > 0.55 else 0
    positions = int(np.clip(3 + breadth_bonus - weakness_penalty + (1 if lead_ret >= 0.012 else 0), 1, 5))

    turnover = 0.18
    if abs(lead_ret) >= 0.01:
        turnover = 0.32
    elif abs(lead_ret) >= 0.004:
        turnover = 0.25
    if float(state.market.drawdown_risk) > 0.45:
        turnover = min(turnover, 0.18)

    return float(exposure), float(positions), float(turnover)


@dataclass(frozen=True)
class _PreparedV2BacktestData:
    settings: dict[str, object]
    market_valid: pd.DataFrame
    market_feature_cols: list[str]
    panel: pd.DataFrame
    feature_cols: list[str]
    stock_frames: dict[str, pd.DataFrame]
    dates: list[pd.Timestamp]


@dataclass(frozen=True)
class _TrajectoryStep:
    date: pd.Timestamp
    next_date: pd.Timestamp
    composite_state: CompositeState
    stock_states: list[StockForecastState]
    horizon_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class _BacktestTrajectory:
    prepared: _PreparedV2BacktestData
    steps: list[_TrajectoryStep]


class ForecastBackend(Protocol):
    name: str

    def build_trajectory(
        self,
        prepared: _PreparedV2BacktestData,
        *,
        retrain_days: int = 20,
    ) -> _BacktestTrajectory:
        ...


def _empty_v2_backtest_result() -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    return (
        _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ),
        [],
    )


def _prepare_v2_backtest_data(
    *,
    config_path: str,
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
) -> _PreparedV2BacktestData | None:
    settings = _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    market_security, _, _ = load_watchlist(str(settings["watchlist"]))
    universe = build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows
    if not stocks:
        return None

    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_dates=market_feat_base["date"],
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = list(MARKET_FEATURE_COLUMNS) + list(market_context.feature_columns)
    market_valid = market_frame.dropna(
        subset=market_feature_cols + ["mkt_target_1d_up", "mkt_target_5d_up", "mkt_target_20d_up"]
    ).sort_values("date").copy()
    if market_valid.empty:
        return None

    panel_bundle = build_stock_panel_dataset(
        stock_securities=stocks,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_frame=market_frame,
        extra_market_cols=list(market_context.feature_columns),
        use_margin_features=bool(settings["use_margin_features"]),
        margin_stock_file=str(settings["margin_stock_file"]),
    )
    panel = panel_bundle.frame
    feature_cols = list(panel_bundle.feature_columns)
    if panel.empty or not feature_cols:
        return None

    stock_frames = {
        str(symbol): frame.sort_values("date").copy()
        for symbol, frame in panel.groupby("symbol", observed=True)
    }
    common_dates = set(pd.to_datetime(market_valid["date"])) & set(pd.to_datetime(panel["date"]))
    dates = sorted(pd.Timestamp(d) for d in common_dates)
    min_train_days = int(settings["min_train_days"])
    if len(dates) <= min_train_days + 1:
        return None

    return _PreparedV2BacktestData(
        settings=settings,
        market_valid=market_valid,
        market_feature_cols=market_feature_cols,
        panel=panel,
        feature_cols=feature_cols,
        stock_frames=stock_frames,
        dates=dates,
    )


def _tensorize_temporal_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    group_col: str | None,
    lag_depth: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty or not feature_cols:
        return frame.copy(), []
    lag_depth = max(1, int(lag_depth))
    if group_col is None:
        work = frame.sort_values("date").copy()
        grouped = None
    else:
        work = frame.sort_values([group_col, "date"]).copy()
        grouped = work.groupby(group_col, observed=True, sort=False)

    out_cols: list[str] = []
    lag_frames: list[pd.DataFrame] = []
    for lag in range(lag_depth):
        if grouped is None:
            shifted = work[feature_cols].shift(lag)
        else:
            shifted = grouped[feature_cols].shift(lag)
        new_cols = [f"{col}__lag{lag}" for col in feature_cols]
        shifted = shifted.copy()
        shifted.columns = new_cols
        lag_frames.append(shifted)
        out_cols.extend(new_cols)
    if lag_frames:
        work = pd.concat([work] + lag_frames, axis=1)
    return work, out_cols


class LinearForecastBackend:
    name = "linear"

    def build_trajectory(
        self,
        prepared: _PreparedV2BacktestData,
        *,
        retrain_days: int = 20,
    ) -> _BacktestTrajectory:
        settings = prepared.settings
        market_valid = prepared.market_valid
        panel = prepared.panel
        market_feature_cols = prepared.market_feature_cols
        feature_cols = prepared.feature_cols
        dates = prepared.dates
        min_train_days = int(settings["min_train_days"])
        steps: list[_TrajectoryStep] = []

        for block_start in range(min_train_days, len(dates) - 1, max(1, int(retrain_days))):
            train_dates = set(dates[:block_start])
            market_train = market_valid[market_valid["date"].isin(train_dates)].copy()
            if market_train.empty:
                continue
            market_short_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_1d_up",
            )
            market_five_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_5d_up",
            )
            market_mid_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_20d_up",
            )
            panel_train = panel[panel["date"].isin(train_dates)].copy()
            if panel_train.empty:
                continue
            panel_short_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_1d_excess_mkt_up",
            )
            panel_five_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_5d_excess_mkt_up",
            )
            panel_mid_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_20d_excess_sector_up",
            )
            panel_short_q_models = _fit_quantile_quintet(
                panel_train,
                feature_cols=feature_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
            panel_mid_q_models = _fit_quantile_quintet(
                panel_train,
                feature_cols=feature_cols,
                target_col="excess_ret_20_vs_sector",
                l2=float(settings["l2"]),
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_row = market_valid[market_valid["date"] == date].copy()
                if market_row.empty:
                    continue
                mkt_short = float(market_short_model.predict_proba(market_row, market_feature_cols)[0])
                mkt_five = float(market_five_model.predict_proba(market_row, market_feature_cols)[0])
                mkt_mid = float(market_mid_model.predict_proba(market_row, market_feature_cols)[0])
                market_state, cross_section = _build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=market_valid[market_valid["date"] <= date].copy(),
                    market_short_prob=mkt_short,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_row = panel[panel["date"] == date].copy()
                stock_states, scored_rows = _build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=feature_cols,
                    short_model=panel_short_model,
                    five_model=panel_five_model,
                    mid_model=panel_mid_model,
                    short_q_models=panel_short_q_models,
                    mid_q_models=panel_mid_q_models,
                )
                if not stock_states:
                    continue
                grouped: dict[str, list[StockForecastState]] = {}
                for stock_state in stock_states:
                    grouped.setdefault(stock_state.sector, []).append(stock_state)
                sector_states: list[SectorForecastState] = []
                for sector, items in grouped.items():
                    n = max(1, len(items))
                    up5 = sum(item.up_5d_prob for item in items) / n
                    up20 = sum(item.up_20d_prob for item in items) / n
                    rel = sum(item.up_20d_prob - 0.5 for item in items) / n
                    rotation = sum(abs(item.up_5d_prob - item.up_20d_prob) for item in items) / n
                    crowding = sum(max(0.0, item.up_20d_prob - 0.5) for item in items) / n
                    sector_states.append(
                        SectorForecastState(
                            sector=sector,
                            up_5d_prob=float(up5),
                            up_20d_prob=float(up20),
                            relative_strength=float(rel),
                            rotation_speed=float(_clip(rotation, 0.0, 1.0)),
                            crowding_score=float(_clip(crowding, 0.0, 1.0)),
                        )
                    )
                composite_state = compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    _TrajectoryStep(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=_panel_horizon_metrics(scored_rows),
                    )
                )

        return _BacktestTrajectory(prepared=prepared, steps=steps)


class DeepForecastBackend:
    name = "deep"

    def build_trajectory(
        self,
        prepared: _PreparedV2BacktestData,
        *,
        retrain_days: int = 20,
    ) -> _BacktestTrajectory:
        settings = prepared.settings
        market_valid = prepared.market_valid
        panel = prepared.panel
        dates = prepared.dates
        min_train_days = int(settings["min_train_days"])
        steps: list[_TrajectoryStep] = []

        tensor_market, tensor_market_cols = _tensorize_temporal_frame(
            market_valid,
            feature_cols=prepared.market_feature_cols,
            group_col=None,
            lag_depth=3,
        )
        tensor_panel, tensor_panel_cols = _tensorize_temporal_frame(
            panel,
            feature_cols=prepared.feature_cols,
            group_col="symbol",
            lag_depth=3,
        )

        for block_start in range(min_train_days, len(dates) - 1, max(1, int(retrain_days))):
            train_dates = set(dates[:block_start])
            market_train = tensor_market[tensor_market["date"].isin(train_dates)].copy()
            if market_train.empty:
                continue
            market_short_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_1d_up",
            )
            market_five_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_5d_up",
            )
            market_mid_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_20d_up",
            )
            panel_train = tensor_panel[tensor_panel["date"].isin(train_dates)].copy()
            if panel_train.empty:
                continue
            panel_short_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_1d_excess_mkt_up",
            )
            panel_five_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_5d_excess_mkt_up",
            )
            panel_mid_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_20d_excess_sector_up",
            )
            panel_short_q_models = _fit_mlp_quantile_quintet(
                panel_train,
                feature_cols=tensor_panel_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
            panel_mid_q_models = _fit_mlp_quantile_quintet(
                panel_train,
                feature_cols=tensor_panel_cols,
                target_col="excess_ret_20_vs_sector",
                l2=float(settings["l2"]),
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_row = tensor_market[tensor_market["date"] == date].copy()
                if market_row.empty:
                    continue
                mkt_short = float(market_short_model.predict_proba(market_row, tensor_market_cols)[0])
                mkt_five = float(market_five_model.predict_proba(market_row, tensor_market_cols)[0])
                mkt_mid = float(market_mid_model.predict_proba(market_row, tensor_market_cols)[0])
                market_state, cross_section = _build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=market_valid[market_valid["date"] <= date].copy(),
                    market_short_prob=mkt_short,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_row = tensor_panel[tensor_panel["date"] == date].copy()
                stock_states, scored_rows = _build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=tensor_panel_cols,
                    short_model=panel_short_model,
                    five_model=panel_five_model,
                    mid_model=panel_mid_model,
                    short_q_models=panel_short_q_models,
                    mid_q_models=panel_mid_q_models,
                )
                if not stock_states:
                    continue
                grouped: dict[str, list[StockForecastState]] = {}
                for stock_state in stock_states:
                    grouped.setdefault(stock_state.sector, []).append(stock_state)
                sector_states: list[SectorForecastState] = []
                for sector, items in grouped.items():
                    n = max(1, len(items))
                    up5 = sum(item.up_5d_prob for item in items) / n
                    up20 = sum(item.up_20d_prob for item in items) / n
                    rel = sum(item.up_20d_prob - 0.5 for item in items) / n
                    rotation = sum(abs(item.up_5d_prob - item.up_20d_prob) for item in items) / n
                    crowding = sum(max(0.0, item.up_20d_prob - 0.5) for item in items) / n
                    sector_states.append(
                        SectorForecastState(
                            sector=sector,
                            up_5d_prob=float(up5),
                            up_20d_prob=float(up20),
                            relative_strength=float(rel),
                            rotation_speed=float(_clip(rotation, 0.0, 1.0)),
                            crowding_score=float(_clip(crowding, 0.0, 1.0)),
                        )
                    )
                composite_state = compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    _TrajectoryStep(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=_panel_horizon_metrics(scored_rows),
                    )
                )

        return _BacktestTrajectory(prepared=prepared, steps=steps)


def _make_forecast_backend(name: str | None) -> ForecastBackend:
    backend = (str(name).strip().lower() if name is not None else "linear") or "linear"
    if backend == "linear":
        return LinearForecastBackend()
    if backend == "deep":
        return DeepForecastBackend()
    raise ValueError(f"Unsupported forecast backend: {backend}")


def _trajectory_cache_key(
    *,
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    retrain_days: int,
    forecast_backend: str,
) -> str:
    payload = {
        "version": "v2-trajectory-cache-1",
        "config_path": str(Path(config_path).resolve()),
        "source": "" if source is None else str(source),
        "universe_file": "" if universe_file is None else str(Path(universe_file).resolve()),
        "universe_limit": -1 if universe_limit is None else int(universe_limit),
        "retrain_days": int(retrain_days),
        "forecast_backend": str(forecast_backend),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _trajectory_cache_path(
    *,
    cache_root: str,
    cache_key: str,
) -> Path:
    root = Path(str(cache_root))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cache_key}.pkl"


def _file_mtime_token(path_like: object) -> int:
    try:
        path = Path(str(path_like))
    except Exception:
        return 0
    if not path.exists():
        return 0
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return 0


def _daily_result_cache_key(
    *,
    strategy_id: str,
    settings: dict[str, object],
    artifact_root: str,
) -> str:
    policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    payload = {
        "version": "v2-daily-cache-1",
        "strategy_id": str(strategy_id),
        "config_path": str(Path(str(settings.get("config_path", ""))).resolve()),
        "source": str(settings.get("source", "")),
        "watchlist": str(Path(str(settings.get("watchlist", ""))).resolve()),
        "watchlist_mtime": _file_mtime_token(settings.get("watchlist", "")),
        "universe_file": str(Path(str(settings.get("universe_file", ""))).resolve()),
        "universe_mtime": _file_mtime_token(settings.get("universe_file", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "min_train_days": int(settings.get("min_train_days", 0)),
        "step_days": int(settings.get("step_days", 0)),
        "l2": float(settings.get("l2", 0.0)),
        "max_positions": int(settings.get("max_positions", 0)),
        "use_margin_features": bool(settings.get("use_margin_features", False)),
        "margin_market_file": str(settings.get("margin_market_file", "")),
        "margin_market_mtime": _file_mtime_token(settings.get("margin_market_file", "")),
        "margin_stock_file": str(settings.get("margin_stock_file", "")),
        "margin_stock_mtime": _file_mtime_token(settings.get("margin_stock_file", "")),
        "published_policy_path": str(policy_path.resolve()),
        "published_policy_mtime": _file_mtime_token(policy_path),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _daily_result_cache_path(
    *,
    cache_root: str,
    cache_key: str,
) -> Path:
    root = Path(str(cache_root))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"daily_{cache_key}.pkl"


def _load_or_build_v2_backtest_trajectory(
    *,
    config_path: str,
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    retrain_days: int = 20,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
) -> _BacktestTrajectory | None:
    backend = _make_forecast_backend(forecast_backend)
    cache_key = _trajectory_cache_key(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        retrain_days=retrain_days,
        forecast_backend=backend.name,
    )
    cache_path = _trajectory_cache_path(cache_root=cache_root, cache_key=cache_key)
    if not refresh_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached is not None:
                return cached
        except Exception:
            pass

    prepared = _prepare_v2_backtest_data(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    if prepared is None:
        return None
    trajectory = _build_v2_backtest_trajectory_from_prepared(
        prepared,
        retrain_days=retrain_days,
        forecast_backend=backend.name,
    )
    try:
        with cache_path.open("wb") as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return trajectory


def _build_v2_backtest_trajectory_from_prepared(
    prepared: _PreparedV2BacktestData,
    *,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
) -> _BacktestTrajectory:
    backend = _make_forecast_backend(forecast_backend)
    return backend.build_trajectory(prepared, retrain_days=retrain_days)


def _execute_v2_backtest_trajectory(
    trajectory: _BacktestTrajectory,
    *,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    _ = retrain_days
    commission_rate = max(0.0, float(commission_bps)) / 10000.0
    slippage_rate = max(0.0, float(slippage_bps)) / 10000.0
    returns: list[float] = []
    benchmark_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    fill_ratios: list[float] = []
    slippage_cost_bps: list[float] = []
    rank_ics: list[float] = []
    top_decile_returns: list[float] = []
    top_bottom_spreads: list[float] = []
    top_k_hit_rates: list[float] = []
    horizon_metric_series: dict[str, dict[str, list[float]]] = {
        "1d": {"rank_ic": [], "top_decile_return": [], "top_bottom_spread": [], "top_k_hit_rate": []},
        "5d": {"rank_ic": [], "top_decile_return": [], "top_bottom_spread": [], "top_k_hit_rate": []},
        "20d": {"rank_ic": [], "top_decile_return": [], "top_bottom_spread": [], "top_k_hit_rate": []},
    }
    out_dates: list[pd.Timestamp] = []
    prev_weights: dict[str, float] = {}
    prev_cash = 1.0
    learning_rows: list[dict[str, float]] = []

    for step in trajectory.steps:
        rank_ics.append(float(step.horizon_metrics["20d"]["rank_ic"]))
        top_decile_returns.append(float(step.horizon_metrics["20d"]["top_decile_return"]))
        top_bottom_spreads.append(float(step.horizon_metrics["20d"]["top_bottom_spread"]))
        top_k_hit_rates.append(float(step.horizon_metrics["20d"]["top_k_hit_rate"]))
        for horizon, metric_map in step.horizon_metrics.items():
            for name, value in metric_map.items():
                horizon_metric_series[horizon][name].append(float(value))

        active_policy_spec = policy_spec
        if learned_policy is not None:
            active_policy_spec = _policy_spec_from_model(
                state=step.composite_state,
                model=learned_policy,
            )
        decision = apply_policy(
            PolicyInput(
                composite_state=step.composite_state,
                current_weights=prev_weights,
                current_cash=max(0.0, prev_cash),
                total_equity=1.0,
            ),
            policy_spec=active_policy_spec,
        )
        gross_ret = float(
            sum(
                float(weight) * _safe_float(
                    trajectory.prepared.stock_frames[symbol][trajectory.prepared.stock_frames[symbol]["date"] == step.date].iloc[0]["fwd_ret_1"],
                    0.0,
                )
                for symbol, weight in decision.symbol_target_weights.items()
                if symbol in trajectory.prepared.stock_frames
                and not trajectory.prepared.stock_frames[symbol][trajectory.prepared.stock_frames[symbol]["date"] == step.date].empty
            )
        )
        daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
            date=step.date,
            next_date=step.next_date,
            decision=decision,
            current_weights=prev_weights,
            current_cash=prev_cash,
            stock_states=step.stock_states,
            stock_frames=trajectory.prepared.stock_frames,
            total_commission_rate=commission_rate,
            base_slippage_rate=slippage_rate,
        )
        benchmark_row = trajectory.prepared.market_valid[trajectory.prepared.market_valid["date"] == step.date]
        benchmark_ret = 0.0
        if not benchmark_row.empty:
            benchmark_ret = _safe_float(benchmark_row.iloc[0].get("mkt_fwd_ret_1", 0.0), 0.0)
        returns.append(float(daily_ret))
        benchmark_returns.append(float(benchmark_ret))
        gross_returns.append(float(gross_ret))
        turnovers.append(turnover)
        costs.append(cost)
        fill_ratios.append(fill_ratio)
        slippage_cost_bps.append(slip_bps)
        out_dates.append(step.next_date)
        prev_weights = next_weights
        prev_cash = next_cash

        if capture_learning_rows:
            target_exposure, target_positions, target_turnover = _derive_learning_targets(
                state=step.composite_state,
                stock_frames=trajectory.prepared.stock_frames,
                date=step.date,
            )
            row = {
                name: float(value)
                for name, value in zip(_policy_feature_names(), _policy_feature_vector(step.composite_state))
            }
            row.update(
                {
                    "target_exposure": float(target_exposure),
                    "target_positions": float(target_positions),
                    "target_turnover": float(target_turnover),
                }
            )
            learning_rows.append(row)

    return (
        _to_v2_backtest_summary(
            returns=returns,
            benchmark_returns=benchmark_returns,
            turnovers=turnovers,
            costs=costs,
            gross_returns=gross_returns,
            fill_ratios=fill_ratios,
            slippage_bps=slippage_cost_bps,
            rank_ics=rank_ics,
            top_decile_returns=top_decile_returns,
            top_bottom_spreads=top_bottom_spreads,
            top_k_hit_rates=top_k_hit_rates,
            horizon_metrics=horizon_metric_series,
            dates=out_dates,
        ),
        learning_rows,
    )


def _run_v2_backtest_core(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    _ = strategy_id
    if trajectory is None:
        trajectory = _load_or_build_v2_backtest_trajectory(
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
        )
        if trajectory is None:
            return _empty_v2_backtest_result()
    return _execute_v2_backtest_trajectory(
        trajectory,
        policy_spec=policy_spec,
        learned_policy=learned_policy,
        retrain_days=retrain_days,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        capture_learning_rows=capture_learning_rows,
    )


def run_v2_backtest_live(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
) -> V2BacktestSummary:
    summary, _ = _run_v2_backtest_core(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        policy_spec=policy_spec,
        learned_policy=learned_policy,
        retrain_days=retrain_days,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        capture_learning_rows=False,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    return summary


def calibrate_v2_policy(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    baseline: V2BacktestSummary | None = None,
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
) -> V2CalibrationResult:
    baseline_spec = PolicySpec()
    baseline = baseline or run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        policy_spec=baseline_spec,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    candidates = [
        baseline_spec,
        PolicySpec(risk_on_exposure=0.75, cautious_exposure=0.50, risk_off_exposure=0.25),
        PolicySpec(risk_on_exposure=0.90, cautious_exposure=0.65, risk_off_exposure=0.35),
        PolicySpec(risk_on_positions=5, cautious_positions=3, risk_off_positions=1),
    ]
    best_spec = baseline_spec
    best_summary = baseline
    best_score = float(baseline.annual_return) - 0.5 * abs(float(baseline.max_drawdown))
    trials: list[dict[str, object]] = [
        {
            "policy": asdict(baseline_spec),
            "summary": asdict(baseline),
            "score": float(best_score),
        }
    ]
    for spec in candidates[1:]:
        summary = run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            policy_spec=spec,
            trajectory=trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
        )
        score = float(summary.annual_return) - 0.5 * abs(float(summary.max_drawdown))
        trials.append(
            {
                "policy": asdict(spec),
                "summary": asdict(summary),
                "score": float(score),
            }
        )
        if score > best_score:
            best_score = float(score)
            best_spec = spec
            best_summary = summary
    return V2CalibrationResult(
        best_policy=best_spec,
        best_score=float(best_score),
        baseline=baseline,
        calibrated=best_summary,
        trials=trials,
    )


def learn_v2_policy_model(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    l2: float = 1.0,
    baseline: V2BacktestSummary | None = None,
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
) -> V2PolicyLearningResult:
    baseline = baseline or run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    _, rows = _run_v2_backtest_core(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        capture_learning_rows=True,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    feature_names = _policy_feature_names()
    if not rows:
        model = LearnedPolicyModel(
            feature_names=feature_names,
            exposure_intercept=0.60,
            exposure_coef=[0.0] * len(feature_names),
            position_intercept=3.0,
            position_coef=[0.0] * len(feature_names),
            turnover_intercept=0.22,
            turnover_coef=[0.0] * len(feature_names),
            train_rows=0,
            train_r2_exposure=0.0,
            train_r2_positions=0.0,
            train_r2_turnover=0.0,
        )
        learned_summary = run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            learned_policy=model,
            trajectory=trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
        )
        return V2PolicyLearningResult(model=model, baseline=baseline, learned=learned_summary)

    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y_exposure = np.asarray([float(row["target_exposure"]) for row in rows], dtype=float)
    y_positions = np.asarray([float(row["target_positions"]) for row in rows], dtype=float)
    y_turnover = np.asarray([float(row["target_turnover"]) for row in rows], dtype=float)

    exp_b, exp_w = _fit_ridge_regression(X, y_exposure, l2=l2)
    pos_b, pos_w = _fit_ridge_regression(X, y_positions, l2=l2)
    turn_b, turn_w = _fit_ridge_regression(X, y_turnover, l2=l2)

    pred_exp = np.asarray([_predict_ridge(row, exp_b, exp_w) for row in X], dtype=float)
    pred_pos = np.asarray([_predict_ridge(row, pos_b, pos_w) for row in X], dtype=float)
    pred_turn = np.asarray([_predict_ridge(row, turn_b, turn_w) for row in X], dtype=float)

    model = LearnedPolicyModel(
        feature_names=feature_names,
        exposure_intercept=float(exp_b),
        exposure_coef=[float(x) for x in exp_w.tolist()],
        position_intercept=float(pos_b),
        position_coef=[float(x) for x in pos_w.tolist()],
        turnover_intercept=float(turn_b),
        turnover_coef=[float(x) for x in turn_w.tolist()],
        train_rows=int(len(rows)),
        train_r2_exposure=float(_r2_score(y_exposure, pred_exp)),
        train_r2_positions=float(_r2_score(y_positions, pred_pos)),
        train_r2_turnover=float(_r2_score(y_turnover, pred_turn)),
    )
    learned_summary = run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        learned_policy=model,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    return V2PolicyLearningResult(
        model=model,
        baseline=baseline,
        learned=learned_summary,
    )


def _baseline_only_calibration(baseline: V2BacktestSummary) -> V2CalibrationResult:
    baseline_spec = PolicySpec()
    score = float(baseline.annual_return) - 0.5 * abs(float(baseline.max_drawdown))
    return V2CalibrationResult(
        best_policy=baseline_spec,
        best_score=float(score),
        baseline=baseline,
        calibrated=baseline,
        trials=[
            {
                "policy": asdict(baseline_spec),
                "summary": asdict(baseline),
                "score": float(score),
            }
        ],
    )


def _placeholder_learning_result(baseline: V2BacktestSummary) -> V2PolicyLearningResult:
    model = LearnedPolicyModel(
        feature_names=_policy_feature_names(),
        exposure_intercept=0.60,
        exposure_coef=[0.0] * len(_policy_feature_names()),
        position_intercept=3.0,
        position_coef=[0.0] * len(_policy_feature_names()),
        turnover_intercept=0.22,
        turnover_coef=[0.0] * len(_policy_feature_names()),
        train_rows=0,
        train_r2_exposure=0.0,
        train_r2_positions=0.0,
        train_r2_turnover=0.0,
    )
    return V2PolicyLearningResult(
        model=model,
        baseline=baseline,
        learned=baseline,
    )


def run_v2_research_workflow(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    skip_calibration: bool = False,
    skip_learning: bool = False,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
) -> tuple[V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult]:
    trajectory = _load_or_build_v2_backtest_trajectory(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    baseline = run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
    )
    calibration = (
        _baseline_only_calibration(baseline)
        if skip_calibration
        else calibrate_v2_policy(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            baseline=baseline,
            trajectory=trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
        )
    )
    learning = (
        _placeholder_learning_result(baseline)
        if skip_learning
        else learn_v2_policy_model(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            baseline=baseline,
            trajectory=trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
        )
    )
    return baseline, calibration, learning


def load_published_v2_policy_model(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
) -> LearnedPolicyModel | None:
    model_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    if not model_path.exists():
        return None
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return LearnedPolicyModel(
        feature_names=[str(x) for x in payload.get("feature_names", [])],
        exposure_intercept=float(payload.get("exposure_intercept", 0.60)),
        exposure_coef=[float(x) for x in payload.get("exposure_coef", [])],
        position_intercept=float(payload.get("position_intercept", 3.0)),
        position_coef=[float(x) for x in payload.get("position_coef", [])],
        turnover_intercept=float(payload.get("turnover_intercept", 0.22)),
        turnover_coef=[float(x) for x in payload.get("turnover_coef", [])],
        train_rows=int(payload.get("train_rows", 0)),
        train_r2_exposure=float(payload.get("train_r2_exposure", 0.0)),
        train_r2_positions=float(payload.get("train_r2_positions", 0.0)),
        train_r2_turnover=float(payload.get("train_r2_turnover", 0.0)),
    )


def publish_v2_research_artifacts(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    settings: dict[str, object] | None = None,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
) -> dict[str, str]:
    settings = settings or _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(str(artifact_root)) / str(strategy_id) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    symbols = []
    universe_path = Path(str(settings.get("universe_file", "")))
    if universe_path.exists():
        try:
            raw = json.loads(universe_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                symbols = [str(item) for item in raw]
        except Exception:
            symbols = []

    dataset_manifest = {
        "strategy_id": str(strategy_id),
        "config_path": str(settings.get("config_path", "")),
        "source": str(settings.get("source", "")),
        "watchlist": str(settings.get("watchlist", "")),
        "universe_file": str(settings.get("universe_file", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "symbols": symbols,
        "symbol_count": len(symbols),
    }
    calibration_manifest = {
        "best_score": float(calibration.best_score),
        "best_policy": asdict(calibration.best_policy),
        "trials": calibration.trials,
    }
    learning_manifest = asdict(learning.model)
    backtest_manifest = {
        "baseline": asdict(baseline),
        "calibrated": asdict(calibration.calibrated),
        "learned": asdict(learning.learned),
    }

    dataset_path = base_dir / "dataset_manifest.json"
    calibration_path = base_dir / "policy_calibration.json"
    learning_path = base_dir / "learned_policy_model.json"
    backtest_path = base_dir / "backtest_summary.json"
    manifest_path = base_dir / "research_manifest.json"
    latest_policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    latest_manifest_path = Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    learning_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    backtest_path.write_text(json.dumps(backtest_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "strategy_id": str(strategy_id),
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "backtest_summary": str(backtest_path),
        "published_policy_model": str(latest_policy_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_policy_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_dir": str(base_dir),
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "backtest_summary": str(backtest_path),
        "research_manifest": str(manifest_path),
        "published_policy_model": str(latest_policy_path),
    }


def run_daily_v2_live(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    artifact_root: str = "artifacts/v2",
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
) -> DailyRunResult:
    settings = _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    cache_key = _daily_result_cache_key(
        strategy_id=strategy_id,
        settings=settings,
        artifact_root=artifact_root,
    )
    cache_path = _daily_result_cache_path(
        cache_root=cache_root,
        cache_key=cache_key,
    )
    if not refresh_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, DailyRunResult):
                return cached
        except Exception:
            pass
    snapshot = build_strategy_snapshot(
        strategy_id=strategy_id,
        universe_id=Path(str(settings["universe_file"])).stem or "v2_universe",
    )

    market_security, current_holdings, base_sector_map = load_watchlist(str(settings["watchlist"]))
    universe = build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows or current_holdings
    sector_map = {
        stock.symbol: (stock.sector or base_sector_map.get(stock.symbol, "其他"))
        for stock in stocks
    }

    market_forecast, stock_rows = run_quant_pipeline(
        market_security=market_security,
        stock_securities=stocks,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        min_train_days=int(settings["min_train_days"]),
        step_days=int(settings["step_days"]),
        l2=float(settings["l2"]),
        max_positions=int(settings["max_positions"]),
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
        margin_stock_file=str(settings["margin_stock_file"]),
        enable_walk_forward_eval=False,
    )

    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_state, cross_section = _build_market_and_cross_section_states(
        market_symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
        market_short_prob=float(market_forecast.short_prob),
        market_five_prob=float(market_forecast.five_prob),
        market_mid_prob=float(market_forecast.mid_prob),
    )
    sector_frames = build_sector_daily_frames(
        stock_securities=stocks,
        sector_map=sector_map,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    sector_records = run_sector_forecast(
        sector_frames=sector_frames,
        market_raw=market_raw,
        l2=float(settings["l2"]),
    )
    sectors = [
        SectorForecastState(
            sector=item.sector,
            up_5d_prob=float(item.up_5d_prob),
            up_20d_prob=float(item.up_20d_prob),
            relative_strength=float(item.excess_vs_market_prob - 0.5),
            rotation_speed=float(item.rotation_speed),
            crowding_score=float(item.crowding_score),
        )
        for item in sector_records
    ]
    sector_strength_map = {item.sector: float(item.excess_vs_market_prob - 0.5) for item in sector_records}
    stocks_state = _build_stock_states_from_rows(stock_rows, sector_map, sector_strength_map=sector_strength_map)
    composite_state = compose_state(
        market=market_state,
        sectors=sectors,
        stocks=stocks_state,
        cross_section=cross_section,
    )

    current_weights = {}
    if current_holdings:
        equal_weight = 1.0 / float(len(current_holdings))
        current_weights = {item.symbol: float(equal_weight) for item in current_holdings}

    learned_policy = load_published_v2_policy_model(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
    )
    active_policy_spec = None
    if learned_policy is not None:
        active_policy_spec = _policy_spec_from_model(
            state=composite_state,
            model=learned_policy,
        )

    policy_decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights=current_weights,
            current_cash=max(0.0, 1.0 - sum(current_weights.values())),
            total_equity=1.0,
        ),
        policy_spec=active_policy_spec,
    )
    trade_actions = build_trade_actions(
        decision=policy_decision,
        current_weights=current_weights,
    )
    result = DailyRunResult(
        snapshot=snapshot,
        composite_state=composite_state,
        policy_decision=policy_decision,
        trade_actions=trade_actions,
    )
    try:
        with cache_path.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return result


def summarize_daily_run(result: DailyRunResult) -> dict[str, object]:
    return {
        "strategy_id": result.snapshot.strategy_id,
        "strategy_mode": result.composite_state.strategy_mode,
        "risk_regime": result.composite_state.risk_regime,
        "market": asdict(result.composite_state.market),
        "policy": asdict(result.policy_decision),
        "trade_plan": [
            {
                "symbol": action.symbol,
                "action": action.action,
                "current_weight": action.current_weight,
                "target_weight": action.target_weight,
                "delta_weight": action.delta_weight,
                "note": action.note,
            }
            for action in result.trade_actions
        ],
    }


def summarize_v2_backtest(result: V2BacktestSummary) -> dict[str, object]:
    payload = asdict(result)
    payload.pop("nav_curve", None)
    payload.pop("benchmark_nav_curve", None)
    payload.pop("excess_nav_curve", None)
    payload.pop("curve_dates", None)
    return payload


def summarize_v2_calibration(result: V2CalibrationResult) -> dict[str, object]:
    return {
        "best_score": float(result.best_score),
        "best_policy": asdict(result.best_policy),
        "baseline": summarize_v2_backtest(result.baseline),
        "calibrated": summarize_v2_backtest(result.calibrated),
        "trial_count": len(result.trials),
    }


def summarize_v2_policy_learning(result: V2PolicyLearningResult) -> dict[str, object]:
    return {
        "model": asdict(result.model),
        "baseline": summarize_v2_backtest(result.baseline),
        "learned": summarize_v2_backtest(result.learned),
    }
