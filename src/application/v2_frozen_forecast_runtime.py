from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.application.v2_contracts import CompositeState


@dataclass(frozen=True)
class FrozenForecastRuntimeDependencies:
    load_symbol_daily: Callable[..., pd.DataFrame]
    make_market_feature_frame: Callable[[pd.DataFrame], pd.DataFrame]
    build_market_context_features: Callable[..., object]
    deserialize_binary_model: Callable[[dict[str, object]], object]
    deserialize_quantile_bundle: Callable[[object], object]
    predict_quantile_profile: Callable[..., object]
    build_market_and_cross_section_from_prebuilt_frame: Callable[..., tuple[object, object]]
    build_stock_live_panel_dataset: Callable[..., object]
    build_stock_states_from_panel_slice: Callable[..., tuple[list[object], pd.DataFrame]]
    build_sector_states: Callable[..., list[object]]
    stock_policy_score: Callable[[object], float]
    compose_state: Callable[..., CompositeState]


@dataclass(frozen=True)
class FrozenForecastBundleDependencies:
    binary_model_cls: Callable[..., Any]
    serialize_binary_model: Callable[[Any], dict[str, object]]
    fit_quantile_quintet: Callable[..., object]
    serialize_quantile_bundle: Callable[[object], object]


def load_frozen_forecast_bundle(
    path_like: object,
    *,
    load_json_dict: Callable[[object], dict[str, object]],
    forecast_bundle_cls: type,
) -> dict[str, object]:
    payload = load_json_dict(path_like)
    if not payload:
        return {}
    if str(payload.get("backend", "")).strip().lower() not in {"linear", "deep"}:
        return {}
    forecast_bundle_cls.from_payload(payload)
    return payload


def build_frozen_linear_forecast_bundle(
    prepared: object,
    *,
    deps: FrozenForecastBundleDependencies,
) -> dict[str, object]:
    settings = getattr(prepared, "settings")
    market_train = getattr(prepared, "market_valid").copy()
    panel_train = getattr(prepared, "panel").copy()
    market_feature_cols = list(getattr(prepared, "market_feature_cols"))
    panel_feature_cols = list(getattr(prepared, "feature_cols"))
    l2 = float(settings["l2"])

    market_models = {
        "1d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(market_train, market_feature_cols, "mkt_target_1d_up")
        ),
        "2d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(market_train, market_feature_cols, "mkt_target_2d_up")
        ),
        "3d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(market_train, market_feature_cols, "mkt_target_3d_up")
        ),
        "5d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(market_train, market_feature_cols, "mkt_target_5d_up")
        ),
        "20d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(market_train, market_feature_cols, "mkt_target_20d_up")
        ),
    }
    market_quantiles = {
        "1d": deps.serialize_quantile_bundle(
            deps.fit_quantile_quintet(
                market_train,
                feature_cols=market_feature_cols,
                target_col="mkt_fwd_ret_1",
                l2=l2,
            )
        ),
        "20d": deps.serialize_quantile_bundle(
            deps.fit_quantile_quintet(
                market_train,
                feature_cols=market_feature_cols,
                target_col="mkt_fwd_ret_20",
                l2=l2,
            )
        ),
    }
    stock_models = {
        "1d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(panel_train, panel_feature_cols, "target_1d_excess_mkt_up")
        ),
        "2d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(panel_train, panel_feature_cols, "target_2d_excess_mkt_up")
        ),
        "3d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(panel_train, panel_feature_cols, "target_3d_excess_mkt_up")
        ),
        "5d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(panel_train, panel_feature_cols, "target_5d_excess_mkt_up")
        ),
        "20d": deps.serialize_binary_model(
            deps.binary_model_cls(l2=l2).fit(panel_train, panel_feature_cols, "target_20d_excess_sector_up")
        ),
    }
    stock_quantiles = {
        "1d": deps.serialize_quantile_bundle(
            deps.fit_quantile_quintet(
                panel_train,
                feature_cols=panel_feature_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=l2,
            )
        ),
        "20d": deps.serialize_quantile_bundle(
            deps.fit_quantile_quintet(
                panel_train,
                feature_cols=panel_feature_cols,
                target_col="excess_ret_20_vs_sector",
                l2=l2,
            )
        ),
    }
    return {
        "format_version": 1,
        "backend": "linear",
        "created_from_end_date": str(pd.Timestamp(max(getattr(prepared, "dates"))).date()),
        "market_feature_cols": market_feature_cols,
        "panel_feature_cols": panel_feature_cols,
        "market_models": market_models,
        "market_quantiles": market_quantiles,
        "stock_models": stock_models,
        "stock_quantiles": stock_quantiles,
    }


def build_live_market_frame(
    *,
    settings: dict[str, object],
    market_symbol: str,
    deps: FrozenForecastRuntimeDependencies,
) -> pd.DataFrame:
    market_raw = deps.load_symbol_daily(
        symbol=market_symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_feat_base = deps.make_market_feature_frame(market_raw)
    market_context = deps.build_market_context_features(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_dates=market_feat_base["date"],
        use_margin_features=bool(settings.get("use_margin_features", False)),
        margin_market_file=str(settings.get("margin_market_file", "")),
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
        use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
        use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
        cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
    )
    return market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")


def score_live_composite_state_from_frozen_bundle(
    *,
    bundle: dict[str, object],
    settings: dict[str, object],
    universe_ctx: object,
    deps: FrozenForecastRuntimeDependencies,
) -> tuple[CompositeState | None, list[object]]:
    if str(bundle.get("backend", "")).strip().lower() != "linear":
        return None, []

    market_feature_cols = [str(item) for item in bundle.get("market_feature_cols", [])]
    panel_feature_cols = [str(item) for item in bundle.get("panel_feature_cols", [])]
    if not market_feature_cols or not panel_feature_cols:
        return None, []

    market_frame = build_live_market_frame(
        settings=settings,
        market_symbol=str(getattr(universe_ctx.market_security, "symbol", "")),
        deps=deps,
    )
    if market_frame.empty:
        return None, []
    latest_market = market_frame.sort_values("date").iloc[[-1]].copy()
    if any(col not in latest_market.columns for col in market_feature_cols):
        return None, []
    latest_market = latest_market.dropna(subset=market_feature_cols)
    if latest_market.empty:
        return None, []

    market_models_raw = bundle.get("market_models", {})
    market_quantiles_raw = bundle.get("market_quantiles", {})
    if not isinstance(market_models_raw, dict) or not isinstance(market_quantiles_raw, dict):
        return None, []
    market_short_model = deps.deserialize_binary_model(dict(market_models_raw.get("1d", {})))
    market_two_model = deps.deserialize_binary_model(dict(market_models_raw.get("2d", {})))
    market_three_model = deps.deserialize_binary_model(dict(market_models_raw.get("3d", {})))
    market_five_model = deps.deserialize_binary_model(dict(market_models_raw.get("5d", {})))
    market_mid_model = deps.deserialize_binary_model(dict(market_models_raw.get("20d", {})))
    market_short_q = deps.deserialize_quantile_bundle(market_quantiles_raw.get("1d"))
    market_mid_q = deps.deserialize_quantile_bundle(market_quantiles_raw.get("20d"))

    mkt_short = float(market_short_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_two = float(market_two_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_three = float(market_three_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_five = float(market_five_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_mid = float(market_mid_model.predict_proba(latest_market, market_feature_cols)[0])
    market_short_profile = deps.predict_quantile_profile(
        latest_market,
        feature_cols=market_feature_cols,
        q_models=market_short_q,
    )
    market_mid_profile = deps.predict_quantile_profile(
        latest_market,
        feature_cols=market_feature_cols,
        q_models=market_mid_q,
    )
    market_state, cross_section = deps.build_market_and_cross_section_from_prebuilt_frame(
        market_frame=market_frame,
        market_short_prob=mkt_short,
        market_two_prob=mkt_two,
        market_three_prob=mkt_three,
        market_five_prob=mkt_five,
        market_mid_prob=mkt_mid,
        market_short_profile=market_short_profile,
        market_mid_profile=market_mid_profile,
    )

    panel_bundle = deps.build_stock_live_panel_dataset(
        stock_securities=universe_ctx.stocks,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_frame=market_frame,
        extra_market_cols=[
            col for col in market_frame.columns
            if col.startswith("us_")
            or col.startswith("cn_")
            or col.startswith("breadth_")
            or col.startswith("fin_")
            or col.startswith("sec_")
        ],
        use_margin_features=bool(settings.get("use_margin_features", False)),
        margin_stock_file=str(settings.get("margin_stock_file", "")),
    )
    live_panel = panel_bundle.frame
    if live_panel.empty or any(col not in live_panel.columns for col in panel_feature_cols):
        return None, []
    latest_panel_date = pd.to_datetime(live_panel["date"], errors="coerce").dropna().max()
    latest_panel = live_panel[live_panel["date"] == latest_panel_date].copy()
    latest_panel = latest_panel.dropna(subset=panel_feature_cols)
    if latest_panel.empty:
        return None, []

    stock_models_raw = bundle.get("stock_models", {})
    stock_quantiles_raw = bundle.get("stock_quantiles", {})
    if not isinstance(stock_models_raw, dict) or not isinstance(stock_quantiles_raw, dict):
        return None, []
    stock_states, _ = deps.build_stock_states_from_panel_slice(
        panel_row=latest_panel,
        feature_cols=panel_feature_cols,
        short_model=deps.deserialize_binary_model(dict(stock_models_raw.get("1d", {}))),
        two_model=deps.deserialize_binary_model(dict(stock_models_raw.get("2d", {}))),
        three_model=deps.deserialize_binary_model(dict(stock_models_raw.get("3d", {}))),
        five_model=deps.deserialize_binary_model(dict(stock_models_raw.get("5d", {}))),
        mid_model=deps.deserialize_binary_model(dict(stock_models_raw.get("20d", {}))),
        short_q_models=deps.deserialize_quantile_bundle(stock_quantiles_raw.get("1d")),
        mid_q_models=deps.deserialize_quantile_bundle(stock_quantiles_raw.get("20d")),
    )
    if not stock_states:
        return None, []

    sector_states = deps.build_sector_states(
        stock_states,
        stock_score_fn=deps.stock_policy_score,
    )
    composite_state = deps.compose_state(
        market=market_state,
        sectors=sector_states,
        stocks=stock_states,
        cross_section=cross_section,
    )
    return composite_state, []
