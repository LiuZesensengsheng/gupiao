from __future__ import annotations

from src.application.v2_contracts import CompositeState, ExecutionPlan, PolicyDecision, StockRoleSnapshot, ThemeEpisode


def _price_band_text(*, low: float, high: float) -> str:
    if low == low and high == high:
        return f"{low:.2f} ~ {high:.2f}"
    return "NA"


def _theme_for_symbol(
    *,
    symbol: str,
    sector: str,
    role_state: StockRoleSnapshot | None,
    theme_episodes: list[ThemeEpisode],
) -> ThemeEpisode | None:
    if role_state is not None:
        for item in theme_episodes:
            if str(item.theme) == str(role_state.theme):
                return item
    for item in theme_episodes:
        if str(sector) in {str(value) for value in item.sectors}:
            return item
    return None


def build_execution_plans(
    *,
    state: CompositeState,
    policy_decision: PolicyDecision,
    current_weights: dict[str, float],
    current_holding_days: dict[str, int] | None = None,
    symbol_names: dict[str, str] | None = None,
) -> list[ExecutionPlan]:
    current_holding_days = dict(current_holding_days or {})
    name_map = dict(symbol_names or {})
    stock_map = {str(item.symbol): item for item in getattr(state, "stocks", []) or []}
    role_map = dict(getattr(state, "stock_role_states", {}) or {})
    theme_episodes = list(getattr(state, "theme_episodes", []) or [])
    trim_scores = {
        str(symbol): float(score)
        for symbol, score in getattr(policy_decision, "trim_candidate_scores", {}).items()
    }
    trim_ranks = {
        str(symbol): int(rank)
        for symbol, rank in getattr(policy_decision, "trim_candidate_ranks", {}).items()
    }
    trim_labels = {
        str(symbol): str(label)
        for symbol, label in getattr(policy_decision, "trim_candidate_labels", {}).items()
    }
    all_symbols = sorted(
        {
            symbol
            for symbol, weight in {
                **{str(k): float(v) for k, v in current_weights.items()},
                **{str(k): float(v) for k, v in policy_decision.symbol_target_weights.items()},
            }.items()
            if float(weight) > 1e-9
        }
    )
    plans: list[ExecutionPlan] = []
    for symbol in all_symbols:
        stock = stock_map.get(symbol)
        if stock is None:
            continue
        current_weight = max(0.0, float(current_weights.get(symbol, 0.0)))
        target_weight = max(0.0, float(policy_decision.symbol_target_weights.get(symbol, 0.0)))
        role_state = role_map.get(symbol)
        trim_score = float(trim_scores.get(symbol, 0.0))
        trim_rank = int(trim_ranks.get(symbol, 0))
        trim_label = str(trim_labels.get(symbol, ""))
        theme_episode = _theme_for_symbol(
            symbol=symbol,
            sector=str(getattr(stock, "sector", "")),
            role_state=role_state,
            theme_episodes=theme_episodes,
        )
        one_day = getattr(stock, "horizon_forecasts", {}).get("1d")
        latest_close = float(getattr(stock, "latest_close", float("nan")))
        buy_low = latest_close * 0.985 if latest_close == latest_close else float("nan")
        buy_high = latest_close * 1.015 if latest_close == latest_close else float("nan")
        if one_day is not None and float(getattr(one_day, "price_low", float("nan"))) == float(getattr(one_day, "price_low", float("nan"))):
            buy_low = float(getattr(one_day, "price_low", buy_low))
            buy_high = float(getattr(one_day, "price_high", buy_high))

        if target_weight > current_weight + 0.02:
            bias = "buy_on_pullback"
        elif target_weight < current_weight - 0.02:
            bias = "reduce"
        elif target_weight > 0.0:
            bias = "hold"
        else:
            bias = "avoid"

        avoid_zone = "gap-up chase above forecast range"
        reduce_if = "near-term edge weakens"
        exit_if = "20d probability breaks below 0.48"
        reasons: list[str] = []

        if role_state is not None:
            reasons.append(f"{role_state.role} in {role_state.theme}")
            if role_state.role_downgrade:
                reduce_if = "role downgrade confirms leadership loss"
                reasons.append("role downgrade active")
        if theme_episode is not None:
            reasons.append(f"theme {theme_episode.phase}")
            if theme_episode.phase == "fading":
                reduce_if = "theme stays fading or event risk stays elevated"
                exit_if = "fading theme persists with stock still underperforming"
            elif theme_episode.phase == "crowded" and float(theme_episode.event_risk) >= 0.55:
                reduce_if = "crowded tape keeps event risk above cutoff"
                avoid_zone = "avoid chasing strength into a crowded tape"
            elif theme_episode.phase == "strengthening" and role_state is not None and role_state.role in {"leader", "core"}:
                reduce_if = "only trim if breakout fails with weaker follow-through"
        if trim_rank > 0:
            reasons.append(f"trim rank #{trim_rank} ({trim_label or 'watch'} {trim_score:.2f})")
            if trim_label == "exit_fast":
                reduce_if = f"{reduce_if}; exit overlay ranks this as the highest-priority defensive candidate"
                exit_if = f"{exit_if}; predicted drag stays in exit-fast zone and price follow-through weakens"
            elif trim_label == "reduce":
                reduce_if = f"{reduce_if}; exit overlay keeps this in the reduce-candidate bucket"
            elif trim_label == "watch":
                reduce_if = f"{reduce_if}; watch for weaker follow-through before trimming"
        if float(getattr(stock, "excess_vs_sector_prob", 0.5)) < 0.50:
            reasons.append("relative edge is no longer strong")
        if current_holding_days.get(symbol, 0) > 0:
            reasons.append(f"held {int(current_holding_days.get(symbol, 0))}d")

        plans.append(
            ExecutionPlan(
                symbol=symbol,
                name=str(name_map.get(symbol, symbol)),
                theme="" if role_state is None else str(role_state.theme),
                role="" if role_state is None else str(role_state.role),
                bias=bias,
                buy_zone=_price_band_text(low=buy_low, high=buy_high),
                avoid_zone=avoid_zone,
                reduce_if=reduce_if,
                exit_if=exit_if,
                reason="; ".join(reasons) or "daily execution overlay",
                trim_score=trim_score,
                trim_rank=trim_rank,
                trim_label=trim_label,
            )
        )
    plans.sort(
        key=lambda item: (
            float(policy_decision.symbol_target_weights.get(item.symbol, 0.0)),
            float(current_weights.get(item.symbol, 0.0)),
        ),
        reverse=True,
    )
    return plans
