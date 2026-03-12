from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from src.application.v2_contracts import DailyRunResult as V2DailyRunResult


def _pct(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _bp(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 10000:.0f}bp"


def _num(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def _score(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:+.3f}"


def _score_color(v: float) -> str:
    if pd.isna(v):
        return "#7f8c8d"
    if v >= 0:
        return "#d4523d"
    return "#2e69c7"


def write_v2_daily_dashboard(out_path: str | Path, result: V2DailyRunResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name_map = dict(result.symbol_names)
    state = result.composite_state
    policy = result.policy_decision
    market = state.market
    market_info = state.market_info_state
    capital = state.capital_flow_state
    macro = state.macro_context_state

    def _stock_name(symbol: str) -> str:
        return str(name_map.get(symbol, symbol))

    def _stock_label(symbol: str) -> str:
        name = _stock_name(symbol)
        if name and name != symbol:
            return (
                f"<div class='ticker'>{escape(symbol)}</div>"
                f"<div class='ticker-name'>{escape(name)}</div>"
            )
        return f"<div class='ticker'>{escape(symbol)}</div>"

    def _badge(text: str, tone: str = "neutral") -> str:
        return f"<span class='badge {tone}'>{escape(text)}</span>"

    def _action_tone(action: str) -> str:
        return "buy" if action == "BUY" else ("sell" if action == "SELL" else "hold")

    def _action_label(action: str) -> str:
        return {"BUY": "Buy", "SELL": "Sell", "HOLD": "Hold"}.get(action, action)

    def _meter_row(label: str, value: float, *, color: str, detail: str = "") -> str:
        width = max(0.0, min(100.0, float(value) * 100.0))
        detail_html = f"<span>{escape(detail)}</span>" if detail else ""
        return (
            "<div class='meter-row'>"
            f"<div class='meter-head'><strong>{escape(label)}</strong>{detail_html}</div>"
            "<div class='meter-track'>"
            f"<div class='meter-fill' style='width:{width:.1f}%; background:{color};'></div>"
            "</div>"
            f"<div class='meter-value'>{_pct(value)}</div>"
            "</div>"
        )

    selected_symbols = {
        str(symbol)
        for symbol, weight in policy.symbol_target_weights.items()
        if float(weight) > 0.0
    }
    selected_stocks = [stock for stock in state.stocks if stock.symbol in selected_symbols]
    watchlist_stocks = list(state.stocks[:12])
    risk_notes = list(policy.risk_notes) + list(policy.execution_notes)
    top_actions = list(result.trade_actions[:6])

    buy_count = sum(1 for item in result.trade_actions if item.action == "BUY")
    sell_count = sum(1 for item in result.trade_actions if item.action == "SELL")
    action_summary_parts = []
    if buy_count:
        action_summary_parts.append(f"{buy_count} buy")
    if sell_count:
        action_summary_parts.append(f"{sell_count} sell")
    if not action_summary_parts:
        action_summary_parts.append("no changes")
    action_summary = ", ".join(action_summary_parts)
    header_note = (
        f"{policy.target_position_count} slots, turnover cap {_pct(policy.turnover_cap)}, "
        f"{'rebalance' if policy.rebalance_now else 'hold'}"
    )

    action_rows_html = "".join(
        "<tr>"
        f"<td>{_stock_label(item.symbol)}</td>"
        f"<td>{_badge(_action_label(item.action), _action_tone(item.action))}</td>"
        f"<td>{_pct(item.current_weight)}</td>"
        f"<td>{_pct(item.target_weight)}</td>"
        f"<td class='score' style='color:{_score_color(item.delta_weight)}'>{_pct(item.delta_weight)}</td>"
        f"<td>{escape(item.note or 'NA')}</td>"
        "</tr>"
        for item in top_actions
    )

    selected_rows_html = "".join(
        "<tr>"
        f"<td>{_stock_label(stock.symbol)}</td>"
        f"<td>{_pct(policy.symbol_target_weights.get(stock.symbol, 0.0))}</td>"
        f"<td>{_pct(stock.up_5d_prob)}</td>"
        f"<td>{_pct(stock.up_20d_prob)}</td>"
        f"<td class='score' style='color:{_score_color(stock.alpha_score - 0.5)}'>{_num(stock.alpha_score, 3)}</td>"
        f"<td>{_pct(stock.tradeability_score)}</td>"
        "</tr>"
        for stock in selected_stocks
    )

    watch_rows_html = "".join(
        "<tr>"
        f"<td>{idx}</td>"
        f"<td>{_stock_label(stock.symbol)}</td>"
        f"<td>{_pct(stock.up_1d_prob)}</td>"
        f"<td>{_pct(stock.up_5d_prob)}</td>"
        f"<td>{_pct(stock.up_20d_prob)}</td>"
        f"<td class='score' style='color:{_score_color(stock.alpha_score - 0.5)}'>{_num(stock.alpha_score, 3)}</td>"
        f"<td>{_pct(stock.tradeability_score)}</td>"
        f"<td>{_pct(policy.symbol_target_weights.get(stock.symbol, 0.0))}</td>"
        "</tr>"
        for idx, stock in enumerate(watchlist_stocks, start=1)
    )

    full_rows_html = "".join(
        "<tr>"
        f"<td>{idx}</td>"
        f"<td>{_stock_label(stock.symbol)}</td>"
        f"<td>{_pct(stock.up_1d_prob)}</td>"
        f"<td>{_pct(stock.up_2d_prob)}</td>"
        f"<td>{_pct(stock.up_3d_prob)}</td>"
        f"<td>{_pct(stock.up_5d_prob)}</td>"
        f"<td>{_pct(stock.up_20d_prob)}</td>"
        f"<td class='score' style='color:{_score_color(stock.alpha_score - 0.5)}'>{_num(stock.alpha_score, 3)}</td>"
        f"<td>{_pct(stock.excess_vs_sector_prob)}</td>"
        f"<td>{_pct(stock.tradeability_score)}</td>"
        f"<td>{_pct(policy.symbol_target_weights.get(stock.symbol, 0.0))}</td>"
        "</tr>"
        for idx, stock in enumerate(state.stocks, start=1)
    )

    sector_budget_rows = "".join(
        "<div class='budget-row'>"
        f"<span>{escape(sector)}</span>"
        "<div class='budget-track'>"
        f"<div class='budget-fill' style='width:{max(0.0, min(100.0, float(weight) * 100.0)):.1f}%;'></div>"
        "</div>"
        f"<strong>{_pct(weight)}</strong>"
        "</div>"
        for sector, weight in sorted(policy.sector_budgets.items(), key=lambda item: float(item[1]), reverse=True)
        if float(weight) > 0.0
    )

    risk_notes_html = "".join(f"<li>{escape(note)}</li>" for note in risk_notes[:6]) or "<li>No explicit risk note.</li>"
    memory_notes_html = "".join(f"<li>{escape(note)}</li>" for note in result.memory_recall.narrative[:5]) or "<li>No memory narrative yet.</li>"
    recurring_symbols = ", ".join(str(symbol) for symbol in result.memory_recall.recurring_symbols[:6]) or "NA"

    negative_rows_html = "".join(
        "<tr>"
        f"<td>{escape(item.target or 'market')}</td>"
        f"<td>{escape(item.event_tag or item.info_type)}</td>"
        f"<td>{_score(item.score)}</td>"
        "</tr>"
        for item in result.top_negative_info_events[:8]
    )
    positive_rows_html = "".join(
        "<tr>"
        f"<td>{escape(item.target or 'market')}</td>"
        f"<td>{escape(item.event_tag or item.info_type)}</td>"
        f"<td>{_score(item.score)}</td>"
        "</tr>"
        for item in result.top_positive_info_signals[:8]
    )
    divergence_rows_html = "".join(
        "<tr>"
        f"<td>{escape(item.symbol)}</td>"
        f"<td>{_pct(item.quant_prob_20d)}</td>"
        f"<td>{_pct(item.info_prob_20d)}</td>"
        f"<td>{_pct(item.shadow_prob_20d)}</td>"
        f"<td>{_bp(item.gap)}</td>"
        "</tr>"
        for item in result.quant_info_divergence[:8]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 Daily Dashboard</title>
  <style>
    :root {{
      --bg: #efe7d6;
      --panel: #fffaf1;
      --panel-strong: #fffdf8;
      --ink: #1f2430;
      --muted: #6f746d;
      --line: #dccfb8;
      --line-strong: #c9b995;
      --accent: #b85632;
      --buy: #c23b32;
      --sell: #2f67c8;
      --hold: #75836f;
      --shadow: 0 18px 48px rgba(78, 58, 27, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(184, 86, 50, 0.12), transparent 28%),
        radial-gradient(circle at 20% 15%, rgba(31, 110, 120, 0.10), transparent 24%),
        linear-gradient(180deg, #f7f1e5 0%, var(--bg) 100%);
    }}
    .page {{ max-width: 1360px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      background: linear-gradient(135deg, rgba(255, 252, 246, 0.96), rgba(248, 239, 224, 0.96));
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 26px;
      box-shadow: var(--shadow);
    }}
    .hero-top {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; }}
    h1 {{ margin: 0; font-size: 38px; line-height: 1; letter-spacing: -0.03em; font-weight: 800; }}
    .hero-sub {{ margin-top: 10px; color: var(--muted); font-size: 14px; line-height: 1.6; }}
    .hero-flags {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .badge {{
      display: inline-flex; align-items: center; justify-content: center; padding: 7px 12px; border-radius: 999px;
      border: 1px solid transparent; font-size: 12px; font-weight: 700; letter-spacing: 0.02em; white-space: nowrap;
    }}
    .badge.neutral {{ background: rgba(31, 36, 48, 0.06); color: var(--ink); border-color: rgba(31, 36, 48, 0.08); }}
    .badge.buy {{ background: rgba(194, 59, 50, 0.10); color: var(--buy); border-color: rgba(194, 59, 50, 0.18); }}
    .badge.sell {{ background: rgba(47, 103, 200, 0.10); color: var(--sell); border-color: rgba(47, 103, 200, 0.18); }}
    .badge.hold {{ background: rgba(117, 131, 111, 0.12); color: var(--hold); border-color: rgba(117, 131, 111, 0.18); }}
    .hero-strip {{ margin-top: 22px; display: grid; gap: 14px; grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .hero-card {{ background: rgba(255, 255, 255, 0.62); border: 1px solid rgba(201, 185, 149, 0.55); border-radius: 20px; padding: 16px; }}
    .eyebrow {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.10em; }}
    .metric {{ margin-top: 8px; font-size: 32px; line-height: 1; font-weight: 800; }}
    .metric-sub {{ margin-top: 8px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .section {{ margin-top: 18px; }}
    .section-grid {{ display: grid; gap: 16px; grid-template-columns: 1.3fr 1fr; }}
    .triple-grid {{ display: grid; gap: 16px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 18px; box-shadow: 0 12px 28px rgba(45, 31, 8, 0.05); }}
    .panel-title {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    .panel-title h2 {{ margin: 0; font-size: 18px; letter-spacing: -0.01em; }}
    .panel-title p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .callout {{ border: 1px solid var(--line-strong); background: linear-gradient(180deg, rgba(255,255,255,0.58), rgba(255,255,255,0.18)); border-radius: 18px; padding: 14px; margin-top: 12px; }}
    .callout strong {{ font-size: 14px; display: block; margin-bottom: 6px; }}
    .meter-stack {{ display: grid; gap: 12px; }}
    .meter-row {{ display: grid; gap: 6px; }}
    .meter-head {{ display: flex; justify-content: space-between; gap: 10px; font-size: 13px; color: var(--muted); }}
    .meter-head strong {{ color: var(--ink); font-weight: 700; }}
    .meter-track {{ height: 12px; border-radius: 999px; background: #ecdfca; overflow: hidden; }}
    .meter-fill {{ height: 100%; border-radius: 999px; }}
    .meter-value {{ font-size: 12px; color: var(--muted); text-align: right; }}
    .budget-stack {{ display: grid; gap: 10px; margin-top: 8px; }}
    .budget-row {{ display: grid; grid-template-columns: 100px 1fr 56px; gap: 10px; align-items: center; font-size: 13px; }}
    .budget-track {{ height: 11px; border-radius: 999px; background: #eadfce; overflow: hidden; }}
    .budget-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), #d98b4f); }}
    .list {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    .list li + li {{ margin-top: 8px; }}
    .table-wrap {{ border: 1px solid var(--line); border-radius: 20px; overflow: auto; background: var(--panel-strong); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; }}
    th, td {{ padding: 11px 10px; border-bottom: 1px solid #eadfce; font-size: 13px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; z-index: 1; background: #fbf5ea; color: #5c615b; text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }}
    tr:last-child td {{ border-bottom: none; }}
    .ticker {{ font-size: 13px; font-weight: 800; letter-spacing: 0.02em; }}
    .ticker-name {{ margin-top: 3px; color: var(--muted); font-size: 12px; }}
    .score {{ font-weight: 800; }}
    .mini-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .mini-stat {{ border: 1px solid var(--line); border-radius: 16px; padding: 12px; background: rgba(255,255,255,0.55); }}
    .mini-stat strong {{ display: block; font-size: 20px; margin-top: 4px; }}
    details {{ margin-top: 16px; border: 1px solid var(--line); border-radius: 20px; background: rgba(255,255,255,0.44); overflow: hidden; }}
    summary {{ cursor: pointer; list-style: none; padding: 16px 18px; font-weight: 800; letter-spacing: 0.01em; }}
    summary::-webkit-details-marker {{ display: none; }}
    .details-body {{ padding: 0 18px 18px; }}
    .empty {{ border: 1px dashed var(--line); border-radius: 16px; padding: 18px; color: var(--muted); text-align: center; background: rgba(255,255,255,0.5); }}
    .footer-note {{ margin-top: 16px; font-size: 12px; color: var(--muted); line-height: 1.6; }}
    @media (max-width: 1080px) {{
      .hero-strip, .section-grid, .triple-grid, .mini-grid {{ grid-template-columns: 1fr; }}
      .hero-top {{ flex-direction: column; }}
      .hero-flags {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1>V2 Daily Dashboard</h1>
          <div class="hero-sub">
            <div>Strategy {escape(result.snapshot.strategy_id)} | As of {escape(market.as_of_date)} | Universe {escape(result.snapshot.universe_id or "custom")}</div>
            <div>Focus: {escape(action_summary)} | {escape(header_note)}</div>
            <div>US context {"on" if result.snapshot.use_us_index_context else "off"} | source {escape(result.snapshot.us_index_source or "NA")}</div>
          </div>
        </div>
        <div class="hero-flags">
          {_badge(state.strategy_mode, "neutral")}
          {_badge(state.risk_regime, "neutral")}
          {_badge("rebalance" if policy.rebalance_now else "hold", "buy" if policy.rebalance_now else "hold")}
          {_badge(f"{len(state.stocks)} names scored", "neutral")}
        </div>
      </div>
      <div class="hero-strip">
        <article class="hero-card"><div class="eyebrow">Target Exposure</div><div class="metric">{_pct(policy.target_exposure)}</div><div class="metric-sub">Target positions {policy.target_position_count} | turnover cap {_pct(policy.turnover_cap)}</div></article>
        <article class="hero-card"><div class="eyebrow">Market Stack</div><div class="metric">{_pct(market.up_20d_prob)}</div><div class="metric-sub">1d {_pct(market.up_1d_prob)} | 5d {_pct(market.up_5d_prob)} | 20d {_pct(market.up_20d_prob)}</div></article>
        <article class="hero-card"><div class="eyebrow">Execution State</div><div class="metric">{buy_count + sell_count}</div><div class="metric-sub">Buy {buy_count} | Sell {sell_count} | intraday T {"on" if policy.intraday_t_allowed else "off"}</div></article>
        <article class="hero-card"><div class="eyebrow">External Layer</div><div class="metric">{escape(capital.flow_regime)}</div><div class="metric-sub">Macro {escape(macro.macro_risk_level)} | info items {result.info_item_count}</div></article>
      </div>
    </section>

    <section class="section section-grid">
      <article class="panel">
        <div class="panel-title"><div><h2>Execution Spotlight</h2><p>Only the actionable plan and the highest-signal reasons stay above the fold.</p></div>{_badge(f"{len(top_actions)} shown", "neutral")}</div>
        {"<div class='table-wrap'><table><thead><tr><th>Name</th><th>Action</th><th>Current</th><th>Target</th><th>Delta</th><th>Note</th></tr></thead><tbody>" + action_rows_html + "</tbody></table></div>" if action_rows_html else "<div class='empty'>No trade action for this run.</div>"}
        <div class="callout"><strong>Decision Drivers</strong><ul class="list">{risk_notes_html}</ul></div>
      </article>
      <article class="panel">
        <div class="panel-title"><div><h2>Market Pulse</h2><p>Compact gauges for tomorrow's decision context.</p></div></div>
        <div class="meter-stack">
          {_meter_row("1d up probability", market.up_1d_prob, color="linear-gradient(90deg, #d87345, #b85632)", detail=market.trend_state)}
          {_meter_row("5d up probability", market.up_5d_prob, color="linear-gradient(90deg, #1f8e86, #1f6e78)", detail=state.strategy_mode)}
          {_meter_row("20d up probability", market.up_20d_prob, color="linear-gradient(90deg, #4d6fd6, #244a8f)", detail=state.risk_regime)}
          {_meter_row("drawdown guard", 1.0 - market.drawdown_risk, color="linear-gradient(90deg, #c7a26b, #9a6b34)", detail=f"risk {_pct(market.drawdown_risk)}")}
          {_meter_row("liquidity health", 1.0 - market.liquidity_stress, color="linear-gradient(90deg, #5f8d68, #3d6d45)", detail=f"stress {_pct(market.liquidity_stress)}")}
        </div>
      </article>
    </section>

    <section class="section section-grid">
      <article class="panel">
        <div class="panel-title"><div><h2>Selected Positions</h2><p>What the portfolio actually wants to hold after all risk clamps.</p></div>{_badge(f"{len(selected_stocks)} selected", "neutral")}</div>
        {"<div class='table-wrap'><table><thead><tr><th>Name</th><th>Target</th><th>5d</th><th>20d</th><th>Alpha</th><th>Tradeability</th></tr></thead><tbody>" + selected_rows_html + "</tbody></table></div>" if selected_rows_html else "<div class='empty'>No position selected.</div>"}
        <div class="callout"><strong>Sector Budget</strong><div class="budget-stack">{sector_budget_rows or "<div class='empty'>No sector budget assigned.</div>"}</div></div>
      </article>
      <article class="panel">
        <div class="panel-title"><div><h2>Watchlist Edge</h2><p>Top-ranked candidates only. The full 300-name table is collapsed below.</p></div>{_badge(f"top {len(watchlist_stocks)}", "neutral")}</div>
        <div class="table-wrap"><table><thead><tr><th>Rank</th><th>Name</th><th>1d</th><th>5d</th><th>20d</th><th>Alpha</th><th>Tradeability</th><th>Target</th></tr></thead><tbody>{watch_rows_html or "<tr><td colspan='8'>No candidates.</td></tr>"}</tbody></table></div>
      </article>
    </section>

    <section class="section triple-grid">
      <article class="panel">
        <div class="panel-title"><div><h2>External Signals</h2><p>Capital flow and macro context in one glance.</p></div>{_badge("enabled" if result.external_signal_enabled else "disabled", "neutral")}</div>
        <div class="mini-grid">
          <div class="mini-stat"><div class="eyebrow">Flow Regime</div><strong>{escape(capital.flow_regime)}</strong><div class="metric-sub">turnover heat {_pct(capital.turnover_heat)}</div></div>
          <div class="mini-stat"><div class="eyebrow">Macro Risk</div><strong>{escape(macro.macro_risk_level)}</strong><div class="metric-sub">style {escape(macro.style_regime)}</div></div>
          <div class="mini-stat"><div class="eyebrow">Northbound</div><strong>{_num(capital.northbound_net_flow, 3)}</strong><div class="metric-sub">large order {_num(capital.large_order_bias, 3)}</div></div>
          <div class="mini-stat"><div class="eyebrow">FX / Commodity</div><strong>{_pct(macro.fx_pressure)}</strong><div class="metric-sub">commodity {_pct(macro.commodity_pressure)}</div></div>
        </div>
      </article>
      <article class="panel">
        <div class="panel-title"><div><h2>Info Overlay</h2><p>This section should only matter when the overlay is non-empty.</p></div>{_badge("on" if result.info_shadow_enabled else "off", "neutral")}</div>
        <div class="mini-grid">
          <div class="mini-stat"><div class="eyebrow">Items</div><strong>{result.info_item_count}</strong><div class="metric-sub">coverage {_pct(market_info.coverage_ratio)}</div></div>
          <div class="mini-stat"><div class="eyebrow">Event Risk</div><strong>{_pct(market_info.event_risk_level)}</strong><div class="metric-sub">negative {_pct(market_info.negative_event_risk)}</div></div>
          <div class="mini-stat"><div class="eyebrow">Catalyst</div><strong>{_pct(market_info.catalyst_strength)}</strong><div class="metric-sub">confidence {_pct(market_info.coverage_confidence)}</div></div>
          <div class="mini-stat"><div class="eyebrow">Shadow 20d</div><strong>{_pct(market_info.shadow_prob_20d)}</strong><div class="metric-sub">market 20d {_pct(market_info.info_prob_20d)}</div></div>
        </div>
      </article>
      <article class="panel">
        <div class="panel-title"><div><h2>Strategy Memory</h2><p>Recent recurring symbols, risk tags, and narrative.</p></div>{_badge(f"{result.memory_recall.recent_daily_run_count} runs", "neutral")}</div>
        <div class="mini-grid">
          <div class="mini-stat"><div class="eyebrow">Avg Exposure</div><strong>{_pct(result.memory_recall.average_target_exposure)}</strong><div class="metric-sub">rebalance ratio {_pct(result.memory_recall.rebalance_ratio)}</div></div>
          <div class="mini-stat"><div class="eyebrow">Research IR</div><strong>{_num(result.memory_recall.latest_research_information_ratio, 2)}</strong><div class="metric-sub">excess {_pct(result.memory_recall.latest_research_excess_annual_return)}</div></div>
        </div>
        <div class="callout"><strong>Recurring symbols</strong>{escape(recurring_symbols)}</div>
        <ul class="list" style="margin-top:12px;">{memory_notes_html}</ul>
      </article>
    </section>

    <section class="section section-grid">
      <article class="panel">
        <div class="panel-title"><div><h2>Negative Event Watch</h2><p>Only top negatives are worth surfacing here.</p></div></div>
        {"<div class='table-wrap'><table><thead><tr><th>Target</th><th>Tag</th><th>Score</th></tr></thead><tbody>" + negative_rows_html + "</tbody></table></div>" if negative_rows_html else "<div class='empty'>No negative event surfaced.</div>"}
      </article>
      <article class="panel">
        <div class="panel-title"><div><h2>Positive Signal Watch</h2><p>Positive signals should support selection, not dominate it.</p></div></div>
        {"<div class='table-wrap'><table><thead><tr><th>Target</th><th>Tag</th><th>Score</th></tr></thead><tbody>" + positive_rows_html + "</tbody></table></div>" if positive_rows_html else "<div class='empty'>No positive signal surfaced.</div>"}
      </article>
    </section>

    <section class="section">
      <article class="panel">
        <div class="panel-title"><div><h2>Quant vs Info Divergence</h2><p>Only the largest disagreements are shown.</p></div></div>
        {"<div class='table-wrap'><table><thead><tr><th>Symbol</th><th>Quant 20d</th><th>Info 20d</th><th>Shadow 20d</th><th>Gap</th></tr></thead><tbody>" + divergence_rows_html + "</tbody></table></div>" if divergence_rows_html else "<div class='empty'>No material quant/info divergence.</div>"}
      </article>
    </section>

    <section class="section">
      <details>
        <summary>Open full ranked universe ({len(state.stocks)} names)</summary>
        <div class="details-body">
          <div class="table-wrap">
            <table>
              <thead><tr><th>Rank</th><th>Name</th><th>1d</th><th>2d</th><th>3d</th><th>5d</th><th>20d</th><th>Alpha</th><th>Excess vs sector</th><th>Tradeability</th><th>Target</th></tr></thead>
              <tbody>{full_rows_html}</tbody>
            </table>
          </div>
        </div>
      </details>
      <div class="footer-note">This dashboard is designed as a decision page first and an audit page second. The actionable plan stays above the fold; the full universe remains available below for review.</div>
    </section>
  </main>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")
    return path
