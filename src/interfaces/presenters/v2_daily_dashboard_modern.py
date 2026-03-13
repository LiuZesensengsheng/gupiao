from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from src.application.v2_contracts import DailyRunResult as V2DailyRunResult


def _pct(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _num(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def write_v2_daily_dashboard(out_path: str | Path, result: V2DailyRunResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = result.composite_state
    market = state.market
    sentiment = market.sentiment
    facts = market.market_facts
    policy = result.policy_decision
    name_map = dict(result.symbol_names)
    candidate_order = {
        str(symbol): idx
        for idx, symbol in enumerate(getattr(state.candidate_selection, "shortlisted_symbols", []))
    }
    ranked_stocks = sorted(
        list(state.stocks),
        key=lambda stock: candidate_order.get(stock.symbol, len(candidate_order) + 999),
    )
    top20 = ranked_stocks[:20]
    selected_symbols = {
        str(symbol)
        for symbol, weight in policy.symbol_target_weights.items()
        if float(weight) > 0.0
    }
    candidate = state.candidate_selection
    shortlist_text = (
        f"{candidate.shortlist_size}/{candidate.total_scored} shortlisted"
        if candidate.shortlist_size and candidate.total_scored
        else f"{len(state.stocks)} names scored"
    )
    shortlist_notes = " | ".join(candidate.selection_notes[:2]) if candidate.selection_notes else "Full ranking in use."
    recurring_symbols = ", ".join(result.memory_recall.recurring_symbols[:4]) or "NA"
    mainline_rows = "".join(
        "<li>"
        f"{escape(item.name)} | conviction {_pct(item.conviction)} | sectors {escape(', '.join(item.sectors[:2]))}"
        "</li>"
        for item in getattr(state, "mainlines", [])[:4]
    )

    def _stock_name(symbol: str) -> str:
        return str(name_map.get(symbol, symbol))

    def _next_session(date_text: str) -> str:
        ts = pd.Timestamp(date_text)
        if pd.isna(ts):
            return ""
        return str((ts + pd.offsets.BDay(1)).date())

    next_session = _next_session(market.as_of_date)

    def _ticker(symbol: str) -> str:
        return (
            f"<div class='ticker'>{escape(_stock_name(symbol))}</div>"
            f"<div class='ticker-sub'>{escape(symbol)}</div>"
        )

    def _range_text(stock: object, key: str) -> str:
        item = getattr(stock, "horizon_forecasts", {}).get(key)
        if item is None:
            return "NA"
        if item.price_low == item.price_low and item.price_high == item.price_high:
            return f"{item.price_low:.2f} ~ {item.price_high:.2f}"
        return "NA"

    def _stock_reason_text(stock: object) -> str:
        if getattr(stock, "symbol", "") in selected_symbols:
            return getattr(stock, "action_reason", "") or getattr(stock, "weight_reason", "") or "NA"
        return getattr(stock, "blocked_reason", "") or "当前仅保留跟踪"

    market_rows = "".join(
        "<tr>"
        f"<td>{escape(item.label)}</td>"
        f"<td>{_pct(item.up_prob)}</td>"
        f"<td>{_pct(item.q10)} ~ {_pct(item.q90)}</td>"
        f"<td>{_pct(item.q50)}</td>"
        f"<td>{_pct(item.confidence)}</td>"
        "</tr>"
        for key in ["1d", "2d", "3d", "5d", "10d", "20d"]
        for item in [market.horizon_forecasts.get(key)]
        if item is not None
    )

    top20_rows = "".join(
        "<tr>"
        f"<td>{idx}</td>"
        f"<td>{_ticker(stock.symbol)}</td>"
        f"<td>{escape(stock.sector)}</td>"
        f"<td>{_range_text(stock, '1d')}</td>"
        f"<td>{_pct(stock.horizon_forecasts.get('5d').q50 if stock.horizon_forecasts.get('5d') else float('nan'))}</td>"
        f"<td>{_pct(stock.horizon_forecasts.get('20d').q50 if stock.horizon_forecasts.get('20d') else float('nan'))}</td>"
        f"<td>{_pct(stock.horizon_forecasts.get('1d').up_prob if stock.horizon_forecasts.get('1d') else float('nan'))}</td>"
        f"<td>{_pct(stock.horizon_forecasts.get('1d').confidence if stock.horizon_forecasts.get('1d') else float('nan'))}</td>"
        "</tr>"
        for idx, stock in enumerate(top20, start=1)
    )

    action_rows = "".join(
        "<tr>"
        f"<td>{_ticker(action.symbol)}</td>"
        f"<td>{escape(action.action)}</td>"
        f"<td>{_pct(action.current_weight)}</td>"
        f"<td>{_pct(action.target_weight)}</td>"
        f"<td>{_pct(action.delta_weight)}</td>"
        f"<td>{escape(action.note or next((_stock_reason_text(stock) for stock in top20 if stock.symbol == action.symbol), 'NA'))}</td>"
        "</tr>"
        for action in result.trade_actions
    )

    cards_html = "".join(
        "<article class='stock-card'>"
        f"<h3>{escape(_stock_name(stock.symbol))} <span>{escape(stock.symbol)}</span></h3>"
        f"<p class='meta'>{escape(stock.sector)} | 最新收盘 {(_num(stock.latest_close) if stock.latest_close == stock.latest_close else 'NA')}</p>"
        f"<p><strong>下一交易日({escape(next_session or 'NA')})</strong>: {_range_text(stock, '1d')}，上涨概率 {_pct(stock.horizon_forecasts.get('1d').up_prob if stock.horizon_forecasts.get('1d') else float('nan'))}，置信度 {_pct(stock.horizon_forecasts.get('1d').confidence if stock.horizon_forecasts.get('1d') else float('nan'))}</p>"
        f"<p><strong>入池原因</strong>: {escape('；'.join(stock.selection_reasons) or '综合排序靠前')}</p>"
        f"<p><strong>排名原因</strong>: {escape('；'.join(stock.ranking_reasons) or '综合排序稳定')}</p>"
        f"<p><strong>{'操作原因' if stock.symbol in selected_symbols else '未入组合原因'}</strong>: {escape(_stock_reason_text(stock))}</p>"
        f"<p><strong>风险点</strong>: {escape('；'.join(stock.risk_flags) or '暂无显著硬风险')}</p>"
        f"<p><strong>失效条件</strong>: {escape(stock.invalidation_rule or '跌破预期下沿且5日概率转弱')}</p>"
        "</article>"
        for stock in top20[:6]
    )

    review_rows = "".join(
        "<tr>"
        f"<td>{escape(item.label)}</td>"
        f"<td>{_pct(item.hit_rate)}</td>"
        f"<td>{_pct(item.avg_edge)}</td>"
        f"<td>{_pct(item.realized_return)}</td>"
        f"<td>{item.sample_size}</td>"
        f"<td>{escape(item.note or 'NA')}</td>"
        "</tr>"
        for key in ["5d", "20d", "60d"]
        for item in [result.prediction_review.windows.get(key)]
        if item is not None
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>次日决策面板</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1e2730;
      --muted: #6f766d;
      --line: #ddd1bc;
      --shadow: 0 18px 46px rgba(45, 33, 12, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(197, 91, 53, 0.10), transparent 26%),
        radial-gradient(circle at left top, rgba(31, 111, 120, 0.10), transparent 24%),
        linear-gradient(180deg, #faf5ec 0%, var(--bg) 100%);
    }}
    .page {{ max-width: 1380px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); }}
    .hero {{ padding: 24px; }}
    .hero-grid, .triple, .stock-grid {{ display: grid; gap: 16px; }}
    .hero-grid {{ grid-template-columns: 1.2fr 1fr; align-items: start; }}
    .triple {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 18px; }}
    .stock-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }}
    .panel {{ padding: 18px; margin-top: 18px; }}
    h1 {{ margin: 0; font-size: 36px; line-height: 1; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    h3 span {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    p, li {{ line-height: 1.6; }}
    .muted {{ color: var(--muted); }}
    .facts {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .fact {{ padding: 14px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.52); }}
    .fact .eyebrow {{ font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: .08em; }}
    .fact strong {{ display: block; margin-top: 8px; font-size: 28px; }}
    .pill {{ display: inline-flex; padding: 7px 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,.08); background: rgba(255,255,255,.72); margin-right: 8px; font-size: 12px; font-weight: 700; }}
    .list {{ margin: 10px 0 0; padding-left: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e9dece; font-size: 13px; text-align: left; vertical-align: top; }}
    th {{ background: #faf5eb; font-size: 11px; text-transform: uppercase; color: #67706a; letter-spacing: .06em; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.55); }}
    .ticker {{ font-weight: 800; }}
    .ticker-sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .stock-card {{ border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.62); padding: 16px; }}
    .stock-card .meta {{ color: var(--muted); margin-top: 0; }}
    @media (max-width: 1080px) {{
      .hero-grid, .triple, .stock-grid, .facts {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <h1>次日决策面板</h1>
          <p class="muted">数据日期 {escape(market.as_of_date)}，下一交易日 {escape(next_session or 'NA')}。当前为 {escape(state.strategy_mode)} / {escape(state.risk_regime)}。</p>
          <p class="muted">strategy {escape(result.snapshot.strategy_id)} | universe {escape(result.snapshot.universe_id)} | generator {escape(result.generator_version or result.snapshot.generator_version or 'NA')} | source {escape(result.snapshot.us_index_source or 'NA')}</p>
          <p class="muted">{escape(shortlist_text)} | {escape(shortlist_notes)}</p>
          <div>
            <span class="pill">情绪阶段: {escape(sentiment.stage)}</span>
            <span class="pill">情绪分: {_num(sentiment.score, 1)}/100</span>
            <span class="pill">目标仓位: {_pct(policy.target_exposure)}</span>
            <span class="pill">目标持仓: {policy.target_position_count}</span>
          </div>
          <ul class="list">
            {''.join(f"<li>{escape(item)}</li>" for item in sentiment.drivers[:4])}
          </ul>
        </div>
        <div class="facts">
          <div class="fact"><div class="eyebrow">涨跌家数</div><strong>{facts.advancers}/{facts.decliners}</strong><div class="muted">平家数 {facts.flats}</div></div>
          <div class="fact"><div class="eyebrow">涨停 / 跌停</div><strong>{facts.limit_up_count}/{facts.limit_down_count}</strong><div class="muted">新高/新低 {facts.new_high_count}/{facts.new_low_count}</div></div>
          <div class="fact"><div class="eyebrow">样本中位涨跌幅</div><strong>{_pct(facts.median_return)}</strong><div class="muted">样本覆盖 {facts.sample_coverage}</div></div>
        </div>
      </div>
      <div class="triple">
        <div class="panel"><h2>大盘情绪</h2><p>{escape(sentiment.summary)}</p><p class="muted">样本成交额 {_num(facts.sample_amount, 0)}，成交热度 {_pct(state.capital_flow_state.turnover_heat)}，两融变化 {_num(state.capital_flow_state.margin_balance_change, 3)}</p></div>
        <div class="panel"><h2>外部信号</h2><p>资金状态 {escape(state.capital_flow_state.flow_regime)}，北向强度 {_num(state.capital_flow_state.northbound_net_flow, 3)}，大单偏向 {_num(state.capital_flow_state.large_order_bias, 3)}</p><p class="muted">宏观风险 {escape(state.macro_context_state.macro_risk_level)}，风格 {escape(state.macro_context_state.style_regime)}</p></div>
        <div class="panel"><h2>策略记忆</h2><p>{escape('；'.join(result.memory_recall.narrative[:2]) or '暂无策略记忆摘要')}</p><p class="muted">高频标的: {escape(recurring_symbols)}</p></div>
      </div>
    </section>

    <section class="panel">
      <h2>大盘多周期预测</h2>
      <div class="table-wrap"><table><thead><tr><th>周期</th><th>上涨概率</th><th>预期区间</th><th>中位预期</th><th>置信度</th></tr></thead><tbody>{market_rows}</tbody></table></div>
    </section>

    <section class="panel">
      <h2>Dynamic Universe Funnel</h2>
      <p class="muted">coarse {int(result.coarse_pool_size or result.snapshot.coarse_pool_size)} | refined {int(result.refined_pool_size or result.snapshot.refined_pool_size)} | selected {int(result.selected_pool_size or result.snapshot.selected_pool_size)}</p>
      <p class="muted">{escape(result.generator_version or result.snapshot.generator_version or 'NA')} | {escape(shortlist_text)}</p>
      <p class="muted">{escape(shortlist_notes)}</p>
    </section>

    <section class="panel">
      <h2>Top20 推荐</h2>
      <div class="table-wrap"><table><thead><tr><th>排名</th><th>股票</th><th>行业</th><th>下一交易日区间</th><th>5日中位预期</th><th>20日中位预期</th><th>1日上涨概率</th><th>置信度</th></tr></thead><tbody>{top20_rows or "<tr><td colspan='8'>暂无候选</td></tr>"}</tbody></table></div>
    </section>

    <section class="panel">
      <h2>实际操作</h2>
      <div class="table-wrap"><table><thead><tr><th>股票</th><th>动作</th><th>当前权重</th><th>目标权重</th><th>权重变化</th><th>操作理由</th></tr></thead><tbody>{action_rows or "<tr><td colspan='6'>当前不触发调仓</td></tr>"}</tbody></table></div>
    </section>

    <section class="stock-grid">
      {cards_html or "<div class='panel'>暂无解释卡</div>"}
    </section>

    <section class="panel">
      <h2>Mainline Radar</h2>
      <ul class="list">{mainline_rows or "<li>暂无主线</li>"}</ul>
    </section>

    <section class="panel">
      <h2>预测复盘</h2>
      <div class="table-wrap"><table><thead><tr><th>窗口</th><th>命中参考</th><th>平均边际</th><th>近窗表现</th><th>样本数</th><th>说明</th></tr></thead><tbody>{review_rows or "<tr><td colspan='6'>暂无复盘数据</td></tr>"}</tbody></table></div>
      <p class="muted">{escape('；'.join(result.prediction_review.notes) or '')}</p>
    </section>
  </main>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")
    return path
