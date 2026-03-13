from __future__ import annotations

import argparse
import html
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _discover_backtest_summary(manifest_path: Path | None) -> dict:
    if manifest_path is None or not manifest_path.exists():
        return {}
    manifest = _load_json(manifest_path)
    ref = manifest.get("backtest_summary")
    if not ref:
        return {}
    summary_path = (manifest_path.parent / str(ref)).resolve()
    if not summary_path.exists():
        return {}
    return _load_json(summary_path)


def _render_html(
    *,
    title: str,
    as_of_date: str,
    strategy_mode: str,
    risk_regime: str,
    target_exposure: float,
    target_positions: int,
    top20: list[dict[str, str]],
    trades: list[dict[str, str]],
    backtest: dict,
) -> str:
    baseline = backtest.get("baseline", {})
    learned = backtest.get("learned", {})

    def row_cells(row: dict[str, str], keys: list[str]) -> str:
        return "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys)

    top20_rows = "\n".join(
        f"<tr>{row_cells(row, ['rank', 'symbol', 'name', 'sector', 'alpha_score', 'target_weight'])}</tr>"
        for row in top20
    )
    trade_rows = "\n".join(
        f"<tr>{row_cells(row, ['action', 'symbol', 'name', 'sector', 'target_weight'])}</tr>"
        for row in trades
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: rgba(255,255,255,0.82);
      --ink: #182126;
      --muted: #64707a;
      --accent: #0e7490;
      --accent-2: #b45309;
      --line: rgba(24,33,38,0.10);
      --shadow: 0 18px 48px rgba(24,33,38,0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(14,116,144,0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(180,83,9,0.14), transparent 28%),
        linear-gradient(180deg, #fbfaf6 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 20px 72px;
    }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(140deg, rgba(255,255,255,0.92), rgba(255,248,235,0.88));
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 4vw, 52px);
      line-height: 1.02;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .card {{
      padding: 18px;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(6px);
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .section {{
      margin-top: 26px;
      padding: 24px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(14,116,144,0.10);
      color: var(--accent);
      font-weight: 700;
      font-size: 12px;
      margin-right: 8px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .foot {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Quant Output</div>
      <h1>{html.escape(title)}</h1>
      <p class="sub">最新可交易数据日期为 {html.escape(as_of_date)}。页面展示学习策略在 300 股票池上的排序前 20 名，以及当前风险约束下的实际交易计划。</p>
      <div class="grid">
        <div class="card"><div class="label">策略模式</div><div class="value">{html.escape(strategy_mode)}</div></div>
        <div class="card"><div class="label">风险状态</div><div class="value">{html.escape(risk_regime)}</div></div>
        <div class="card"><div class="label">目标仓位</div><div class="value">{_pct(target_exposure)}</div></div>
        <div class="card"><div class="label">目标持仓数</div><div class="value">{target_positions}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>今日实际操作</h2>
      <table>
        <thead>
          <tr><th>动作</th><th>代码</th><th>名称</th><th>行业</th><th>目标权重</th></tr>
        </thead>
        <tbody>
          {trade_rows}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Top 20 推荐</h2>
      <div class="foot">
        <span class="pill">排序输出</span>
        这里显示的是学习策略在 `generated_300` 股票池上的前 20 名，不等于全部都会买入。
      </div>
      <table>
        <thead>
          <tr><th>排名</th><th>代码</th><th>名称</th><th>行业</th><th>Alpha 分数</th><th>当前目标权重</th></tr>
        </thead>
        <tbody>
          {top20_rows}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>回测摘要</h2>
      <div class="two-col">
        <div class="card">
          <div class="label">基线策略</div>
          <div class="foot">总收益 {_pct(float(baseline.get('total_return', 0.0)))} / 年化 {_pct(float(baseline.get('annual_return', 0.0)))} / 超额年化 {_pct(float(baseline.get('excess_annual_return', 0.0)))} / IR {float(baseline.get('information_ratio', 0.0)):.3f} / 回撤 {_pct(float(baseline.get('max_drawdown', 0.0)))}</div>
        </div>
        <div class="card">
          <div class="label">学习型策略</div>
          <div class="foot">总收益 {_pct(float(learned.get('total_return', 0.0)))} / 年化 {_pct(float(learned.get('annual_return', 0.0)))} / 超额年化 {_pct(float(learned.get('excess_annual_return', 0.0)))} / IR {float(learned.get('information_ratio', 0.0)):.3f} / 回撤 {_pct(float(learned.get('max_drawdown', 0.0)))}</div>
        </div>
      </div>
      <div class="foot">本页只包含静态结果，不包含任何交易接口、账号、密钥或写入能力。</div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="量化推荐看板")
    parser.add_argument("--backtest-manifest", default="")
    args = parser.parse_args()

    daily_cache = Path(args.daily_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = _load_pickle(daily_cache)
    state = result.composite_state
    selection = state.candidate_selection
    names = dict(getattr(result, "symbol_names", {}) or {})
    target_weights = dict(getattr(result.policy_decision, "symbol_target_weights", {}) or {})
    sector_by_symbol = {str(stock.symbol): str(stock.sector) for stock in state.stocks}
    alpha_by_symbol = {str(stock.symbol): float(getattr(stock, "alpha_score", 0.0)) for stock in state.stocks}

    top20: list[dict[str, str]] = []
    for idx, symbol in enumerate(selection.shortlisted_symbols[:20], start=1):
        top20.append(
            {
                "rank": str(idx),
                "symbol": symbol,
                "name": str(names.get(symbol, symbol)),
                "sector": str(sector_by_symbol.get(symbol, "")),
                "alpha_score": f"{alpha_by_symbol.get(symbol, 0.0):.4f}",
                "target_weight": _pct(float(target_weights.get(symbol, 0.0))),
            }
        )

    trades: list[dict[str, str]] = []
    for symbol, weight in sorted(target_weights.items(), key=lambda item: float(item[1]), reverse=True):
        if float(weight) <= 1e-9:
            continue
        trades.append(
            {
                "action": "BUY",
                "symbol": str(symbol),
                "name": str(names.get(symbol, symbol)),
                "sector": str(sector_by_symbol.get(symbol, "")),
                "target_weight": _pct(float(weight)),
            }
        )

    manifest_path = Path(str(args.backtest_manifest)).resolve() if args.backtest_manifest else None
    backtest = _discover_backtest_summary(manifest_path)
    html_text = _render_html(
        title=args.title,
        as_of_date=str(state.market.as_of_date),
        strategy_mode=str(state.strategy_mode),
        risk_regime=str(state.risk_regime),
        target_exposure=float(result.policy_decision.target_exposure),
        target_positions=int(result.policy_decision.target_position_count),
        top20=top20,
        trades=trades,
        backtest=backtest,
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")

    metadata = {
        "daily_cache": str(daily_cache),
        "as_of_date": str(state.market.as_of_date),
        "top20_count": len(top20),
        "trade_count": len(trades),
        "backtest_manifest": str(manifest_path) if manifest_path else "",
    }
    (output_dir / "site_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
