from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_dynamic300_rule_plan import (  # noqa: E402
    _build_signal_schedule,
    _build_snapshots,
    _load_close_frame,
    _load_settings,
    _simulate_fast_backtest,
)


@dataclass(frozen=True)
class SweepSpec:
    lookback_trading_days: int
    rebalance_interval: int
    top_n: int
    max_per_theme: int


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _score_row(row: dict[str, Any]) -> float:
    summary = dict(row.get("summary", {}))
    return float(summary.get("excess_annual_return", 0.0)) + 0.15 * float(summary.get("information_ratio", 0.0)) + 0.35 * float(summary.get("max_drawdown", 0.0))


def _evaluate_spec(
    *,
    settings: dict[str, Any],
    config_path: str,
    end_date: str,
    benchmark_symbol: str,
    commission_bps: float,
    slippage_bps: float,
    workers: int,
    spec: SweepSpec,
) -> dict[str, Any]:
    benchmark_frame = _load_close_frame(settings["data_dir"], benchmark_symbol)
    schedule = _build_signal_schedule(
        benchmark_frame=benchmark_frame,
        end_date=end_date,
        lookback_trading_days=spec.lookback_trading_days,
        rebalance_interval=spec.rebalance_interval,
    )
    snapshots = _build_snapshots(
        settings=settings,
        schedule=schedule,
        top_n=spec.top_n,
        max_per_theme=spec.max_per_theme,
        workers=workers,
    )
    result = _simulate_fast_backtest(
        settings=settings,
        benchmark_symbol=benchmark_symbol,
        end_date=end_date,
        snapshots=snapshots,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    summary = dict(result.get("summary", {}))
    first_holdings = []
    if snapshots:
        first_holdings = [
            f"{item.name}({item.symbol})"
            for item in snapshots[-1].holdings
        ]
    row = {
        "params": {
            "config_path": config_path,
            "end_date": end_date,
            "benchmark_symbol": benchmark_symbol,
            "commission_bps": float(commission_bps),
            "slippage_bps": float(slippage_bps),
            "lookback_trading_days": int(spec.lookback_trading_days),
            "rebalance_interval": int(spec.rebalance_interval),
            "top_n": int(spec.top_n),
            "max_per_theme": int(spec.max_per_theme),
            "workers": int(workers),
        },
        "summary": summary,
        "snapshot_count": len(snapshots),
        "latest_holdings": first_holdings,
    }
    row["robust_score"] = _score_row(row)
    return row


def _markdown_report(
    *,
    end_date: str,
    benchmark_symbol: str,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Dynamic 300 轻量规则快回测参数扫描",
        "",
        f"- 截止日期: {end_date}",
        f"- 基准: {benchmark_symbol}",
        f"- 组合数: {len(rows)}",
        "",
        "## 排名",
        "",
        "| 排名 | lookback | rebalance | top_n | max_theme | 总收益 | 超额总收益 | 年化超额 | 回撤 | IR | robust_score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        params = dict(row.get("params", {}))
        summary = dict(row.get("summary", {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(params.get("lookback_trading_days", "")),
                    str(params.get("rebalance_interval", "")),
                    str(params.get("top_n", "")),
                    str(params.get("max_per_theme", "")),
                    f"{float(summary.get('total_return', 0.0)):.2%}",
                    f"{float(summary.get('excess_total_return', 0.0)):.2%}",
                    f"{float(summary.get('excess_annual_return', 0.0)):.2%}",
                    f"{float(summary.get('max_drawdown', 0.0)):.2%}",
                    f"{float(summary.get('information_ratio', 0.0)):.3f}",
                    f"{float(row.get('robust_score', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top 3 组合",
            "",
        ]
    )
    for index, row in enumerate(rows[:3], start=1):
        params = dict(row.get("params", {}))
        summary = dict(row.get("summary", {}))
        latest_holdings = ", ".join(row.get("latest_holdings", []))
        lines.extend(
            [
                f"### {index}. lookback={params.get('lookback_trading_days')} | rebalance={params.get('rebalance_interval')} | top_n={params.get('top_n')} | max_theme={params.get('max_per_theme')}",
                f"- 超额总收益: {float(summary.get('excess_total_return', 0.0)):.2%}",
                f"- 年化超额: {float(summary.get('excess_annual_return', 0.0)):.2%}",
                f"- 最大回撤: {float(summary.get('max_drawdown', 0.0)):.2%}",
                f"- 信息比率: {float(summary.get('information_ratio', 0.0)):.3f}",
                f"- 最新一期持仓: {latest_holdings or '无'}",
                "",
            ]
        )
    return "\n".join(lines)


def _html_report(
    *,
    end_date: str,
    benchmark_symbol: str,
    rows: list[dict[str, Any]],
) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    table_rows = []
    for index, row in enumerate(rows, start=1):
        params = dict(row.get("params", {}))
        summary = dict(row.get("summary", {}))
        table_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{params.get('lookback_trading_days')}</td>"
            f"<td>{params.get('rebalance_interval')}</td>"
            f"<td>{params.get('top_n')}</td>"
            f"<td>{params.get('max_per_theme')}</td>"
            f"<td>{float(summary.get('total_return', 0.0)):.2%}</td>"
            f"<td>{float(summary.get('excess_total_return', 0.0)):.2%}</td>"
            f"<td>{float(summary.get('excess_annual_return', 0.0)):.2%}</td>"
            f"<td>{float(summary.get('max_drawdown', 0.0)):.2%}</td>"
            f"<td>{float(summary.get('information_ratio', 0.0)):.3f}</td>"
            f"<td>{float(row.get('robust_score', 0.0)):.3f}</td>"
            "</tr>"
        )

    cards = []
    for index, row in enumerate(rows[:3], start=1):
        params = dict(row.get("params", {}))
        summary = dict(row.get("summary", {}))
        holdings = "".join(f"<li>{esc(item)}</li>" for item in row.get("latest_holdings", []))
        cards.append(
            "<article class='card'>"
            f"<div class='rank'>#{index}</div>"
            f"<h3>lookback={params.get('lookback_trading_days')} / rebalance={params.get('rebalance_interval')}</h3>"
            f"<p>top_n={params.get('top_n')} | max_theme={params.get('max_per_theme')}</p>"
            f"<p><strong>超额总收益</strong>: {float(summary.get('excess_total_return', 0.0)):.2%}</p>"
            f"<p><strong>年化超额</strong>: {float(summary.get('excess_annual_return', 0.0)):.2%}</p>"
            f"<p><strong>最大回撤</strong>: {float(summary.get('max_drawdown', 0.0)):.2%}</p>"
            f"<p><strong>IR</strong>: {float(summary.get('information_ratio', 0.0)):.3f}</p>"
            f"<ul>{holdings}</ul>"
            "</article>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dynamic 300 参数扫描</title>
  <style>
    :root {{
      --bg: #f6f0e7;
      --panel: rgba(255,255,255,0.86);
      --ink: #152126;
      --muted: #5d6b6b;
      --line: #ddd1c1;
      --accent: #8f4a1d;
      --accent-soft: #f6ddc8;
      --shadow: 0 18px 42px rgba(36, 27, 16, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(143,74,29,0.10), transparent 22%),
        radial-gradient(circle at left top, rgba(24,117,104,0.08), transparent 26%),
        linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
    }}
    .page {{ max-width: 1320px; margin: 0 auto; padding: 24px 18px 48px; }}
    .hero, .panel, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero, .panel, .card {{ padding: 20px; }}
    .hero {{ margin-bottom: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 18px; }}
    .rank {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 800;
      font-size: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #eadfce; text-align: left; font-size: 13px; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; background: rgba(249,242,231,0.9); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.5); }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; }}
    h3 {{ margin-top: 0; }}
    p, li {{ line-height: 1.6; }}
    @media (max-width: 1080px) {{
      .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Dynamic 300 轻量规则快回测参数扫描</h1>
      <p>截止日期 {esc(end_date)}，基准 {esc(benchmark_symbol)}。这份扫描用来判断轻量规则版是否具备参数稳健性，而不是只在某一组参数上碰巧跑得好。</p>
    </section>
    <section class="cards">
      {''.join(cards)}
    </section>
    <section class="panel">
      <h2>全部组合</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>排名</th>
              <th>lookback</th>
              <th>rebalance</th>
              <th>top_n</th>
              <th>max_theme</th>
              <th>总收益</th>
              <th>超额总收益</th>
              <th>年化超额</th>
              <th>回撤</th>
              <th>IR</th>
              <th>robust_score</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>"""


def run_sweep(
    *,
    config_path: str,
    end_date: str,
    lookbacks: list[int],
    rebalance_intervals: list[int],
    top_ns: list[int],
    max_per_themes: list[int],
    benchmark_symbol: str,
    commission_bps: float,
    slippage_bps: float,
    workers: int,
) -> dict[str, str]:
    settings = _load_settings(config_path)
    specs = [
        SweepSpec(
            lookback_trading_days=int(lookback),
            rebalance_interval=int(rebalance),
            top_n=int(top_n),
            max_per_theme=int(max_theme),
        )
        for lookback, rebalance, top_n, max_theme in product(lookbacks, rebalance_intervals, top_ns, max_per_themes)
    ]
    rows: list[dict[str, Any]] = []
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        print(
            "[sweep] "
            f"{index}/{total}: lookback={spec.lookback_trading_days}, "
            f"rebalance={spec.rebalance_interval}, top_n={spec.top_n}, "
            f"max_theme={spec.max_per_theme}"
        )
        rows.append(
            _evaluate_spec(
                settings=settings,
                config_path=config_path,
                end_date=end_date,
                benchmark_symbol=benchmark_symbol,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                workers=workers,
                spec=spec,
            )
        )
    rows = sorted(
        rows,
        key=lambda item: (
            -float(item.get("robust_score", 0.0)),
            -float(dict(item.get("summary", {})).get("excess_annual_return", 0.0)),
            -float(dict(item.get("summary", {})).get("information_ratio", 0.0)),
        ),
    )

    payload = {
        "end_date": end_date,
        "benchmark_symbol": benchmark_symbol,
        "combos": rows,
    }
    date_token = str(end_date).replace("-", "")
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / f"dynamic300_rule_fast_sweep_{date_token}.json"
    md_path = report_root / f"dynamic300_rule_fast_sweep_{date_token}.md"
    html_path = report_root / f"dynamic300_rule_fast_sweep_{date_token}.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        _markdown_report(end_date=end_date, benchmark_symbol=benchmark_symbol, rows=rows),
        encoding="utf-8",
    )
    html_path.write_text(
        _html_report(end_date=end_date, benchmark_symbol=benchmark_symbol, rows=rows),
        encoding="utf-8",
    )
    return {
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
        "html": str(html_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep the dynamic300 rule fast backtest across multiple parameter settings.")
    parser.add_argument("--config", default="config/api.json")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--lookbacks", default="120,180")
    parser.add_argument("--rebalances", default="10")
    parser.add_argument("--top-ns", default="4")
    parser.add_argument("--max-per-themes", default="1,2")
    parser.add_argument("--benchmark-symbol", default="000300.SH")
    parser.add_argument("--commission-bps", type=float, default=1.5)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    outputs = run_sweep(
        config_path=str(args.config),
        end_date=str(args.end_date),
        lookbacks=_parse_int_list(args.lookbacks),
        rebalance_intervals=_parse_int_list(args.rebalances),
        top_ns=_parse_int_list(args.top_ns),
        max_per_themes=_parse_int_list(args.max_per_themes),
        benchmark_symbol=str(args.benchmark_symbol),
        commission_bps=float(args.commission_bps),
        slippage_bps=float(args.slippage_bps),
        workers=max(1, int(args.workers)),
    )
    print(f"[sweep] markdown: {outputs['markdown']}")
    print(f"[sweep] html: {outputs['html']}")
    print(f"[sweep] json: {outputs['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
