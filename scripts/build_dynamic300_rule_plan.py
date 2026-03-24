from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dynamic300_rule_utils import (
    RuleCandidateRow,
    avoid_upper_bound,
    buy_zone_bounds,
    candidate_bucket,
    coalesce,
    is_primary_candidate,
    load_config,
    monitor_reason,
    portfolio_score,
    select_portfolio,
)
from src.application.v2_universe_generator import generate_dynamic_universe

PlanRow = RuleCandidateRow


def _load_settings(config_path: str) -> dict[str, Any]:
    payload = load_config(config_path)
    daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
    common = payload.get("common", {}) if isinstance(payload, dict) else {}
    return {
        "data_dir": str(coalesce(daily.get("data_dir"), common.get("data_dir"), default="data")),
        "cache_root": str(coalesce(daily.get("cache_root"), default="cache")),
        "base_universe_file": str(
            coalesce(
                daily.get("generated_universe_base_file"),
                daily.get("universe_file"),
                default="config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json",
            )
        ),
        "target_size": int(coalesce(daily.get("dynamic_universe_target_size"), default=300)),
        "coarse_size": int(coalesce(daily.get("dynamic_universe_coarse_size"), default=1000)),
        "min_history_days": int(coalesce(daily.get("dynamic_universe_min_history_days"), default=480)),
        "min_recent_amount": float(coalesce(daily.get("dynamic_universe_min_recent_amount"), default=100000.0)),
        "theme_cap_ratio": float(coalesce(daily.get("dynamic_universe_theme_cap_ratio"), default=0.16)),
        "theme_floor_count": int(coalesce(daily.get("dynamic_universe_theme_floor_count"), default=2)),
        "turnover_quality_weight": float(
            coalesce(daily.get("dynamic_universe_turnover_quality_weight"), default=0.25)
        ),
        "theme_weight": float(coalesce(daily.get("dynamic_universe_theme_weight"), default=0.18)),
        "main_board_only": bool(coalesce(daily.get("main_board_only_universe"), default=True)),
        "use_concepts": bool(coalesce(daily.get("generator_use_concepts"), default=True)),
    }


def _buy_zone(row: PlanRow) -> str:
    lower, upper = buy_zone_bounds(row)
    return f"{lower:.2f} - {upper:.2f}"


def _avoid_zone(row: PlanRow) -> str:
    return f">{avoid_upper_bound(row):.2f} 不追高 / <{row.ma20:.2f} 不接"


def _reduce_if(row: PlanRow) -> str:
    return (
        f"收盘跌回 20 日线 {row.ma20:.2f} 下方先减仓；"
        "如果冲高回落明显、量能不跟，也先降仓位。"
    )


def _exit_if(row: PlanRow) -> str:
    return f"有效跌破 60 日线 {row.ma60:.2f} 直接退出；20 日线失守后继续弱化也退出。"


def _reason_lines(row: PlanRow) -> list[str]:
    return [
        f"bucket={row.bucket}，fresh={row.fresh_pool_score:.3f}，refined={row.refined_score:.3f}",
        f"距 20 日高点 {abs(row.recent_high_gap20) * 100:.1f}%，量能比 {row.amount_ratio20:.2f}",
        f"主题 {row.theme} 当前入池 {row.theme_selected_count} 只，主题强度 {row.theme_strength:.3f}",
        f"近 20 日涨幅 {row.ret20:.1%}，近 60 日涨幅 {row.ret60:.1%}，120 日突破位置 {row.breakout_pos_120:.2f}",
    ]


def _markdown_report(
    *,
    as_of_date: str,
    selected_count: int,
    fresh_pool_pass_count: int,
    fresh_pool_funnel: list[dict[str, object]],
    top_themes: list[dict[str, object]],
    portfolio: list[PlanRow],
    monitors: list[PlanRow],
) -> str:
    lines = [
        "# Dynamic 300 规则版操作计划",
        "",
        f"- 数据日期: {as_of_date}",
        "- 方案类型: dynamic300 fast rule",
        "- 逻辑口径: dynamic 300 + 分桶候选 + 同主题限额 + 次日纪律计划",
        f"- dynamic 300 入池数: {selected_count}",
        f"- fresh gate 通过数: {fresh_pool_pass_count}",
        "",
        "## Fresh Funnel",
        "",
        "| 阶段 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| {item['label']} | {item['count']} |" for item in fresh_pool_funnel)
    lines.extend(
        [
            "",
            "## 当前主线",
            "",
            "| 主线 | 入池数 | 强度 |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {item['theme']} | {item['selected_count']} | {float(item['theme_strength']):.3f} |"
        for item in top_themes
    )
    lines.extend(
        [
            "",
            "## 次日买入候选",
            "",
            "| 排名 | 股票 | 主题 | Bucket | 建议权重 | 买入区间 | 回避区间 |",
            "|---:|---|---|---|---:|---|---|",
        ]
    )
    lines.extend(
        f"| {index} | {row.name} ({row.symbol}) | {row.theme} | {row.bucket} | {row.weight * 100:.1f}% | {_buy_zone(row)} | {_avoid_zone(row)} |"
        for index, row in enumerate(portfolio, start=1)
    )
    for index, row in enumerate(portfolio, start=1):
        lines.extend(
            [
                "",
                f"### {index}. {row.name} ({row.symbol})",
                f"- 最新价: {row.close:.2f}",
                f"- 减仓条件: {_reduce_if(row)}",
                f"- 退出条件: {_exit_if(row)}",
            ]
        )
        lines.extend(f"- 理由: {reason}" for reason in _reason_lines(row))
    lines.extend(
        [
            "",
            "## 观察名单",
            "",
            "| 股票 | 主题 | Bucket | 备注 |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(
        f"| {row.name} ({row.symbol}) | {row.theme} | {row.bucket} | {monitor_reason(row)} |"
        for row in monitors
    )
    lines.extend(
        [
            "",
            "## 交易纪律",
            "",
            "- 默认最多开 4 只，且同主题默认不超过 1 只。",
            "- 高开超过回避区间不追，跌破 20 日线不接。",
            "- 这份计划先强调可执行性，不把高波动追涨包装成纪律化交易。",
        ]
    )
    return "\n".join(lines)


def _html_report(
    *,
    as_of_date: str,
    selected_count: int,
    fresh_pool_pass_count: int,
    fresh_pool_funnel: list[dict[str, object]],
    top_themes: list[dict[str, object]],
    portfolio: list[PlanRow],
    monitors: list[PlanRow],
) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    card_html = []
    for index, row in enumerate(portfolio, start=1):
        reasons = "".join(f"<li>{esc(reason)}</li>" for reason in _reason_lines(row))
        card_html.append(
            (
                "<article class='card'>"
                f"<div class='rank'>#{index}</div>"
                f"<div class='bucket'>{esc(row.bucket)}</div>"
                f"<h3>{esc(row.name)} <span>{esc(row.symbol)}</span></h3>"
                f"<p class='meta'>{esc(row.theme)} | 最新价 {row.close:.2f} | 建议权重 {row.weight * 100:.1f}%</p>"
                f"<p><strong>买入区间</strong>: {esc(_buy_zone(row))}</p>"
                f"<p><strong>回避区间</strong>: {esc(_avoid_zone(row))}</p>"
                f"<p><strong>减仓条件</strong>: {esc(_reduce_if(row))}</p>"
                f"<p><strong>退出条件</strong>: {esc(_exit_if(row))}</p>"
                f"<ul>{reasons}</ul>"
                "</article>"
            )
        )

    funnel_rows = "".join(
        f"<tr><td>{esc(item['label'])}</td><td>{int(item['count'])}</td></tr>" for item in fresh_pool_funnel
    )
    theme_rows = "".join(
        f"<tr><td>{esc(item['theme'])}</td><td>{int(item['selected_count'])}</td><td>{float(item['theme_strength']):.3f}</td></tr>"
        for item in top_themes
    )
    portfolio_rows = "".join(
        (
            f"<tr><td>{index}</td><td>{esc(row.name)} ({esc(row.symbol)})</td><td>{esc(row.theme)}</td>"
            f"<td>{esc(row.bucket)}</td><td>{row.weight * 100:.1f}%</td><td>{esc(_buy_zone(row))}</td><td>{esc(_avoid_zone(row))}</td></tr>"
        )
        for index, row in enumerate(portfolio, start=1)
    )
    monitor_rows = "".join(
        f"<tr><td>{esc(row.name)} ({esc(row.symbol)})</td><td>{esc(row.theme)}</td><td>{esc(row.bucket)}</td><td>{esc(monitor_reason(row))}</td></tr>"
        for row in monitors
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dynamic 300 规则版操作计划</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --panel: rgba(255, 255, 255, 0.84);
      --ink: #142026;
      --muted: #57656b;
      --line: #d8cdbd;
      --accent: #8b4418;
      --accent-soft: #f2d8c3;
      --shadow: 0 18px 36px rgba(40, 28, 11, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(139,68,24,0.12), transparent 28%),
        radial-gradient(circle at left top, rgba(28,100,91,0.10), transparent 26%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 24px 18px 56px; }}
    .hero, .panel, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero {{ padding: 24px; }}
    .hero-grid, .facts, .cards {{ display: grid; gap: 16px; }}
    .hero-grid {{ grid-template-columns: 1.35fr 1fr; align-items: start; }}
    .facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .panel {{ margin-top: 18px; padding: 18px; }}
    .card {{ padding: 18px; position: relative; }}
    .fact {{
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.54);
    }}
    .eyebrow {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
    .fact strong {{ display: block; margin-top: 8px; font-size: 28px; }}
    h1 {{ margin: 0; font-size: 38px; line-height: 1.05; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; }}
    h3 {{ margin: 0 0 8px; font-size: 20px; }}
    h3 span {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    p, li {{ line-height: 1.6; }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-flex;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid rgba(0,0,0,0.08);
      background: rgba(255,255,255,0.72);
      font-size: 12px;
      font-weight: 700;
    }}
    .rank {{
      position: absolute;
      top: 14px;
      right: 14px;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
    }}
    .bucket {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(20, 32, 38, 0.06);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid #eadfce;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: rgba(249, 242, 231, 0.92);
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.48);
    }}
    @media (max-width: 1080px) {{
      .hero-grid, .facts, .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-grid">
        <div>
          <h1>Dynamic 300 规则版操作计划</h1>
          <p class="muted">数据日期 {esc(as_of_date)}。这份计划基于 dynamic 300 候选池、分桶排序和次日执行纪律，强调可执行性而不是盲目追高。</p>
          <div>
            <span class="pill">dynamic 300: {selected_count}</span>
            <span class="pill">fresh gate: {fresh_pool_pass_count}</span>
            <span class="pill">核心开仓: {len(portfolio)}</span>
            <span class="pill">同主题默认上限: 1</span>
          </div>
        </div>
        <div class="facts">
          <div class="fact"><div class="eyebrow">核心候选</div><strong>{len(portfolio)}</strong><div class="muted">默认控制在 4 只左右</div></div>
          <div class="fact"><div class="eyebrow">观察名单</div><strong>{len(monitors)}</strong><div class="muted">作为轮换和观察缓冲</div></div>
          <div class="fact"><div class="eyebrow">最强主线</div><strong>{esc(top_themes[0]['theme'] if top_themes else 'NA')}</strong><div class="muted">按当日 dynamic 300 统计</div></div>
          <div class="fact"><div class="eyebrow">执行原则</div><strong>先执行</strong><div class="muted">高开不过度追，失守均线先收缩</div></div>
        </div>
      </div>
    </section>
    <section class="panel">
      <h2>Fresh Funnel</h2>
      <div class="table-wrap"><table><thead><tr><th>阶段</th><th>数量</th></tr></thead><tbody>{funnel_rows}</tbody></table></div>
    </section>
    <section class="panel">
      <h2>当前主线</h2>
      <div class="table-wrap"><table><thead><tr><th>主线</th><th>入池数</th><th>强度</th></tr></thead><tbody>{theme_rows}</tbody></table></div>
    </section>
    <section class="panel">
      <h2>次日买入候选</h2>
      <div class="table-wrap"><table><thead><tr><th>排名</th><th>股票</th><th>主题</th><th>Bucket</th><th>建议权重</th><th>买入区间</th><th>回避区间</th></tr></thead><tbody>{portfolio_rows}</tbody></table></div>
    </section>
    <section class="panel">
      <h2>观察名单</h2>
      <div class="table-wrap"><table><thead><tr><th>股票</th><th>主题</th><th>Bucket</th><th>备注</th></tr></thead><tbody>{monitor_rows}</tbody></table></div>
    </section>
    <section class="cards">{''.join(card_html)}</section>
  </main>
</body>
</html>"""


def build_plan(
    *,
    config_path: str,
    end_date: str,
    refresh_cache: bool,
    top_n: int,
    max_per_theme: int,
) -> dict[str, str]:
    settings = _load_settings(config_path)
    result = generate_dynamic_universe(
        universe_file=settings["base_universe_file"],
        data_dir=settings["data_dir"],
        cache_root=settings["cache_root"],
        target_size=settings["target_size"],
        coarse_size=settings["coarse_size"],
        theme_aware=True,
        use_concepts=settings["use_concepts"],
        end_date=end_date,
        min_history_days=settings["min_history_days"],
        min_recent_amount=settings["min_recent_amount"],
        theme_cap_ratio=settings["theme_cap_ratio"],
        theme_floor_count=settings["theme_floor_count"],
        turnover_quality_weight=settings["turnover_quality_weight"],
        theme_weight=settings["theme_weight"],
        main_board_only=settings["main_board_only"],
        refresh_cache=refresh_cache,
    )
    theme_map = {
        str(item.theme): {
            "selected_count": int(item.selected_count),
            "theme_strength": float(item.theme_strength),
        }
        for item in result.theme_allocations
    }

    plan_rows: list[PlanRow] = []
    for raw in result.selected_300:
        row = dict(raw)
        theme = str(row.get("sector", "其他"))
        theme_info = theme_map.get(theme, {"selected_count": 0, "theme_strength": 0.0})
        bucket = candidate_bucket(row)
        score = portfolio_score(
            {**row, "bucket": bucket},
            theme_count=int(theme_info["selected_count"]),
            theme_strength=float(theme_info["theme_strength"]),
        )
        if not is_primary_candidate({**row, "bucket": bucket}, int(theme_info["selected_count"])):
            score *= 0.92
        plan_rows.append(
            PlanRow(
                symbol=str(row["symbol"]),
                name=str(row["name"]),
                theme=theme,
                refined_score=float(row.get("refined_score", 0.0)),
                fresh_pool_score=float(row.get("fresh_pool_score", 0.0)),
                fresh_pool_pass=bool(row.get("fresh_pool_pass", False)),
                recent_high_gap20=float(row.get("recent_high_gap20", 0.0)),
                amount_ratio20=float(row.get("amount_ratio20", 0.0)),
                theme_selected_count=int(theme_info["selected_count"]),
                theme_strength=float(theme_info["theme_strength"]),
                close=float(row.get("close", 0.0)),
                ma20=float(row.get("ma20", 0.0)),
                ma60=float(row.get("ma60", 0.0)),
                ret20=float(row.get("ret20", 0.0)),
                ret60=float(row.get("ret60", 0.0)),
                breakout_pos_120=float(row.get("breakout_pos_120", 0.0)),
                volatility20=float(row.get("volatility20", 0.0)),
                tradeability=float(row.get("tradeability", 0.0)),
                bucket=bucket,
                portfolio_score=float(score),
            )
        )

    ranked = sorted(
        plan_rows,
        key=lambda item: (
            -item.portfolio_score,
            -item.refined_score,
            -item.fresh_pool_score,
            -item.theme_selected_count,
            item.symbol,
        ),
    )
    portfolio = select_portfolio(ranked, top_n=top_n, max_per_theme=max_per_theme)
    selected_symbols = {row.symbol for row in portfolio}
    monitors = [row for row in ranked if row.symbol not in selected_symbols][:8]
    top_themes = sorted(
        (
            {"theme": theme, "selected_count": meta["selected_count"], "theme_strength": meta["theme_strength"]}
            for theme, meta in theme_map.items()
        ),
        key=lambda item: (-int(item["selected_count"]), -float(item["theme_strength"]), str(item["theme"])),
    )[:10]
    fresh_pool_funnel = list(result.generator_manifest.config.get("fresh_pool_funnel", []))
    fresh_pool_pass_count = int(result.generator_manifest.config.get("fresh_pool_pass_count", 0))

    date_token = str(end_date).replace("-", "")
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)
    markdown_path = report_root / f"daily_plan_{date_token}_dynamic300_rule.md"
    html_path = report_root / f"daily_plan_{date_token}_dynamic300_rule.html"

    markdown_path.write_text(
        _markdown_report(
            as_of_date=end_date,
            selected_count=len(result.selected_300),
            fresh_pool_pass_count=fresh_pool_pass_count,
            fresh_pool_funnel=fresh_pool_funnel,
            top_themes=top_themes,
            portfolio=portfolio,
            monitors=monitors,
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        _html_report(
            as_of_date=end_date,
            selected_count=len(result.selected_300),
            fresh_pool_pass_count=fresh_pool_pass_count,
            fresh_pool_funnel=fresh_pool_funnel,
            top_themes=top_themes,
            portfolio=portfolio,
            monitors=monitors,
        ),
        encoding="utf-8",
    )
    return {
        "markdown": str(markdown_path.resolve()),
        "html": str(html_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight rule-based plan from today's dynamic 300 universe.")
    parser.add_argument("--config", default="config/api.json")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--max-per-theme", type=int, default=1)
    args = parser.parse_args()

    outputs = build_plan(
        config_path=str(args.config),
        end_date=str(args.end_date),
        refresh_cache=bool(args.refresh_cache),
        top_n=max(1, int(args.top_n)),
        max_per_theme=max(1, int(args.max_per_theme)),
    )
    print(f"[dynamic300-rule-plan] markdown: {outputs['markdown']}")
    print(f"[dynamic300-rule-plan] html: {outputs['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
