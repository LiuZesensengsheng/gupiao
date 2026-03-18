from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.v2_universe_generator import generate_dynamic_universe


@dataclass(frozen=True)
class PlanRow:
    symbol: str
    name: str
    theme: str
    close: float
    ma20: float
    ma60: float
    refined_score: float
    fresh_pool_score: float
    fresh_pool_pass: bool
    recent_high_gap20: float
    amount_ratio20: float
    theme_selected_count: int
    theme_strength: float
    portfolio_score: float
    weight: float = 0.0


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coalesce(*values: object, default: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _load_daily_metrics(data_dir: str, symbol: str) -> dict[str, float]:
    path = Path(data_dir) / f"{symbol}.csv"
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return {"close": 0.0, "ma20": 0.0, "ma60": 0.0}
    closes = frame["close"]
    return {
        "close": float(closes.iloc[-1]),
        "ma20": float(closes.tail(20).mean()),
        "ma60": float(closes.tail(60).mean()),
    }


def _portfolio_score(row: dict[str, object], theme_count: int, theme_strength: float) -> float:
    refined = float(row.get("refined_score", 0.0))
    fresh = float(row.get("fresh_pool_score", 0.0))
    amount_ratio20 = float(row.get("amount_ratio20", 0.0))
    near_high = float(row.get("recent_high_gap20", -0.5))
    theme_count_score = min(1.0, max(0.0, float(theme_count) / 12.0))
    amount_balance = 1.0 - min(abs(amount_ratio20 - 1.15) / 1.05, 1.0)
    near_high_score = 1.0 - min(abs(near_high + 0.03) / 0.12, 1.0)
    return float(
        0.50 * refined
        + 0.20 * fresh
        + 0.12 * theme_count_score
        + 0.10 * float(theme_strength)
        + 0.04 * amount_balance
        + 0.04 * near_high_score
    )


def _is_primary_candidate(row: dict[str, object], theme_count: int) -> bool:
    return bool(
        bool(row.get("fresh_pool_pass", False))
        and float(row.get("fresh_pool_score", 0.0)) >= 0.55
        and theme_count >= 4
    )


def _select_portfolio(rows: list[PlanRow], *, top_n: int, max_per_theme: int) -> list[PlanRow]:
    selected: list[PlanRow] = []
    counts: dict[str, int] = {}
    for row in rows:
        if len(selected) >= top_n:
            break
        if counts.get(row.theme, 0) >= int(max_per_theme):
            continue
        selected.append(row)
        counts[row.theme] = counts.get(row.theme, 0) + 1
    if len(selected) < top_n:
        selected_symbols = {row.symbol for row in selected}
        for row in rows:
            if len(selected) >= top_n:
                break
            if row.symbol in selected_symbols:
                continue
            selected.append(row)
            selected_symbols.add(row.symbol)
    total_score = sum(max(row.portfolio_score, 1e-9) for row in selected)
    weighted: list[PlanRow] = []
    for row in selected:
        weight = max(row.portfolio_score, 1e-9) / max(total_score, 1e-9)
        weighted.append(
            PlanRow(
                symbol=row.symbol,
                name=row.name,
                theme=row.theme,
                close=row.close,
                ma20=row.ma20,
                ma60=row.ma60,
                refined_score=row.refined_score,
                fresh_pool_score=row.fresh_pool_score,
                fresh_pool_pass=row.fresh_pool_pass,
                recent_high_gap20=row.recent_high_gap20,
                amount_ratio20=row.amount_ratio20,
                theme_selected_count=row.theme_selected_count,
                theme_strength=row.theme_strength,
                portfolio_score=row.portfolio_score,
                weight=float(weight),
            )
        )
    return weighted


def _buy_zone(row: PlanRow) -> str:
    lower = max(row.ma20, row.close * 0.985)
    upper = row.close * 1.010
    if lower > upper:
        lower = min(row.close, row.ma20)
    return f"{lower:.2f} - {upper:.2f}"


def _avoid_zone(row: PlanRow) -> str:
    return f">{row.close * 1.035:.2f} 不追高 / <{row.ma20:.2f} 不接弱"


def _reduce_if(row: PlanRow) -> str:
    return f"冲高后回落且失守 {row.close:.2f}，或量比明显失衡"


def _exit_if(row: PlanRow) -> str:
    return f"日收盘跌破20日线 {row.ma20:.2f}；弱势延续则看60日线 {row.ma60:.2f}"


def _reason_lines(row: PlanRow) -> list[str]:
    return [
        f"fresh 分 {row.fresh_pool_score:.3f}，refined 分 {row.refined_score:.3f}",
        f"距20日高点 {abs(row.recent_high_gap20) * 100:.1f}%，量能比 {row.amount_ratio20:.2f}",
        f"主题 {row.theme} 当前入选 {row.theme_selected_count} 只，主题强度 {row.theme_strength:.3f}",
    ]


def _monitor_reason(row: PlanRow) -> str:
    if row.fresh_pool_pass:
        return "通过 fresh gate，但本轮被更高分主题/个股挤出核心组合。"
    return "未完全通过 fresh gate，保留观察，不做明日优先开仓。"


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
    lines: list[str] = []
    lines.append("# Dynamic 300 轻量规则操作单")
    lines.append("")
    lines.append(f"- 数据日期: {as_of_date}")
    lines.append("- 方案类型: 轻量规则版")
    lines.append("- 适用场景: 今天先看规则型明日计划，不等待完整 daily-run")
    lines.append(f"- dynamic 300 入池数: {selected_count}")
    lines.append(f"- fresh gate 通过数: {fresh_pool_pass_count}")
    lines.append("- 说明: 这份计划只使用今天更新后的日线数据 + dynamic 300 + 规则化组合约束，不使用旧 80 池 snapshot。")
    lines.append("")
    lines.append("## Fresh Funnel")
    lines.append("")
    lines.append("| 阶段 | 数量 |")
    lines.append("|---|---:|")
    for item in fresh_pool_funnel:
        lines.append(f"| {item['label']} | {item['count']} |")
    lines.append("")
    lines.append("## 当前主线")
    lines.append("")
    lines.append("| 主线 | 入池数 | 强度 |")
    lines.append("|---|---:|---:|")
    for item in top_themes:
        lines.append(f"| {item['theme']} | {item['selected_count']} | {float(item['theme_strength']):.3f} |")
    lines.append("")
    lines.append("## 明日买入候选")
    lines.append("")
    lines.append("| 排名 | 股票 | 主题 | 建议权重 | 买入区间 | 回避区间 |")
    lines.append("|---:|---|---|---:|---|---|")
    for index, row in enumerate(portfolio, start=1):
        lines.append(
            f"| {index} | {row.name} ({row.symbol}) | {row.theme} | {row.weight * 100:.1f}% | {_buy_zone(row)} | {_avoid_zone(row)} |"
        )
    lines.append("")
    for index, row in enumerate(portfolio, start=1):
        lines.append(f"### {index}. {row.name} ({row.symbol})")
        lines.append(f"- 最新价: {row.close:.2f}")
        lines.append(f"- 减仓条件: {_reduce_if(row)}")
        lines.append(f"- 退出条件: {_exit_if(row)}")
        for reason in _reason_lines(row):
            lines.append(f"- 理由: {reason}")
        lines.append("")
    lines.append("## 观察名单")
    lines.append("")
    lines.append("| 股票 | 主题 | 备注 |")
    lines.append("|---|---|---|")
    for row in monitors:
        lines.append(f"| {row.name} ({row.symbol}) | {row.theme} | {_monitor_reason(row)} |")
    lines.append("")
    lines.append("## 交易纪律")
    lines.append("")
    lines.append("- 这份是规则版计划，默认开仓不超过 4 只。")
    lines.append("- 同主题默认不超过 1 只，优先保证主题分散。")
    lines.append("- 若明天高开过猛，先不追；宁可错过，也不把规则版做成情绪追高版。")
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
    def esc(text: object) -> str:
        return html.escape(str(text))

    cards_html = []
    for index, row in enumerate(portfolio, start=1):
        reason_html = "".join(f"<li>{esc(item)}</li>" for item in _reason_lines(row))
        cards_html.append(
            (
                "<article class='card'>"
                f"<div class='rank'>#{index}</div>"
                f"<h3>{esc(row.name)} <span>{esc(row.symbol)}</span></h3>"
                f"<p class='meta'>{esc(row.theme)} | 最新价 {row.close:.2f} | 建议权重 {row.weight * 100:.1f}%</p>"
                f"<p><strong>买入区间</strong>: {esc(_buy_zone(row))}</p>"
                f"<p><strong>回避区间</strong>: {esc(_avoid_zone(row))}</p>"
                f"<p><strong>减仓条件</strong>: {esc(_reduce_if(row))}</p>"
                f"<p><strong>退出条件</strong>: {esc(_exit_if(row))}</p>"
                f"<ul>{reason_html}</ul>"
                "</article>"
            )
        )

    funnel_rows = "".join(
        f"<tr><td>{esc(item['label'])}</td><td>{int(item['count'])}</td></tr>"
        for item in fresh_pool_funnel
    )
    theme_rows = "".join(
        f"<tr><td>{esc(item['theme'])}</td><td>{int(item['selected_count'])}</td><td>{float(item['theme_strength']):.3f}</td></tr>"
        for item in top_themes
    )
    portfolio_rows = "".join(
        (
            f"<tr><td>{index}</td><td>{esc(row.name)} ({esc(row.symbol)})</td><td>{esc(row.theme)}</td>"
            f"<td>{row.weight * 100:.1f}%</td><td>{esc(_buy_zone(row))}</td><td>{esc(_avoid_zone(row))}</td></tr>"
        )
        for index, row in enumerate(portfolio, start=1)
    )
    monitor_rows = "".join(
        f"<tr><td>{esc(row.name)} ({esc(row.symbol)})</td><td>{esc(row.theme)}</td><td>{esc(_monitor_reason(row))}</td></tr>"
        for row in monitors
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dynamic 300 轻量规则操作单</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --panel: rgba(255,255,255,0.82);
      --ink: #172126;
      --muted: #5e6a67;
      --line: #d9d0c2;
      --accent: #9a4d1f;
      --accent-soft: #f1d4bf;
      --shadow: 0 18px 40px rgba(42, 29, 12, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(154,77,31,0.10), transparent 24%),
        radial-gradient(circle at left top, rgba(29,113,103,0.10), transparent 26%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }}
    .page {{ max-width: 1360px; margin: 0 auto; padding: 24px 18px 48px; }}
    .hero, .panel, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero {{ padding: 24px; }}
    .hero-grid, .triple, .cards {{ display: grid; gap: 16px; }}
    .hero-grid {{ grid-template-columns: 1.3fr 1fr; align-items: start; }}
    .triple {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 18px; }}
    .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .panel {{ padding: 18px; margin-top: 18px; }}
    .card {{ padding: 18px; position: relative; }}
    h1 {{ margin: 0; font-size: 38px; line-height: 1; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; }}
    h3 {{ margin: 0 0 8px; font-size: 20px; }}
    h3 span {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .facts {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .fact {{ padding: 14px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.55); }}
    .fact .eyebrow {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
    .fact strong {{ display: block; margin-top: 8px; font-size: 28px; }}
    .pill {{
      display: inline-flex;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.74);
      border: 1px solid rgba(0,0,0,0.08);
      font-size: 12px;
      font-weight: 700;
    }}
    .rank {{
      position: absolute;
      top: 14px;
      right: 14px;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 800;
    }}
    p, li {{ line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #eadfce; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ font-size: 11px; text-transform: uppercase; color: var(--muted); background: rgba(249, 242, 231, 0.9); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.48); }}
    .meta {{ color: var(--muted); margin-top: 0; }}
    @media (max-width: 1080px) {{
      .hero-grid, .triple, .cards, .facts {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <h1>Dynamic 300 轻量规则操作单</h1>
          <p class="muted">数据日期 {esc(as_of_date)}。这是今天补完数据后的轻量规则版，不等待完整 daily-run。</p>
          <div>
            <span class="pill">dynamic 300: {selected_count}</span>
            <span class="pill">fresh gate 通过: {fresh_pool_pass_count}</span>
            <span class="pill">核心开仓: {len(portfolio)}</span>
            <span class="pill">主题分散: 默认 1 主题 1 只</span>
          </div>
          <p class="muted">这版更像“今天先上手”的明日计划，优先解决速度和可执行性，不冒充完整 learned policy。</p>
        </div>
        <div class="facts">
          <div class="fact"><div class="eyebrow">核心候选</div><strong>{len(portfolio)}</strong><div class="muted">默认等价于 4 格仓位</div></div>
          <div class="fact"><div class="eyebrow">观察名单</div><strong>{len(monitors)}</strong><div class="muted">候选轮换缓冲区</div></div>
          <div class="fact"><div class="eyebrow">主线聚焦</div><strong>{esc(top_themes[0]['theme'] if top_themes else 'NA')}</strong><div class="muted">按今日 dynamic 300 统计</div></div>
        </div>
      </div>
      <div class="triple">
        <div class="panel"><h2>执行原则</h2><p>只做 fresh 分和 refined 分都靠前、且主题不太窄的票；高开过猛不追，跌破 20 日线不接。</p></div>
        <div class="panel"><h2>组合结构</h2><p>核心仓位控制在 4 只左右；规则版优先降低同主题扎堆，先求稳定再求极致进攻。</p></div>
        <div class="panel"><h2>风险声明</h2><p>这是轻量规则版，不含分钟级别择时，也不含完整 daily-run 的 learned policy / exit overlay。</p></div>
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
      <h2>明日买入候选</h2>
      <div class="table-wrap"><table><thead><tr><th>排名</th><th>股票</th><th>主题</th><th>建议权重</th><th>买入区间</th><th>回避区间</th></tr></thead><tbody>{portfolio_rows}</tbody></table></div>
    </section>
    <section class="panel">
      <h2>观察名单</h2>
      <div class="table-wrap"><table><thead><tr><th>股票</th><th>主题</th><th>备注</th></tr></thead><tbody>{monitor_rows}</tbody></table></div>
    </section>
    <section class="cards">{''.join(cards_html)}</section>
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
    payload = _load_config(config_path)
    daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
    common = payload.get("common", {}) if isinstance(payload, dict) else {}
    settings = {
        "data_dir": str(_coalesce(daily.get("data_dir"), common.get("data_dir"), default="data")),
        "cache_root": str(_coalesce(daily.get("cache_root"), default="cache")),
        "base_universe_file": str(
            _coalesce(
                daily.get("generated_universe_base_file"),
                daily.get("universe_file"),
                default="config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json",
            )
        ),
        "target_size": int(_coalesce(daily.get("dynamic_universe_target_size"), default=300)),
        "coarse_size": int(_coalesce(daily.get("dynamic_universe_coarse_size"), default=1000)),
        "min_history_days": int(_coalesce(daily.get("dynamic_universe_min_history_days"), default=480)),
        "min_recent_amount": float(_coalesce(daily.get("dynamic_universe_min_recent_amount"), default=100000.0)),
        "theme_cap_ratio": float(_coalesce(daily.get("dynamic_universe_theme_cap_ratio"), default=0.16)),
        "theme_floor_count": int(_coalesce(daily.get("dynamic_universe_theme_floor_count"), default=2)),
        "turnover_quality_weight": float(_coalesce(daily.get("dynamic_universe_turnover_quality_weight"), default=0.25)),
        "theme_weight": float(_coalesce(daily.get("dynamic_universe_theme_weight"), default=0.18)),
        "main_board_only": bool(_coalesce(daily.get("main_board_only_universe"), default=True)),
        "use_concepts": bool(_coalesce(daily.get("generator_use_concepts"), default=True)),
    }
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
    for row in result.selected_300:
        theme = str(row.get("sector", "其他"))
        theme_info = theme_map.get(theme, {"selected_count": 0, "theme_strength": 0.0})
        metrics = _load_daily_metrics(settings["data_dir"], str(row["symbol"]))
        row_payload = dict(row)
        score = _portfolio_score(
            row_payload,
            theme_count=int(theme_info["selected_count"]),
            theme_strength=float(theme_info["theme_strength"]),
        )
        if not _is_primary_candidate(row_payload, int(theme_info["selected_count"])):
            score *= 0.92
        plan_rows.append(
            PlanRow(
                symbol=str(row["symbol"]),
                name=str(row["name"]),
                theme=theme,
                close=float(metrics["close"]),
                ma20=float(metrics["ma20"]),
                ma60=float(metrics["ma60"]),
                refined_score=float(row.get("refined_score", 0.0)),
                fresh_pool_score=float(row.get("fresh_pool_score", 0.0)),
                fresh_pool_pass=bool(row.get("fresh_pool_pass", False)),
                recent_high_gap20=float(row.get("recent_high_gap20", 0.0)),
                amount_ratio20=float(row.get("amount_ratio20", 0.0)),
                theme_selected_count=int(theme_info["selected_count"]),
                theme_strength=float(theme_info["theme_strength"]),
                portfolio_score=score,
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
    portfolio = _select_portfolio(ranked, top_n=top_n, max_per_theme=max_per_theme)
    portfolio_symbols = {item.symbol for item in portfolio}
    monitors = [item for item in ranked if item.symbol not in portfolio_symbols][:8]
    top_themes = sorted(
        [
            {"theme": theme, "selected_count": meta["selected_count"], "theme_strength": meta["theme_strength"]}
            for theme, meta in theme_map.items()
        ],
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
    print(f"[rule-plan] markdown: {outputs['markdown']}")
    print(f"[rule-plan] html: {outputs['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
