from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.use_cases import DailyFusionResult
from src.domain.policies import market_regime, target_exposure


def _pct(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _bp(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 10000:.0f}bp"


def _score(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:+.3f}"


def _mix_hex(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    r = int(a[0] + (b[0] - a[0]) * t)
    g = int(a[1] + (b[1] - a[1]) * t)
    bl = int(a[2] + (b[2] - a[2]) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


def _score_color(v: float) -> str:
    if pd.isna(v):
        return "#7f8c8d"
    if v >= 0:
        return _mix_hex((236, 86, 86), (212, 48, 48), min(1.0, v))
    return _mix_hex((57, 126, 255), (24, 91, 217), min(1.0, -v))


def _prob_color(v: float) -> str:
    if pd.isna(v):
        return "#7f8c8d"
    return _mix_hex((66, 117, 255), (222, 71, 54), v)


def _bar_html(value: float, scale: float = 1.0, color: str = "#3e8cf8") -> str:
    if pd.isna(value):
        return '<div class="bar"><span style="width:0%"></span></div>'
    width = max(0.0, min(100.0, abs(value / scale) * 100.0))
    return f'<div class="bar"><span style="width:{width:.1f}%; background:{color};"></span></div>'


def write_daily_dashboard(out_path: str | Path, result: DailyFusionResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    regime = market_regime(result.market_final_short, result.market_final_mid)
    exposure = target_exposure(result.market_final_short, result.market_final_mid)

    stock_rows = sorted(result.blended_rows, key=lambda x: x.final_score, reverse=True)
    sector_rows = result.sector_table.to_dict("records") if not result.sector_table.empty else []

    stock_table_rows: list[str] = []
    matrix_rows: list[str] = []
    for row in stock_rows:
        name = escape(str(row.name))
        symbol = escape(str(row.symbol))
        stock_table_rows.append(
            "<tr>"
            f"<td><div class='sym'>{name}</div><div class='code'>{symbol}</div></td>"
            f"<td>{_pct(row.final_short)}</td>"
            f"<td>{_pct(row.final_mid)}</td>"
            f"<td class='score' style='color:{_score_color(row.final_score * 2 - 1)}'>{row.final_score:.3f}</td>"
            f"<td>{_bar_html(row.final_score, scale=1.0, color=_prob_color(row.final_score))}</td>"
            f"<td>{_pct(row.suggested_weight)}</td>"
            "</tr>"
        )
        matrix_rows.append(
            "<tr>"
            f"<td><div class='sym'>{name}</div><div class='code'>{symbol}</div></td>"
            "<td>短期</td>"
            f"<td><span class='pill bull'>{row.short_sent.bullish:.3f}</span></td>"
            f"<td><span class='pill bear'>{row.short_sent.bearish:.3f}</span></td>"
            f"<td>{row.short_sent.neutral:.3f}</td>"
            f"<td>{row.short_sent.items}</td>"
            "</tr>"
        )
        matrix_rows.append(
            "<tr>"
            f"<td><div class='sym'>{name}</div><div class='code'>{symbol}</div></td>"
            "<td>中期</td>"
            f"<td><span class='pill bull'>{row.mid_sent.bullish:.3f}</span></td>"
            f"<td><span class='pill bear'>{row.mid_sent.bearish:.3f}</span></td>"
            f"<td>{row.mid_sent.neutral:.3f}</td>"
            f"<td>{row.mid_sent.items}</td>"
            "</tr>"
        )

    sector_lines: list[str] = []
    for row in sector_rows:
        heat = float(row["heat_score"])
        color = _score_color(heat)
        sector_lines.append(
            "<tr>"
            f"<td>{escape(str(row['sector']))}</td>"
            f"<td class='score' style='color:{color}'>{_score(heat)}</td>"
            f"<td>{_pct(float(row['win_rate_1d']))}</td>"
            f"<td>{_bp(float(row['median_ret_5d']))}</td>"
            f"<td>{_score(float(row['money_score']))}</td>"
            f"<td>{_score(float(row['chip_score']))}</td>"
            f"<td>{int(row['count'])}</td>"
            f"<td>{_bar_html(heat, scale=1.0, color=color)}</td>"
            "</tr>"
        )

    eff = result.effect_summary
    market = result.market_forecast

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>A股每日融合仪表盘</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #14213d;
      --muted: #6b7280;
      --line: #e8edf5;
      --bull: #cf3131;
      --bear: #1f63d8;
      --grad: linear-gradient(145deg, #163877 0%, #2158b6 42%, #4b86e8 100%);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, #e7f0ff 0%, transparent 42%),
        radial-gradient(circle at 95% 10%, #ffecec 0%, transparent 35%),
        var(--bg);
    }}
    .wrap {{ max-width: 1160px; margin: 0 auto; padding: 20px 14px 36px; }}
    .hero {{
      background: var(--grad);
      color: #fff;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 12px 40px rgba(21, 62, 130, 0.22);
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -120px;
      top: -80px;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.12);
    }}
    h1 {{ margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.02em; }}
    .meta {{
      margin-top: 10px;
      color: rgba(255,255,255,0.92);
      font-size: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
    }}
    .meta span {{
      padding: 4px 10px;
      border: 1px solid rgba(255,255,255,0.25);
      border-radius: 999px;
    }}
    .grid {{
      margin-top: 14px;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 22px rgba(15, 32, 64, 0.06);
    }}
    .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .v {{ margin-top: 6px; font-size: 26px; font-weight: 700; }}
    .sub {{ margin-top: 4px; font-size: 13px; color: var(--muted); }}
    h2 {{ margin: 18px 0 10px; font-size: 18px; letter-spacing: 0.02em; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
    }}
    th {{ background: #f6f9ff; color: #31456f; font-weight: 700; }}
    tr:last-child td {{ border-bottom: none; }}
    .sym {{ font-weight: 700; }}
    .code {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
    .score {{ font-weight: 700; }}
    .bar {{ width: 100%; height: 9px; border-radius: 999px; background: #e8edf5; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; border-radius: 999px; }}
    .pill {{
      display: inline-block;
      padding: 2px 9px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
      min-width: 56px;
      text-align: center;
    }}
    .bull {{ background: rgba(207,49,49,0.1); color: var(--bull); }}
    .bear {{ background: rgba(31,99,216,0.1); color: var(--bear); }}
    .section-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: 2fr 1fr;
      margin-top: 14px;
    }}
    .note {{ margin-top: 14px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .section-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>A股每日融合仪表盘</h1>
      <div class="meta">
        <span>报告日期 {escape(str(result.as_of_date.date()))}</span>
        <span>数据源 {escape(result.source)}</span>
        <span>{escape(market.name)} ({escape(market.symbol)})</span>
        <span>市场状态 {escape(regime)}</span>
        <span>新闻 {result.news_items_count} 条</span>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <div class="k">建议总仓位</div>
        <div class="v">{_pct(exposure)}</div>
        <div class="sub">依据大盘融合概率自动映射</div>
      </article>
      <article class="card">
        <div class="k">大盘短期概率</div>
        <div class="v">{_pct(result.market_final_short)}</div>
        <div class="sub">模型 {_pct(market.short_prob)} / 新闻净分 {_score(result.market_short_sent.score)}</div>
      </article>
      <article class="card">
        <div class="k">大盘中期概率</div>
        <div class="v">{_pct(result.market_final_mid)}</div>
        <div class="sub">模型 {_pct(market.mid_prob)} / 新闻净分 {_score(result.market_mid_sent.score)}</div>
      </article>
      <article class="card">
        <div class="k">风险温度</div>
        <div class="v">{escape(eff.risk_label)}</div>
        <div class="sub">亏钱效应与回撤共同决定</div>
      </article>
    </section>

    <div class="section-grid">
      <section>
        <h2>个股综合排序</h2>
        <table>
          <thead>
            <tr><th>个股</th><th>融合短期</th><th>融合中期</th><th>综合分数</th><th>分数强度</th><th>建议权重</th></tr>
          </thead>
          <tbody>
            {''.join(stock_table_rows) if stock_table_rows else '<tr><td colspan="6">无数据</td></tr>'}
          </tbody>
        </table>
      </section>

      <section>
        <h2>效应面板</h2>
        <article class="card">
          <div class="k">赚钱效应</div>
          <div class="v">{escape(eff.pnl_label)}</div>
          <div class="sub">1日胜率 {_pct(eff.win_rate_1d)} / 5日胜率 {_pct(eff.win_rate_5d)}</div>
        </article>
        <article class="card" style="margin-top:10px;">
          <div class="k">亏钱效应</div>
          <div class="v">{escape(eff.risk_label)}</div>
          <div class="sub">深亏率 {_pct(eff.deep_loss_rate)} / 中位回撤 {_pct(eff.median_drawdown_20)}</div>
        </article>
        <article class="card" style="margin-top:10px;">
          <div class="k">资金状态</div>
          <div class="v">{escape(eff.money_label)}</div>
          <div class="sub">量能比 {eff.avg_vol_ratio_20:.2f} / 资金分数 {_score(eff.money_score)}</div>
        </article>
        <article class="card" style="margin-top:10px;">
          <div class="k">筹码结构</div>
          <div class="v">{escape(eff.chip_label)}</div>
          <div class="sub">20日位置 {eff.avg_price_pos_20:.2f} / 筹码分数 {_score(eff.chip_score)}</div>
        </article>
      </section>
    </div>

    <section>
      <h2>新闻模糊矩阵</h2>
      <table>
        <thead>
          <tr><th>标的</th><th>周期</th><th>利好隶属</th><th>利空隶属</th><th>中性隶属</th><th>条数</th></tr>
        </thead>
        <tbody>
          <tr><td><div class="sym">{escape(market.name)}</div><div class="code">{escape(market.symbol)}</div></td><td>短期</td><td><span class="pill bull">{result.market_short_sent.bullish:.3f}</span></td><td><span class="pill bear">{result.market_short_sent.bearish:.3f}</span></td><td>{result.market_short_sent.neutral:.3f}</td><td>{result.market_short_sent.items}</td></tr>
          <tr><td><div class="sym">{escape(market.name)}</div><div class="code">{escape(market.symbol)}</div></td><td>中期</td><td><span class="pill bull">{result.market_mid_sent.bullish:.3f}</span></td><td><span class="pill bear">{result.market_mid_sent.bearish:.3f}</span></td><td>{result.market_mid_sent.neutral:.3f}</td><td>{result.market_mid_sent.items}</td></tr>
          {''.join(matrix_rows) if matrix_rows else '<tr><td colspan="6">无数据</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>板块热度</h2>
      <table>
        <thead>
          <tr><th>板块</th><th>热度</th><th>1日胜率</th><th>5日中位收益</th><th>资金</th><th>筹码</th><th>样本</th><th>热度条</th></tr>
        </thead>
        <tbody>
          {''.join(sector_lines) if sector_lines else '<tr><td colspan="8">无可用板块数据</td></tr>'}
        </tbody>
      </table>
    </section>

    <p class="note">说明：本仪表盘为统计研究结果，不构成投资建议。请结合交易规则、流动性与风险约束使用。</p>
  </div>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")
    return path

