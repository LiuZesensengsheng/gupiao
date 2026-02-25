from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.use_cases import DailyFusionResult
from src.interfaces.presenters.driver_explainer import format_driver_list


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


def _num(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


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


def _line_chart_html(curve: pd.DataFrame) -> str:
    if curve.empty:
        return "<div class='empty'>无可用回测曲线数据</div>"

    data = curve.sort_values("date").reset_index(drop=True)
    has_quant_compare = "quant_nav" in data.columns
    values = [data["strategy_nav"].to_numpy(dtype=float), data["benchmark_nav"].to_numpy(dtype=float), data["excess_nav"].to_numpy(dtype=float)]
    if has_quant_compare:
        values.append(data["quant_nav"].to_numpy(dtype=float))
    values_arr = np.concatenate(values)
    y_min = float(np.nanmin(values_arr))
    y_max = float(np.nanmax(values_arr))
    span = max(1e-6, y_max - y_min)
    y_pad = span * 0.08
    y_min -= y_pad
    y_max += y_pad
    y_span = max(1e-6, y_max - y_min)

    width = 980.0
    height = 310.0
    px = 52.0
    py = 20.0
    plot_w = width - px - 16.0
    plot_h = height - py - 40.0
    n = len(data)

    def _x(i: int) -> float:
        if n <= 1:
            return px
        return px + plot_w * i / float(n - 1)

    def _y(v: float) -> float:
        return py + (1.0 - (v - y_min) / y_span) * plot_h

    def _poly(col: str, color: str, dashed: bool = False, element_id: str = "", hidden: bool = False) -> str:
        pts = " ".join(f"{_x(i):.1f},{_y(float(v)):.1f}" for i, v in enumerate(data[col].to_numpy(dtype=float)))
        dash_attr = " stroke-dasharray='6 4'" if dashed else ""
        id_attr = f" id='{element_id}'" if element_id else ""
        style_attr = " style='display:none;'" if hidden else ""
        return f"<polyline{id_attr}{style_attr} fill='none' stroke='{color}' stroke-width='2.4'{dash_attr} points='{pts}'/>"

    grid_lines = []
    for r in range(5):
        y = py + plot_h * r / 4.0
        v = y_max - (y_max - y_min) * r / 4.0
        grid_lines.append(
            f"<line x1='{px:.1f}' y1='{y:.1f}' x2='{(px + plot_w):.1f}' y2='{y:.1f}' stroke='#e6edf7' stroke-width='1'/>"
        )
        grid_lines.append(
            f"<text x='{(px - 6):.1f}' y='{(y + 4):.1f}' text-anchor='end' fill='#6b7280' font-size='11'>{v:.2f}</text>"
        )

    start_text = escape(str(pd.Timestamp(data["date"].iloc[0]).date()))
    end_text = escape(str(pd.Timestamp(data["date"].iloc[-1]).date()))

    toggle_html = ""
    script_html = ""
    spread_html = ""
    legend_mode_text = "策略净值"
    quant_legend_html = ""
    strategy_line = _poly("strategy_nav", "#d43030")
    quant_line = ""
    if has_quant_compare:
        legend_mode_text = "融合策略净值"
        strategy_line = _poly("strategy_nav", "#d43030", element_id="line-fused")
        quant_line = _poly("quant_nav", "#c27a00", dashed=True, element_id="line-quant", hidden=True)
        quant_legend_html = "<span id='legend-quant' style='display:none;'><i style='background:#c27a00;'></i>量化基线净值</span>"
        toggle_html = (
            "<div class='curve-toggle'>"
            "<button type='button' id='toggle-fused' class='active'>看融合策略</button>"
            "<button type='button' id='toggle-quant'>看量化基线</button>"
            "</div>"
        )
        spread = data["strategy_nav"].to_numpy(dtype=float) / np.maximum(data["quant_nav"].to_numpy(dtype=float), 1e-12)
        spread_min = float(np.nanmin(spread))
        spread_max = float(np.nanmax(spread))
        spread_span = max(1e-6, spread_max - spread_min)
        spread_pad = spread_span * 0.15
        spread_min -= spread_pad
        spread_max += spread_pad
        spread_span = max(1e-6, spread_max - spread_min)
        spread_note_html = ""
        if len(spread) >= 40:
            spread_move = np.abs(np.diff(spread))
            move_tol = 1e-10
            move_idx = np.where(spread_move > move_tol)[0] + 1
            move_days = int(move_idx.size)
            sparse_cutoff = max(8, int(len(spread) * 0.08))
            if move_days <= sparse_cutoff:
                if move_days > 0:
                    first_move_date = escape(str(pd.Timestamp(data["date"].iloc[int(move_idx[0])]).date()))
                    last_move_date = escape(str(pd.Timestamp(data["date"].iloc[int(move_idx[-1])]).date()))
                    spread_note_html = (
                        "<div class='subcurve-note'>"
                        f"提示：回测期仅 {move_days} 个交易日出现融合与基线差异（{first_move_date} 至 {last_move_date}），"
                        "其余区间两条曲线基本重合。若希望全周期都有区分度，请补充更早历史新闻。"
                        "</div>"
                    )
                else:
                    spread_note_html = (
                        "<div class='subcurve-note'>"
                        "提示：回测期融合与基线曲线完全重合，说明当前新闻样本对该区间未形成有效差异。"
                        "</div>"
                    )

        def _y_spread(v: float) -> float:
            return py + (1.0 - (v - spread_min) / spread_span) * plot_h

        spread_grid = []
        for r in range(5):
            y = py + plot_h * r / 4.0
            v = spread_max - (spread_max - spread_min) * r / 4.0
            spread_grid.append(
                f"<line x1='{px:.1f}' y1='{y:.1f}' x2='{(px + plot_w):.1f}' y2='{y:.1f}' stroke='#eef3fa' stroke-width='1'/>"
            )
            spread_grid.append(
                f"<text x='{(px - 6):.1f}' y='{(y + 4):.1f}' text-anchor='end' fill='#6b7280' font-size='11'>{v:.3f}</text>"
            )
        spread_pts = " ".join(f"{_x(i):.1f},{_y_spread(float(v)):.1f}" for i, v in enumerate(spread))
        spread_html = (
            "<div class='subcurve-wrap'>"
            "<div class='subcurve-title'>融合/基线 相对净值 (1.00 为持平)</div>"
            f"{spread_note_html}"
            "<svg viewBox='0 0 980 310' role='img' aria-label='融合与基线差值曲线'>"
            f"{''.join(spread_grid)}"
            f"<polyline fill='none' stroke='#7a55d1' stroke-width='2.4' points='{spread_pts}'/>"
            f"<line x1='{px:.1f}' y1='{_y_spread(1.0):.1f}' x2='{(px + plot_w):.1f}' y2='{_y_spread(1.0):.1f}' stroke='#9aa8be' stroke-dasharray='5 4' stroke-width='1.1'/>"
            f"<line x1='{px:.1f}' y1='{(py + plot_h):.1f}' x2='{(px + plot_w):.1f}' y2='{(py + plot_h):.1f}' stroke='#9eb0cd' stroke-width='1.1'/>"
            f"<text x='{px:.1f}' y='{(py + plot_h + 18):.1f}' fill='#6b7280' font-size='11'>{start_text}</text>"
            f"<text x='{(px + plot_w):.1f}' y='{(py + plot_h + 18):.1f}' text-anchor='end' fill='#6b7280' font-size='11'>{end_text}</text>"
            "</svg>"
            "</div>"
        )
        script_html = (
            "<script>"
            "(function(){"
            "const fusedBtn=document.getElementById('toggle-fused');"
            "const quantBtn=document.getElementById('toggle-quant');"
            "const fusedLine=document.getElementById('line-fused');"
            "const quantLine=document.getElementById('line-quant');"
            "const modeText=document.getElementById('legend-mode-text');"
            "const quantLegend=document.getElementById('legend-quant');"
            "if(!fusedBtn||!quantBtn||!fusedLine||!quantLine||!modeText||!quantLegend){return;}"
            "function showFused(){fusedBtn.classList.add('active');quantBtn.classList.remove('active');fusedLine.style.display='';quantLine.style.display='none';modeText.textContent='融合策略净值';quantLegend.style.display='none';}"
            "function showQuant(){quantBtn.classList.add('active');fusedBtn.classList.remove('active');fusedLine.style.display='none';quantLine.style.display='';modeText.textContent='量化基线净值';quantLegend.style.display='';}"
            "fusedBtn.addEventListener('click',showFused);"
            "quantBtn.addEventListener('click',showQuant);"
            "showFused();"
            "})();"
            "</script>"
        )

    return (
        "<div class='curve-wrap'>"
        f"{toggle_html}"
        "<svg viewBox='0 0 980 310' role='img' aria-label='策略与基准收益曲线'>"
        f"{''.join(grid_lines)}"
        f"{strategy_line}"
        f"{quant_line}"
        f"{_poly('benchmark_nav', '#1f63d8')}"
        f"{_poly('excess_nav', '#0f8a64')}"
        f"<line x1='{px:.1f}' y1='{(py + plot_h):.1f}' x2='{(px + plot_w):.1f}' y2='{(py + plot_h):.1f}' stroke='#9eb0cd' stroke-width='1.1'/>"
        f"<text x='{px:.1f}' y='{(py + plot_h + 18):.1f}' fill='#6b7280' font-size='11'>{start_text}</text>"
        f"<text x='{(px + plot_w):.1f}' y='{(py + plot_h + 18):.1f}' text-anchor='end' fill='#6b7280' font-size='11'>{end_text}</text>"
        "</svg>"
        "<div class='legend'>"
        "<span><i style='background:#d43030;'></i><span id='legend-mode-text'>"
        f"{legend_mode_text}"
        "</span></span>"
        f"{quant_legend_html}"
        "<span><i style='background:#1f63d8;'></i>沪深300基准</span>"
        "<span><i style='background:#0f8a64;'></i>超额净值</span>"
        "</div>"
        f"{spread_html}"
        "</div>"
        f"{script_html}"
    )


def write_daily_dashboard(out_path: str | Path, result: DailyFusionResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    regime = result.market_state_label
    exposure = float(result.effective_total_exposure)

    stock_rows = sorted(result.blended_rows, key=lambda x: x.final_score, reverse=True)
    stock_rows_display = stock_rows[:20]
    sector_rows = result.sector_table.to_dict("records") if not result.sector_table.empty else []

    stock_table_rows: list[str] = []
    matrix_rows: list[str] = []
    driver_rows: list[str] = []
    for row in stock_rows_display:
        name = escape(str(row.name))
        symbol = escape(str(row.symbol))
        short_drivers = escape(format_driver_list(row.short_drivers))
        mid_drivers = escape(format_driver_list(row.mid_drivers))
        stock_table_rows.append(
            "<tr>"
            f"<td><div class='sym'>{name}</div><div class='code'>{symbol}</div></td>"
            f"<td>{_pct(row.final_short)}</td>"
            f"<td>{_pct(row.final_mid)}</td>"
            f"<td class='score' style='color:{_score_color(row.final_score * 2 - 1)}'>{row.final_score:.3f}</td>"
            f"<td>{_bar_html(row.final_score, scale=1.0, color=_prob_color(row.final_score))}</td>"
            f"<td>{_pct(row.suggested_weight)}</td>"
            f"<td>{escape(row.fusion_mode_short)}</td>"
            f"<td>{escape(row.fusion_mode_mid)}</td>"
            f"<td>{'高位巨量阴线风险' if row.volume_risk_flag else '正常'}</td>"
            "</tr>"
        )
        driver_rows.append(
            "<tr>"
            f"<td><div class='sym'>{name}</div><div class='code'>{symbol}</div></td>"
            f"<td>{short_drivers}</td>"
            f"<td>{mid_drivers}</td>"
            f"<td>{escape(row.volume_risk_note or 'NA')}</td>"
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

    backtest_rows: list[str] = []
    for metrics in result.backtest_metrics:
        start_text = escape(str(metrics.start_date.date())) if not pd.isna(metrics.start_date) else "NA"
        end_text = escape(str(metrics.end_date.date())) if not pd.isna(metrics.end_date) else "NA"
        backtest_rows.append(
            "<tr>"
            f"<td>{escape(metrics.label)}</td>"
            f"<td>{start_text}</td>"
            f"<td>{end_text}</td>"
            f"<td>{metrics.n_days}</td>"
            f"<td>{_pct(metrics.total_return)}</td>"
            f"<td>{_pct(metrics.annual_return)}</td>"
            f"<td>{_pct(metrics.excess_annual_return)}</td>"
            f"<td>{_pct(metrics.max_drawdown)}</td>"
            f"<td>{_num(metrics.sharpe)}</td>"
            f"<td>{_num(metrics.sortino)}</td>"
            f"<td>{_num(metrics.calmar)}</td>"
            f"<td>{_num(metrics.information_ratio)}</td>"
            f"<td>{_pct(metrics.win_rate)}</td>"
            f"<td>{_pct(metrics.annual_turnover)}</td>"
            f"<td>{_pct(metrics.total_cost)}</td>"
            f"<td>{_num(metrics.avg_trades_per_stock_per_week, 2)}</td>"
            "</tr>"
        )
    strategy_rows: list[str] = []
    for trial in result.strategy_trials:
        strategy_rows.append(
            "<tr>"
            f"<td>{trial.rank}</td>"
            f"<td>{escape(trial.metric_label)}</td>"
            f"<td>{trial.retrain_days}</td>"
            f"<td>{trial.weight_threshold:.2f}</td>"
            f"<td>{trial.max_positions}</td>"
            f"<td>{trial.market_news_strength:.2f}</td>"
            f"<td>{trial.stock_news_strength:.2f}</td>"
            f"<td>{_num(trial.objective_score, 4)}</td>"
            f"<td>{_pct(trial.annual_return)}</td>"
            f"<td>{_pct(trial.excess_annual_return)}</td>"
            f"<td>{_pct(trial.max_drawdown)}</td>"
            f"<td>{_pct(trial.annual_turnover)}</td>"
            f"<td>{_pct(trial.total_cost)}</td>"
            f"<td>{_num(trial.avg_trades_per_stock_per_week, 2)}</td>"
            f"<td>{_num(trial.sharpe)}</td>"
            "</tr>"
        )
    strategy_summary = "默认参数（优化器关闭或无有效试验）"
    if result.strategy_selected is not None:
        selected = result.strategy_selected
        strategy_summary = (
            f"已选参数: retrain={selected.retrain_days}日, 阈值={selected.weight_threshold:.2f}, 持仓上限={selected.max_positions}, "
            f"市场新闻强度={selected.market_news_strength:.2f}, 个股新闻强度={selected.stock_news_strength:.2f}; "
            f"单票周交易={_num(selected.avg_trades_per_stock_per_week, 2)} 次; 目标得分={_num(selected.objective_score, 4)}"
        )
    learning_rows: list[str] = []
    for diag in result.learning_diagnostics:
        learning_rows.append(
            "<tr>"
            f"<td>{escape(diag.target)}</td>"
            f"<td>{escape(diag.horizon)}</td>"
            f"<td>{escape(diag.mode)}</td>"
            f"<td>{escape(diag.reason)}</td>"
            f"<td>{diag.samples}</td>"
            f"<td>{diag.holdout_n}</td>"
            f"<td>{_pct(diag.holdout_accuracy)}</td>"
            f"<td>{_num(diag.holdout_auc, 3)}</td>"
            f"<td>{_num(diag.holdout_brier, 3)}</td>"
            f"<td>{_num(diag.news_coef_score, 3)}</td>"
            f"<td>{_num(diag.fusion_coef_quant, 3)}</td>"
            f"<td>{_num(diag.fusion_coef_news, 3)}</td>"
            "</tr>"
        )
    curve_html = _line_chart_html(result.backtest_curve)

    eff = result.effect_summary
    market = result.market_forecast
    acceptance_text = "通过" if result.acceptance_ab_pass and result.acceptance_constraints_pass else "未通过"
    acceptance_sub = (
        f"A/B {'通过' if result.acceptance_ab_pass else '未通过'} | 约束 {'通过' if result.acceptance_constraints_pass else '未通过'}"
    )

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
    .table-scroll {{
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: auto;
      max-height: 560px;
      background: #fff;
    }}
    .table-scroll table {{
      border: none;
      border-radius: 0;
      min-width: 980px;
      margin: 0;
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
    .curve-wrap {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 10px 8px;
      box-shadow: 0 8px 22px rgba(15, 32, 64, 0.05);
    }}
    .curve-toggle {{
      display: inline-flex;
      gap: 8px;
      margin: 4px 0 8px;
    }}
    .curve-toggle button {{
      border: 1px solid #c7d4ea;
      background: #f7faff;
      color: #2d4675;
      border-radius: 999px;
      font-size: 12px;
      padding: 4px 12px;
      cursor: pointer;
    }}
    .curve-toggle button.active {{
      border-color: #1f63d8;
      background: #1f63d8;
      color: #fff;
    }}
    .curve-wrap svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .subcurve-wrap {{
      margin-top: 10px;
      border-top: 1px dashed var(--line);
      padding-top: 10px;
    }}
    .subcurve-title {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .subcurve-note {{
      margin: 0 0 6px;
      color: #64748b;
      font-size: 12px;
      line-height: 1.5;
    }}
    .legend {{
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 12px;
      color: var(--muted);
      padding: 18px;
      text-align: center;
      background: #fbfcff;
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
        <div class="sub">模板 {escape(result.strategy_template)} / T强度 {escape(result.intraday_t_level)} / 阈值 {result.effective_weight_threshold:.2f} / 持仓上限 {int(result.effective_max_positions)} / 日交易 {int(result.effective_max_trades_per_stock_per_day)} / 周交易 {int(result.effective_max_trades_per_stock_per_week)}</div>
      </article>
      <article class="card">
        <div class="k">大盘短期概率</div>
        <div class="v">{_pct(result.market_final_short)}</div>
        <div class="sub">模型 {_pct(market.short_prob)} / 新闻模型 {_pct(result.market_news_short_prob)} / {escape(result.market_fusion_mode_short)}</div>
      </article>
      <article class="card">
        <div class="k">大盘中期概率</div>
        <div class="v">{_pct(result.market_final_mid)}</div>
        <div class="sub">模型 {_pct(market.mid_prob)} / 新闻模型 {_pct(result.market_news_mid_prob)} / {escape(result.market_fusion_mode_mid)}</div>
      </article>
      <article class="card">
        <div class="k">风险温度</div>
        <div class="v">{escape(eff.risk_label)}</div>
        <div class="sub">亏钱效应与回撤共同决定</div>
      </article>
      <article class="card">
        <div class="k">验收状态</div>
        <div class="v">{acceptance_text}</div>
        <div class="sub">{escape(acceptance_sub)}</div>
      </article>
    </section>

    <div class="section-grid">
      <section>
        <h2>个股综合排序 (前{len(stock_rows_display)} / 共{len(stock_rows)})</h2>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>个股</th><th>融合短期</th><th>融合中期</th><th>综合分数</th><th>分数强度</th><th>建议权重</th><th>短期方式</th><th>中期方式</th><th>量价风险</th></tr>
            </thead>
            <tbody>
              {''.join(stock_table_rows) if stock_table_rows else '<tr><td colspan="9">无数据</td></tr>'}
            </tbody>
          </table>
        </div>
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
      <h2>组合回测曲线 (交易级, 含成本)</h2>
      {curve_html}
    </section>

    <section>
      <h2>因子解释 (最新截面)</h2>
      <div class="table-scroll">
        <table>
          <thead>
            <tr><th>个股</th><th>短期驱动</th><th>中期驱动</th><th>风险备注</th></tr>
          </thead>
          <tbody>
            {''.join(driver_rows) if driver_rows else '<tr><td colspan="4">无数据</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>回测指标 (全样本 + 近3/5年)</h2>
      <table>
        <thead>
          <tr><th>窗口</th><th>开始</th><th>结束</th><th>交易日</th><th>策略总收益</th><th>年化收益</th><th>年化超额</th><th>最大回撤</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th><th>信息比率</th><th>日胜率</th><th>年化换手</th><th>成本损耗</th><th>单票周交易</th></tr>
        </thead>
        <tbody>
          {''.join(backtest_rows) if backtest_rows else '<tr><td colspan="16">无可用回测数据</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>策略优化 (收益目标 + 换手约束)</h2>
      <article class="card">
        <div class="k">目标函数</div>
        <div class="sub"><code>{escape(result.strategy_objective_text)}</code></div>
        <div class="sub" style="margin-top:6px;">评估窗口: {escape(result.strategy_target_metric_label)}</div>
        <div class="sub" style="margin-top:6px;">{escape(strategy_summary)}</div>
      </article>
      <table style="margin-top:10px;">
        <thead>
          <tr><th>排名</th><th>指标窗口</th><th>retrain(日)</th><th>阈值</th><th>持仓上限</th><th>市场新闻强度</th><th>个股新闻强度</th><th>目标得分</th><th>年化收益</th><th>年化超额</th><th>最大回撤</th><th>年化换手</th><th>成本损耗</th><th>单票周交易</th><th>Sharpe</th></tr>
        </thead>
        <tbody>
          {''.join(strategy_rows) if strategy_rows else '<tr><td colspan="15">无可用优化试验数据</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>新闻模糊矩阵 (个股前{len(stock_rows_display)})</h2>
      <div class="table-scroll">
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
      </div>
    </section>

    <section>
      <h2>学习型融合诊断</h2>
      <table>
        <thead>
          <tr><th>标的</th><th>周期</th><th>模式</th><th>原因</th><th>样本</th><th>验证样本</th><th>验证准确率</th><th>验证AUC</th><th>验证Brier</th><th>新闻系数</th><th>融合系数(quant)</th><th>融合系数(news)</th></tr>
        </thead>
        <tbody>
          {''.join(learning_rows) if learning_rows else '<tr><td colspan="12">无数据</td></tr>'}
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
