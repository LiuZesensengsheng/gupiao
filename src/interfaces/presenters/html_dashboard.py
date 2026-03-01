from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.application.use_cases import DailyFusionResult
from src.application.v2_contracts import (
    DailyRunResult as V2DailyRunResult,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
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


def _money(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:,.0f}"


def _int(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return str(int(round(float(v))))


def _v2_cn_action(value: str) -> str:
    mapping = {
        "BUY": "买入",
        "SELL": "卖出",
        "HOLD": "持有",
    }
    return mapping.get(str(value), str(value))


def _v2_cn_strategy_mode(value: str) -> str:
    mapping = {
        "trend_follow": "趋势跟随",
        "range_rotation": "震荡轮动",
        "defensive": "防守",
    }
    return mapping.get(str(value), str(value))


def _v2_cn_risk_regime(value: str) -> str:
    mapping = {
        "risk_on": "积极",
        "cautious": "谨慎",
        "risk_off": "收缩",
    }
    return mapping.get(str(value), str(value))


def _v2_cn_trend_state(value: str) -> str:
    mapping = {
        "trend": "趋势",
        "range": "震荡",
        "risk_off": "风险收缩",
    }
    return mapping.get(str(value), str(value))


def _v2_cn_volatility(value: str) -> str:
    mapping = {
        "high": "高波动",
        "normal": "正常",
        "low": "低波动",
    }
    return mapping.get(str(value), str(value))


def _v2_cn_policy_field(value: str) -> str:
    mapping = {
        "risk_on_exposure": "积极状态总仓位",
        "cautious_exposure": "谨慎状态总仓位",
        "risk_off_exposure": "收缩状态总仓位",
        "risk_on_positions": "积极状态持仓数",
        "cautious_positions": "谨慎状态持仓数",
        "risk_off_positions": "收缩状态持仓数",
        "risk_on_turnover_cap": "积极状态换手上限",
        "cautious_turnover_cap": "谨慎状态换手上限",
        "risk_off_turnover_cap": "收缩状态换手上限",
        "baseline": "基线",
        "calibrated": "校准后",
    }
    return mapping.get(str(value), str(value))


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


def _mini_gauge_html(
    value: float,
    *,
    label: str,
    color: str = "#b4472f",
    width: float = 220.0,
    height: float = 130.0,
) -> str:
    safe_value = float(np.clip(0.0 if pd.isna(value) else float(value), 0.0, 1.0))
    cx = width / 2.0
    cy = height - 14.0
    radius = min(width * 0.36, height * 0.78)

    def _polar(ratio: float) -> tuple[float, float]:
        angle = np.pi * (1.0 - ratio)
        return cx + radius * float(np.cos(angle)), cy - radius * float(np.sin(angle))

    sx, sy = _polar(0.0)
    ex, ey = _polar(1.0)
    vx, vy = _polar(safe_value)
    return (
        "<div class='gauge-card'>"
        f"<div class='gauge-label'>{escape(label)}</div>"
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' class='gauge' role='img' aria-label='{escape(label)}'>"
        f"<path d='M {sx:.1f} {sy:.1f} A {radius:.1f} {radius:.1f} 0 0 1 {ex:.1f} {ey:.1f}' "
        "fill='none' stroke='#e7ddcf' stroke-width='14' stroke-linecap='round'/>"
        f"<path d='M {sx:.1f} {sy:.1f} A {radius:.1f} {radius:.1f} 0 0 1 {vx:.1f} {vy:.1f}' "
        f"fill='none' stroke='{color}' stroke-width='14' stroke-linecap='round'/>"
        f"<circle cx='{vx:.1f}' cy='{vy:.1f}' r='5' fill='{color}'/>"
        f"<text x='{cx:.1f}' y='{(cy - radius * 0.25):.1f}' text-anchor='middle' font-size='30' font-weight='700' fill='#1e2a2f'>{_pct(safe_value)}</text>"
        f"<text x='{(cx - radius):.1f}' y='{(cy + 8):.1f}' text-anchor='middle' font-size='11' fill='#8a7664'>0%</text>"
        f"<text x='{(cx + radius):.1f}' y='{(cy + 8):.1f}' text-anchor='middle' font-size='11' fill='#8a7664'>100%</text>"
        "</svg>"
        "</div>"
    )


def _v2_hbar_chart_html(
    items: list[tuple[str, float]],
    *,
    title: str,
    color: str = "#b4472f",
    formatter: Callable[[float], str] | None = None,
    empty_text: str = "无数据",
) -> str:
    if not items:
        return f"<div class='empty'>{escape(empty_text)}</div>"

    formatter = formatter or _pct
    max_value = max(max(0.0, float(value)) for _, value in items)
    max_value = max(max_value, 1e-9)
    rows = []
    for label, value in items:
        safe = max(0.0, float(value))
        width = 14.0 + 86.0 * safe / max_value
        rows.append(
            "<div class='hbar-row'>"
            f"<div class='hbar-label'>{escape(label)}</div>"
            "<div class='hbar-track'>"
            f"<div class='hbar-fill' style='width:{width:.1f}%; background:{color};'></div>"
            "</div>"
            f"<div class='hbar-value'>{escape(formatter(safe))}</div>"
            "</div>"
        )
    return (
        "<div class='viz-card'>"
        f"<div class='viz-title'>{escape(title)}</div>"
        f"{''.join(rows)}"
        "</div>"
    )


def _v2_action_donut_html(actions: list[object]) -> str:
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for action in actions:
        counts[str(getattr(action, "action", "HOLD"))] = counts.get(str(getattr(action, "action", "HOLD")), 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return "<div class='empty'>无交易动作</div>"

    palette = {"BUY": "#b13232", "SELL": "#2f66c8", "HOLD": "#7a8a6a"}
    circumference = 2.0 * np.pi * 52.0
    offset = 0.0
    segments = []
    legends = []
    for name in ("BUY", "SELL", "HOLD"):
        ratio = counts[name] / float(total)
        dash = circumference * ratio
        segments.append(
            f"<circle cx='70' cy='70' r='52' fill='none' stroke='{palette[name]}' stroke-width='16' "
            f"stroke-dasharray='{dash:.2f} {max(0.0, circumference - dash):.2f}' "
            f"stroke-dashoffset='{-offset:.2f}' transform='rotate(-90 70 70)' stroke-linecap='butt'/>"
        )
        offset += dash
        legends.append(
            "<div class='donut-legend-row'>"
            f"<span><i style='background:{palette[name]};'></i>{escape(_v2_cn_action(name))}</span>"
            f"<strong>{counts[name]}</strong>"
            "</div>"
        )
    return (
        "<div class='viz-card donut-wrap'>"
        "<div class='viz-title'>交易动作分布</div>"
        "<div class='donut-layout'>"
        "<svg viewBox='0 0 140 140' class='donut' role='img' aria-label='交易动作分布'>"
        "<circle cx='70' cy='70' r='52' fill='none' stroke='#eee4d7' stroke-width='16'/>"
        f"{''.join(segments)}"
        f"<text x='70' y='66' text-anchor='middle' font-size='24' font-weight='700' fill='#1e2a2f'>{total}</text>"
        "<text x='70' y='85' text-anchor='middle' font-size='11' fill='#6c756f'>动作</text>"
        "</svg>"
        f"<div class='donut-legend'>{''.join(legends)}</div>"
        "</div>"
        "</div>"
    )


def _v2_compare_bars_html(
    rows: list[tuple[str, float, float, bool]],
    *,
    left_label: str,
    right_label: str,
) -> str:
    if not rows:
        return "<div class='empty'>无可对比指标</div>"

    scale = max(max(abs(float(a)), abs(float(b))) for _, a, b, _ in rows)
    scale = max(scale, 1e-9)
    blocks = []
    for label, left, right, lower_is_better in rows:
        left_width = 8.0 + 92.0 * abs(float(left)) / scale
        right_width = 8.0 + 92.0 * abs(float(right)) / scale
        delta = float(right) - float(left)
        improved = delta < 0 if lower_is_better else delta > 0
        delta_text = f"{'改善' if improved else '变化'} {_pct(abs(delta))}"
        blocks.append(
            "<div class='compare-row'>"
            f"<div class='compare-label'>{escape(label)}</div>"
            "<div class='compare-bars'>"
            f"<div class='compare-bar left'><span style='width:{left_width:.1f}%;'></span><em>{escape(_v2_cn_policy_field(left_label))} {escape(_pct(left))}</em></div>"
            f"<div class='compare-bar right'><span style='width:{right_width:.1f}%;'></span><em>{escape(_v2_cn_policy_field(right_label))} {escape(_pct(right))}</em></div>"
            "</div>"
            f"<div class='compare-delta {'up' if improved else 'down'}'>{escape(delta_text)}</div>"
            "</div>"
        )
    return "<div class='viz-card compare-wrap'>" + "".join(blocks) + "</div>"


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

    trade_rows_html: list[str] = []
    buy_value = 0.0
    sell_value = 0.0
    buy_count = 0
    sell_count = 0
    hold_count = 0
    for action in result.trade_actions:
        delta_value = float(action.est_delta_value)
        if action.action == "BUY":
            buy_count += 1
            if not pd.isna(delta_value) and delta_value > 0:
                buy_value += delta_value
        elif action.action == "SELL":
            sell_count += 1
            if not pd.isna(delta_value) and delta_value < 0:
                sell_value += abs(delta_value)
        else:
            hold_count += 1
        action_color = "#cf3131" if action.action == "BUY" else ("#1f63d8" if action.action == "SELL" else "#6b7280")
        trade_rows_html.append(
            "<tr>"
            f"<td><div class='sym'>{escape(action.name)}</div><div class='code'>{escape(action.symbol)}</div></td>"
            f"<td style='font-weight:700;color:{action_color};'>{escape(action.action)}</td>"
            f"<td>{_pct(action.current_weight)}</td>"
            f"<td>{_pct(action.target_weight)}</td>"
            f"<td class='score' style='color:{_score_color(action.delta_weight)}'>{_pct(action.delta_weight)}</td>"
            f"<td>{_num(action.est_price, 3)}</td>"
            f"<td>{_money(action.est_delta_value)}</td>"
            f"<td>{_int(action.est_delta_shares)}</td>"
            f"<td>{_num(action.est_delta_lots, 2)}</td>"
            f"<td>{escape(action.note or 'NA')}</td>"
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
    nav_text = _money(result.trade_plan_nav)
    trade_basis_text = escape(result.trade_plan_basis)

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
      <article class="card">
        <div class="k">买入预算(估算)</div>
        <div class="v">{_money(buy_value)}</div>
        <div class="sub">BUY {buy_count} / HOLD {hold_count} / 口径 {trade_basis_text}</div>
      </article>
      <article class="card">
        <div class="k">卖出规模(估算)</div>
        <div class="v">{_money(sell_value)}</div>
        <div class="sub">SELL {sell_count} / 组合净值 {nav_text if nav_text != 'NA' else 'NA'} / 每手 {int(result.trade_plan_lot_size)} 股</div>
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
      <h2>调仓执行建议</h2>
      <article class="card">
        <div class="k">计算口径</div>
        <div class="sub">基础 {trade_basis_text} / 组合净值 {nav_text if nav_text != 'NA' else 'NA(仅输出权重差)'} / 每手 {int(result.trade_plan_lot_size)} 股 / BUY {buy_count} / SELL {sell_count} / HOLD {hold_count}</div>
      </article>
      <div class="table-scroll" style="margin-top:10px;">
        <table>
          <thead>
            <tr><th>个股</th><th>动作</th><th>当前权重</th><th>目标权重</th><th>权重变化</th><th>参考价格</th><th>估算金额</th><th>估算股数</th><th>估算手数</th><th>备注</th></tr>
          </thead>
          <tbody>
            {''.join(trade_rows_html) if trade_rows_html else '<tr><td colspan="10">无可用调仓数据</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>

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


def write_v2_daily_dashboard(out_path: str | Path, result: V2DailyRunResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sector_rows = []
    for sector in result.composite_state.sectors:
        sector_rows.append(
            "<tr>"
            f"<td>{escape(sector.sector)}</td>"
            f"<td>{_pct(sector.up_5d_prob)}</td>"
            f"<td>{_pct(sector.up_20d_prob)}</td>"
            f"<td class='score' style='color:{_score_color(sector.relative_strength)}'>{_score(sector.relative_strength)}</td>"
            f"<td>{_num(sector.rotation_speed, 3)}</td>"
            f"<td>{_num(sector.crowding_score, 3)}</td>"
            f"<td>{_pct(result.policy_decision.sector_budgets.get(sector.sector, 0.0))}</td>"
            "</tr>"
        )

    stock_rows = []
    for stock in result.composite_state.stocks:
        stock_rows.append(
            "<tr>"
            f"<td><div class='sym'>{escape(stock.symbol)}</div><div class='code'>{escape(stock.sector)}</div></td>"
            f"<td>{_pct(stock.up_1d_prob)}</td>"
            f"<td>{_pct(stock.up_5d_prob)}</td>"
            f"<td>{_pct(stock.up_20d_prob)}</td>"
            f"<td>{_pct(stock.excess_vs_sector_prob)}</td>"
            f"<td>{_pct(stock.tradeability_score)}</td>"
            f"<td>{_pct(result.policy_decision.symbol_target_weights.get(stock.symbol, 0.0))}</td>"
            "</tr>"
        )

    trade_rows = []
    for action in result.trade_actions:
        action_color = "#cf3131" if action.action == "BUY" else ("#1f63d8" if action.action == "SELL" else "#6b7280")
        trade_rows.append(
            "<tr>"
            f"<td>{escape(action.symbol)}</td>"
            f"<td style='font-weight:700;color:{action_color};'>{escape(_v2_cn_action(action.action))}</td>"
            f"<td>{_pct(action.current_weight)}</td>"
            f"<td>{_pct(action.target_weight)}</td>"
            f"<td class='score' style='color:{_score_color(action.delta_weight)}'>{_pct(action.delta_weight)}</td>"
            f"<td>{escape(action.note or 'NA')}</td>"
            "</tr>"
        )

    risk_notes = "".join(f"<li>{escape(note)}</li>" for note in result.policy_decision.risk_notes) or "<li>无</li>"
    sector_chart = _v2_hbar_chart_html(
        [
            (sector.sector, float(result.policy_decision.sector_budgets.get(sector.sector, 0.0)))
            for sector in result.composite_state.sectors
            if float(result.policy_decision.sector_budgets.get(sector.sector, 0.0)) > 0.0
        ],
        title="板块预算分布",
        color="#b4472f",
    )
    stock_chart = _v2_hbar_chart_html(
        sorted(
            result.policy_decision.symbol_target_weights.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        ),
        title="目标持仓权重",
        color="#1f6b72",
    )
    market_gauges = (
        _mini_gauge_html(result.composite_state.market.up_20d_prob, label="20日上涨概率", color="#b4472f")
        + _mini_gauge_html(1.0 - result.composite_state.market.drawdown_risk, label="抗回撤强度", color="#1f6b72")
        + _mini_gauge_html(1.0 - result.composite_state.market.liquidity_stress, label="流动性健康度", color="#7a8a6a")
    )
    action_donut = _v2_action_donut_html(result.trade_actions)
    cross_rows = [
        ("大小盘偏好", float(abs(result.composite_state.cross_section.large_vs_small_bias))),
        ("成长价值偏好", float(abs(result.composite_state.cross_section.growth_vs_value_bias))),
        ("资金强度", float(max(0.0, result.composite_state.cross_section.fund_flow_strength))),
        ("宽度强度", float(max(0.0, result.composite_state.cross_section.breadth_strength))),
    ]
    cross_chart = _v2_hbar_chart_html(
        cross_rows,
        title="横截面强度",
        color="#3c6fd1",
        formatter=lambda value: f"{value:.3f}",
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 每日策略看板</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --paper: #fffaf2;
      --ink: #1e2a2f;
      --muted: #6c756f;
      --line: #d8cdbd;
      --accent: #b4472f;
      --accent-2: #1f6b72;
      --good: #b13232;
      --bad: #2f66c8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Avenir Next", "PingFang SC", sans-serif; color: var(--ink); background:
      radial-gradient(circle at top right, rgba(180,71,47,0.10), transparent 30%),
      radial-gradient(circle at 20% 20%, rgba(31,107,114,0.08), transparent 25%),
      var(--bg); }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg, #fff9f1, #f8efe1); border: 1px solid var(--line); border-radius: 24px; padding: 24px; box-shadow: 0 18px 40px rgba(78,55,24,0.08); }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 34px; letter-spacing: -0.02em; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    .sub {{ margin-top: 8px; color: var(--muted); }}
    .grid {{ display: grid; gap: 18px; margin-top: 18px; }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .grid.four {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 22px; padding: 18px; box-shadow: 0 10px 24px rgba(40,30,10,0.05); }}
    .metric {{ font-size: 30px; font-weight: 700; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .pill {{ display: inline-block; padding: 5px 10px; border-radius: 999px; background: rgba(180,71,47,0.10); color: var(--accent); font-size: 12px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #e8dece; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .score {{ font-weight: 700; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    .viz-grid {{ display: grid; gap: 18px; grid-template-columns: 1.3fr 1fr; margin-top: 18px; }}
    .viz-card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 22px; padding: 18px; box-shadow: 0 10px 24px rgba(40,30,10,0.05); }}
    .viz-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; }}
    .gauge-panel {{ display: grid; gap: 14px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .gauge-card {{ background: rgba(255,255,255,0.5); border: 1px solid #eadfce; border-radius: 18px; padding: 12px; }}
    .gauge-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }}
    .gauge {{ width: 100%; height: auto; display: block; }}
    .hbar-row {{ display: grid; grid-template-columns: 110px 1fr 64px; gap: 10px; align-items: center; margin-top: 10px; }}
    .hbar-row:first-of-type {{ margin-top: 0; }}
    .hbar-label, .hbar-value {{ font-size: 12px; color: var(--muted); }}
    .hbar-value {{ text-align: right; font-weight: 700; color: var(--ink); }}
    .hbar-track {{ height: 12px; border-radius: 999px; background: #efe5d7; overflow: hidden; }}
    .hbar-fill {{ height: 100%; border-radius: 999px; }}
    .donut-wrap {{ height: 100%; }}
    .donut-layout {{ display: grid; grid-template-columns: 160px 1fr; gap: 10px; align-items: center; }}
    .donut {{ width: 100%; height: auto; display: block; }}
    .donut-legend-row {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 0; border-bottom: 1px solid #ede3d5; font-size: 13px; }}
    .donut-legend-row:last-child {{ border-bottom: none; }}
    .donut-legend-row span {{ display: inline-flex; align-items: center; gap: 8px; }}
    .donut-legend-row i {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
    .empty {{ border: 1px dashed #ddcfbc; border-radius: 16px; color: var(--muted); padding: 18px; text-align: center; background: rgba(255,255,255,0.55); }}
    @media (max-width: 900px) {{
      .grid.two, .grid.three, .grid.four, .viz-grid, .gauge-panel, .donut-layout {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 14px; }}
      h1 {{ font-size: 28px; }}
      .hbar-row {{ grid-template-columns: 88px 1fr 56px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>V2 每日策略看板</h1>
      <div class="sub">策略 {escape(result.snapshot.strategy_id)} | 股票池 {escape(result.snapshot.universe_id)} | 日期 {escape(result.composite_state.market.as_of_date)}</div>
      <div class="grid three">
        <div>
          <div class="label">策略模式</div>
          <div class="metric">{escape(_v2_cn_strategy_mode(result.composite_state.strategy_mode))}</div>
        </div>
        <div>
          <div class="label">风险状态</div>
          <div class="metric">{escape(_v2_cn_risk_regime(result.composite_state.risk_regime))}</div>
        </div>
        <div>
          <div class="label">目标总仓位</div>
          <div class="metric">{_pct(result.policy_decision.target_exposure)}</div>
        </div>
      </div>
    </section>

    <section class="viz-grid">
      <div class="viz-card">
        <div class="viz-title">市场温度仪表</div>
        <div class="gauge-panel">{market_gauges}</div>
      </div>
      {action_donut}
    </section>

    <section class="viz-grid">
      {sector_chart}
      {stock_chart}
    </section>

    <section class="grid two">
      <div class="card">
        <h2>大盘状态</h2>
        <table>
          <tr><th>1日上涨概率</th><td>{_pct(result.composite_state.market.up_1d_prob)}</td></tr>
          <tr><th>5日上涨概率</th><td>{_pct(result.composite_state.market.up_5d_prob)}</td></tr>
          <tr><th>20日上涨概率</th><td>{_pct(result.composite_state.market.up_20d_prob)}</td></tr>
          <tr><th>趋势状态</th><td>{escape(_v2_cn_trend_state(result.composite_state.market.trend_state))}</td></tr>
          <tr><th>回撤风险</th><td>{_pct(result.composite_state.market.drawdown_risk)}</td></tr>
          <tr><th>波动状态</th><td>{escape(_v2_cn_volatility(result.composite_state.market.volatility_regime))}</td></tr>
          <tr><th>流动性压力</th><td>{_pct(result.composite_state.market.liquidity_stress)}</td></tr>
        </table>
      </div>
      <div class="card">
        <h2>横截面状态</h2>
        <table>
          <tr><th>大小盘偏好</th><td>{_num(result.composite_state.cross_section.large_vs_small_bias, 3)}</td></tr>
          <tr><th>成长价值偏好</th><td>{_num(result.composite_state.cross_section.growth_vs_value_bias, 3)}</td></tr>
          <tr><th>资金强度</th><td>{_num(result.composite_state.cross_section.fund_flow_strength, 3)}</td></tr>
          <tr><th>两融风险偏好</th><td>{_num(result.composite_state.cross_section.margin_risk_on_score, 3)}</td></tr>
          <tr><th>宽度强度</th><td>{_num(result.composite_state.cross_section.breadth_strength, 3)}</td></tr>
          <tr><th>龙头参与率</th><td>{_pct(result.composite_state.cross_section.leader_participation)}</td></tr>
          <tr><th>弱势股比例</th><td>{_pct(result.composite_state.cross_section.weak_stock_ratio)}</td></tr>
        </table>
        <div style="margin-top:14px;">{cross_chart}</div>
      </div>
    </section>

    <section class="grid two">
      <div class="card">
        <h2>板块预算</h2>
        <table>
          <thead><tr><th>板块</th><th>5日</th><th>20日</th><th>强度</th><th>轮动</th><th>拥挤</th><th>预算</th></tr></thead>
          <tbody>{''.join(sector_rows) or '<tr><td colspan="7">无数据</td></tr>'}</tbody>
        </table>
      </div>
      <div class="card">
        <h2>个股目标仓位</h2>
        <table>
          <thead><tr><th>股票</th><th>1日</th><th>5日</th><th>20日</th><th>板块内超额</th><th>交易性</th><th>目标</th></tr></thead>
          <tbody>{''.join(stock_rows) or '<tr><td colspan="7">无数据</td></tr>'}</tbody>
        </table>
      </div>
    </section>

    <section class="grid two">
      <div class="card">
        <h2>策略决策</h2>
        <div class="pill">持仓数 {result.policy_decision.target_position_count}</div>
        <table>
          <tr><th>是否调仓</th><td>{'是' if result.policy_decision.rebalance_now else '否'}</td></tr>
          <tr><th>调仓强度</th><td>{_pct(result.policy_decision.rebalance_intensity)}</td></tr>
          <tr><th>日内T</th><td>{'允许' if result.policy_decision.intraday_t_allowed else '不允许'}</td></tr>
          <tr><th>换手上限</th><td>{_pct(result.policy_decision.turnover_cap)}</td></tr>
        </table>
        <div class="label" style="margin-top:12px;">风险备注</div>
        <ul>{risk_notes}</ul>
      </div>
      <div class="card">
        <h2>交易计划</h2>
        <table>
          <thead><tr><th>股票</th><th>动作</th><th>当前</th><th>目标</th><th>变化</th><th>备注</th></tr></thead>
          <tbody>{''.join(trade_rows) or '<tr><td colspan="6">无数据</td></tr>'}</tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


def write_v2_research_dashboard(
    out_path: str | Path,
    *,
    strategy_id: str,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    artifacts: dict[str, str] | None = None,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _metric_tile(label: str, value: str, tone: str = "#b4472f") -> str:
        return (
            "<div class='tile'>"
            f"<div class='label'>{escape(label)}</div>"
            f"<div class='value' style='color:{tone};'>{escape(value)}</div>"
            "</div>"
        )

    compare_chart = _v2_compare_bars_html(
        [
            ("年化收益", float(baseline.annual_return), float(calibration.calibrated.annual_return), False),
            ("总收益", float(baseline.total_return), float(calibration.calibrated.total_return), False),
            ("最大回撤", abs(float(baseline.max_drawdown)), abs(float(calibration.calibrated.max_drawdown)), True),
            ("平均换手", float(baseline.avg_turnover), float(calibration.calibrated.avg_turnover), True),
            ("总成本", float(baseline.total_cost), float(calibration.calibrated.total_cost), True),
        ],
        left_label="baseline",
        right_label="calibrated",
    )
    improvement_tiles = "".join(
        [
            _metric_tile(
                "年化增量",
                _pct(float(calibration.calibrated.annual_return) - float(baseline.annual_return)),
                "#185e66" if float(calibration.calibrated.annual_return) >= float(baseline.annual_return) else "#c14d2d",
            ),
            _metric_tile(
                "收益增量",
                _pct(float(calibration.calibrated.total_return) - float(baseline.total_return)),
                "#185e66" if float(calibration.calibrated.total_return) >= float(baseline.total_return) else "#c14d2d",
            ),
            _metric_tile(
                "换手变化",
                _pct(float(calibration.calibrated.avg_turnover) - float(baseline.avg_turnover)),
                "#c14d2d" if float(calibration.calibrated.avg_turnover) > float(baseline.avg_turnover) else "#185e66",
            ),
            _metric_tile(
                "成本变化",
                _pct(float(calibration.calibrated.total_cost) - float(baseline.total_cost)),
                "#c14d2d" if float(calibration.calibrated.total_cost) > float(baseline.total_cost) else "#185e66",
            ),
            _metric_tile(
                "学习年化",
                _pct(float(learning.learned.annual_return)),
                "#185e66" if float(learning.learned.annual_return) >= float(baseline.annual_return) else "#c14d2d",
            ),
        ]
    )
    artifact_rows = "".join(
        f"<tr><th>{escape(str(label))}</th><td>{escape(str(path_value))}</td></tr>"
        for label, path_value in (artifacts or {}).items()
    )
    horizon_rows = ""
    for label, summary in [
        ("基线", baseline),
        ("校准", calibration.calibrated),
        ("学习", learning.learned),
    ]:
        for horizon in ["1d", "5d", "20d"]:
            metrics = summary.horizon_metrics.get(horizon, {})
            if not metrics:
                continue
            horizon_rows += (
                f"<tr><td>{escape(label)}</td><td>{escape(horizon)}</td>"
                f"<td>{_num(float(metrics.get('rank_ic', 0.0)), 3)}</td>"
                f"<td>{_pct(float(metrics.get('top_decile_return', 0.0)))}</td>"
                f"<td>{_pct(float(metrics.get('top_bottom_spread', 0.0)))}</td>"
                f"<td>{_pct(float(metrics.get('top_k_hit_rate', 0.0)))}</td></tr>"
            )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 研究回测看板</title>
  <style>
    :root {{
      --bg: #eef3f1;
      --paper: #ffffff;
      --ink: #18242a;
      --muted: #60717a;
      --line: #d7e0e5;
      --accent: #185e66;
      --accent-2: #c14d2d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Avenir Next", "PingFang SC", sans-serif; color: var(--ink); background:
      linear-gradient(180deg, #f7fbfc 0%, #eef3f1 100%); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero, .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 22px; box-shadow: 0 14px 34px rgba(23,45,52,0.08); }}
    .hero {{ padding: 24px; }}
    .card {{ padding: 18px; }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 34px; letter-spacing: -0.02em; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; }}
    .sub {{ margin-top: 8px; color: var(--muted); }}
    .grid {{ display: grid; gap: 18px; margin-top: 18px; }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid.four {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .tile {{ padding: 14px; border-radius: 18px; background: linear-gradient(135deg, #f9fcfd, #f1f7f8); border: 1px solid #e4edf0; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .value {{ margin-top: 6px; font-size: 28px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #e8eef0; text-align: left; font-size: 13px; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .viz-card {{ padding: 18px; border-radius: 22px; background: linear-gradient(180deg, #fbfefe, #f3f7f8); border: 1px solid #e2ebef; }}
    .compare-wrap {{ display: grid; gap: 14px; }}
    .compare-row {{ display: grid; gap: 10px; grid-template-columns: 96px 1fr 92px; align-items: center; }}
    .compare-label {{ font-size: 12px; color: var(--muted); font-weight: 700; }}
    .compare-bars {{ display: grid; gap: 8px; }}
    .compare-bar {{ position: relative; height: 26px; border-radius: 999px; overflow: hidden; background: #e8eef0; }}
    .compare-bar span {{ display: block; height: 100%; }}
    .compare-bar.left span {{ background: #185e66; }}
    .compare-bar.right span {{ background: #c14d2d; }}
    .compare-bar em {{ position: absolute; inset: 0; display: flex; align-items: center; padding: 0 10px; color: #102228; font-size: 12px; font-style: normal; font-weight: 700; }}
    .compare-delta {{ text-align: right; font-size: 12px; font-weight: 700; }}
    .compare-delta.up {{ color: #185e66; }}
    .compare-delta.down {{ color: #c14d2d; }}
    @media (max-width: 900px) {{
      .grid.two, .grid.four {{ grid-template-columns: 1fr; }}
      .compare-row {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 14px; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>V2 研究回测看板</h1>
      <div class="sub">策略 {escape(strategy_id)} | 基线回测与策略校准结果</div>
      <div class="grid four">
        {_metric_tile("基线年化", _pct(baseline.annual_return), "#185e66")}
        {_metric_tile("基线回撤", _pct(baseline.max_drawdown), "#c14d2d")}
        {_metric_tile("最优评分", _num(calibration.best_score, 4), "#185e66")}
        {_metric_tile("校准年化", _pct(calibration.calibrated.annual_return), "#c14d2d")}
      </div>
    </section>

    <section class="grid">
      <div class="viz-card">
        <h2>基线方案 vs 校准方案</h2>
        {compare_chart}
      </div>
    </section>

    <section class="grid four">
      {improvement_tiles}
    </section>

    <section class="grid two">
      <div class="card">
        <h2>回测结果</h2>
        <table>
          <thead><tr><th>方案</th><th>开始</th><th>结束</th><th>交易日</th><th>总收益</th><th>年化</th><th>回撤</th><th>换手</th><th>平均RankIC</th><th>头尾价差</th><th>TopK命中率</th><th>成本</th></tr></thead>
          <tbody>
            <tr><td>基线方案</td><td>{escape(baseline.start_date or 'NA')}</td><td>{escape(baseline.end_date or 'NA')}</td><td>{baseline.n_days}</td><td>{_pct(baseline.total_return)}</td><td>{_pct(baseline.annual_return)}</td><td>{_pct(baseline.max_drawdown)}</td><td>{_pct(baseline.avg_turnover)}</td><td>{_num(baseline.avg_rank_ic, 3)}</td><td>{_pct(baseline.avg_top_bottom_spread)}</td><td>{_pct(baseline.avg_top_k_hit_rate)}</td><td>{_pct(baseline.total_cost)}</td></tr>
            <tr><td>校准方案</td><td>{escape(calibration.calibrated.start_date or 'NA')}</td><td>{escape(calibration.calibrated.end_date or 'NA')}</td><td>{calibration.calibrated.n_days}</td><td>{_pct(calibration.calibrated.total_return)}</td><td>{_pct(calibration.calibrated.annual_return)}</td><td>{_pct(calibration.calibrated.max_drawdown)}</td><td>{_pct(calibration.calibrated.avg_turnover)}</td><td>{_num(calibration.calibrated.avg_rank_ic, 3)}</td><td>{_pct(calibration.calibrated.avg_top_bottom_spread)}</td><td>{_pct(calibration.calibrated.avg_top_k_hit_rate)}</td><td>{_pct(calibration.calibrated.total_cost)}</td></tr>
            <tr><td>学习方案</td><td>{escape(learning.learned.start_date or 'NA')}</td><td>{escape(learning.learned.end_date or 'NA')}</td><td>{learning.learned.n_days}</td><td>{_pct(learning.learned.total_return)}</td><td>{_pct(learning.learned.annual_return)}</td><td>{_pct(learning.learned.max_drawdown)}</td><td>{_pct(learning.learned.avg_turnover)}</td><td>{_num(learning.learned.avg_rank_ic, 3)}</td><td>{_pct(learning.learned.avg_top_bottom_spread)}</td><td>{_pct(learning.learned.avg_top_k_hit_rate)}</td><td>{_pct(learning.learned.total_cost)}</td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <h2>最优策略参数</h2>
        <table>
          <tr><th>{escape(_v2_cn_policy_field('risk_on_exposure'))}</th><td>{_pct(calibration.best_policy.risk_on_exposure)}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('cautious_exposure'))}</th><td>{_pct(calibration.best_policy.cautious_exposure)}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('risk_off_exposure'))}</th><td>{_pct(calibration.best_policy.risk_off_exposure)}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('risk_on_positions'))}</th><td>{calibration.best_policy.risk_on_positions}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('cautious_positions'))}</th><td>{calibration.best_policy.cautious_positions}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('risk_off_positions'))}</th><td>{calibration.best_policy.risk_off_positions}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('risk_on_turnover_cap'))}</th><td>{_pct(calibration.best_policy.risk_on_turnover_cap)}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('cautious_turnover_cap'))}</th><td>{_pct(calibration.best_policy.cautious_turnover_cap)}</td></tr>
          <tr><th>{escape(_v2_cn_policy_field('risk_off_turnover_cap'))}</th><td>{_pct(calibration.best_policy.risk_off_turnover_cap)}</td></tr>
        </table>
      </div>
    </section>

    <section class="grid two">
      <div class="card">
        <h2>学习型策略诊断</h2>
        <table>
          <tr><th>训练样本数</th><td>{learning.model.train_rows}</td></tr>
          <tr><th>仓位拟合R²</th><td>{_num(learning.model.train_r2_exposure, 4)}</td></tr>
          <tr><th>持仓数拟合R²</th><td>{_num(learning.model.train_r2_positions, 4)}</td></tr>
          <tr><th>换手上限拟合R²</th><td>{_num(learning.model.train_r2_turnover, 4)}</td></tr>
          <tr><th>学习方案平均成交率</th><td>{_pct(learning.learned.avg_fill_ratio)}</td></tr>
          <tr><th>学习方案平均滑点</th><td>{_num(learning.learned.avg_slippage_bps, 1)}bp</td></tr>
          <tr><th>学习方案平均RankIC</th><td>{_num(learning.learned.avg_rank_ic, 3)}</td></tr>
          <tr><th>学习方案TopK命中率</th><td>{_pct(learning.learned.avg_top_k_hit_rate)}</td></tr>
        </table>
      </div>
      <div class="card">
        <h2>研究产物</h2>
        <table>
          <tbody>{artifact_rows or '<tr><td colspan="2">本次未写出产物</td></tr>'}</tbody>
        </table>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>多周期横截面分层指标</h2>
        <table>
          <thead><tr><th>方案</th><th>周期</th><th>RankIC</th><th>头部分层收益</th><th>头尾价差</th><th>TopK命中率</th></tr></thead>
          <tbody>{horizon_rows or '<tr><td colspan="6">暂无多周期指标</td></tr>'}</tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path
