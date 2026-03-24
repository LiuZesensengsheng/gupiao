from __future__ import annotations

from html import escape
from pathlib import Path

from src.application.v2_info_research_runtime import V2InfoResearchResult


def _fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number * 100.0:.2f}%"


def _fmt_num(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}"


def _variant_table_lines(title: str, variants: dict[str, dict[str, object]], horizons: tuple[str, ...]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| variant | items | best_horizon | best_rank_ic | event_day_hit(info) | event_day_hit(quant) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in variants.items():
        lines.append(
            f"| {name} | {int(payload.get('item_count', 0))} | {payload.get('best_horizon', '')} | "
            f"{_fmt_num(payload.get('best_horizon_rank_ic', 0.0))} | "
            f"{_fmt_pct(payload.get('event_day_hit_rate_shadow', 0.0))} | "
            f"{_fmt_pct(payload.get('event_day_hit_rate_quant', 0.0))} |"
        )
        metrics = dict(payload.get("horizon_metrics", {}))
        if metrics:
            lines.append("")
            lines.append("| horizon | available_rows | shadow_ic | quant_ic | uplift_ic | shadow_spread | uplift_spread | shadow_hit | uplift_hit |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for horizon in horizons:
                row = dict(metrics.get(horizon, {}))
                lines.append(
                    f"| {horizon} | {int(row.get('available_rows', 0))} | {_fmt_num(row.get('shadow_rank_ic', 0.0))} | "
                    f"{_fmt_num(row.get('quant_rank_ic', 0.0))} | {_fmt_num(row.get('uplift_rank_ic', 0.0))} | "
                    f"{_fmt_pct(row.get('shadow_top_bottom_spread', 0.0))} | {_fmt_pct(row.get('uplift_top_bottom_spread', 0.0))} | "
                    f"{_fmt_pct(row.get('shadow_top_k_hit_rate', 0.0))} | {_fmt_pct(row.get('uplift_top_k_hit_rate', 0.0))} |"
                )
            lines.append("")
    return lines


def write_v2_info_research_report(out_path: str | Path, result: V2InfoResearchResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(result.info_manifest)

    lines = [
        "# V2 Info Research Report",
        "",
        f"- strategy: `{result.strategy_id}`",
        f"- latest_state_date: `{result.latest_state_date}`",
        f"- forecast_backend: `{result.forecast_backend}`",
        f"- retrain_days: `{result.retrain_days}`",
        f"- training_window_days: `{result.training_window_days}`",
        f"- split: `{result.split_mode}` | embargo=`{result.embargo_days}d` | fit_scope=`{result.fit_scope}` | evaluation_scope=`{result.evaluation_scope}`",
        f"- trajectory_steps: `{result.trajectory_steps}` | fit_steps: `{result.fit_steps}` | evaluation_steps: `{result.evaluation_steps}`",
        f"- evaluation_window: `{result.evaluation_window_start}` -> `{result.evaluation_window_end}`",
        f"- info_file: `{result.info_file}`",
        f"- info_item_count: `{manifest.get('info_item_count', 0)}` | publish_timestamp_coverage: `{_fmt_pct(manifest.get('publish_timestamp_coverage_ratio', 0.0))}`",
        f"- date_window: `{dict(manifest.get('date_window', {})).get('start', '')}` -> `{dict(manifest.get('date_window', {})).get('end', '')}`",
        "",
        "## Info Breakdown",
        "",
        f"- sources: `{manifest.get('info_source_breakdown', {})}`",
        f"- event_tags: `{manifest.get('event_tag_distribution', {})}`",
        "",
    ]
    lines.extend(_variant_table_lines("Source Variants", result.source_variants, result.horizons))
    lines.append("")
    lines.extend(_variant_table_lines("Timestamp Variants", result.timestamp_variants, result.horizons))
    lines.extend(
        [
            "",
            "## Tag Variants",
            "",
            "| event_tag | items | best_horizon | best_rank_ic | best_spread |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for payload in result.tag_variants:
        lines.append(
            f"| {payload.get('event_tag', '')} | {int(payload.get('item_count', 0))} | "
            f"{payload.get('best_horizon', '')} | {_fmt_num(payload.get('best_horizon_rank_ic', 0.0))} | "
            f"{_fmt_pct(payload.get('best_horizon_top_bottom_spread', 0.0))} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_v2_info_research_dashboard(out_path: str | Path, result: V2InfoResearchResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(result.info_manifest)
    all_info = dict(result.source_variants.get("all_info", {}))
    all_info_metrics = dict(all_info.get("horizon_metrics", {}))
    metric_rows = []
    for horizon in result.horizons:
        payload = dict(all_info_metrics.get(horizon, {}))
        metric_rows.append(
            "<tr>"
            f"<td>{escape(horizon)}</td>"
            f"<td>{int(payload.get('available_rows', 0))}</td>"
            f"<td>{escape(_fmt_num(payload.get('shadow_rank_ic', 0.0)))}</td>"
            f"<td>{escape(_fmt_num(payload.get('quant_rank_ic', 0.0)))}</td>"
            f"<td>{escape(_fmt_num(payload.get('uplift_rank_ic', 0.0)))}</td>"
            f"<td>{escape(_fmt_pct(payload.get('shadow_top_bottom_spread', 0.0)))}</td>"
            f"<td>{escape(_fmt_pct(payload.get('uplift_top_bottom_spread', 0.0)))}</td>"
            "</tr>"
        )
    tag_rows = []
    for payload in result.tag_variants[:12]:
        tag_rows.append(
            "<tr>"
            f"<td>{escape(str(payload.get('event_tag', '')))}</td>"
            f"<td>{int(payload.get('item_count', 0))}</td>"
            f"<td>{escape(str(payload.get('best_horizon', '')))}</td>"
            f"<td>{escape(_fmt_num(payload.get('best_horizon_rank_ic', 0.0)))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 Info Research</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #23180e;
      --muted: #7d6a58;
      --accent: #9b4d2f;
      --line: #e2d5c5;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: linear-gradient(180deg, #fff6ea 0%, var(--bg) 50%, #ede2d3 100%);
      color: var(--ink);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 48px; }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 14px 34px rgba(66, 38, 13, 0.06);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 4px 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .card {{
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .value {{ margin-top: 8px; font-size: 26px; font-weight: 700; }}
    .section {{ margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 14px; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .mono {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>V2 Info Research</h1>
      <p>strategy=<span class="mono">{escape(result.strategy_id)}</span> | as_of=<span class="mono">{escape(result.latest_state_date)}</span></p>
      <p>backend=<span class="mono">{escape(result.forecast_backend)}</span> | eval=<span class="mono">{escape(result.evaluation_window_start)} -> {escape(result.evaluation_window_end)}</span></p>
      <div class="grid">
        <div class="card"><div class="label">Info Items</div><div class="value">{int(manifest.get('info_item_count', 0))}</div></div>
        <div class="card"><div class="label">Timestamp Coverage</div><div class="value">{escape(_fmt_pct(manifest.get('publish_timestamp_coverage_ratio', 0.0)))}</div></div>
        <div class="card"><div class="label">Best Horizon</div><div class="value">{escape(str(all_info.get('best_horizon', '-')))}</div></div>
        <div class="card"><div class="label">Best Rank IC</div><div class="value">{escape(_fmt_num(all_info.get('best_horizon_rank_ic', 0.0)))}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>All Info Horizon Metrics</h2>
      <table>
        <thead>
          <tr><th>horizon</th><th>available_rows</th><th>shadow_ic</th><th>quant_ic</th><th>uplift_ic</th><th>shadow_spread</th><th>uplift_spread</th></tr>
        </thead>
        <tbody>{''.join(metric_rows)}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Top Event Tags</h2>
      <table>
        <thead>
          <tr><th>tag</th><th>items</th><th>best_horizon</th><th>best_rank_ic</th></tr>
        </thead>
        <tbody>{''.join(tag_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
