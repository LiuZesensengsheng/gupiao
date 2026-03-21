from __future__ import annotations

from html import escape
from pathlib import Path

from src.application.v2_ranking_research_runtime import V2RankingResearchResult


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


def _top_coefficients(model_payload: dict[str, object], *, limit: int = 5) -> list[tuple[str, float]]:
    feature_names = model_payload.get("feature_names", [])
    coef = model_payload.get("coef", [])
    if not isinstance(feature_names, list) or not isinstance(coef, list):
        return []
    ranked: list[tuple[str, float]] = []
    for name, value in zip(feature_names, coef):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        ranked.append((str(name), number))
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    return ranked[: max(0, int(limit))]


def write_v2_ranking_research_report(out_path: str | Path, result: V2RankingResearchResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    leader_eval = dict(result.leader_manifest.get("evaluation", {}))
    signal = dict(result.signal_training_manifest)
    leader_coefs = _top_coefficients(result.leader_rank_model)
    exit_coefs = _top_coefficients(result.exit_behavior_model)

    lines = [
        f"# V2 Ranking Research Report",
        "",
        f"- strategy: `{result.strategy_id}`",
        f"- latest_state_date: `{result.latest_state_date}`",
        f"- forecast_backend: `{result.forecast_backend}`",
        f"- retrain_days: `{result.retrain_days}`",
        f"- training_window_days: `{result.training_window_days}`",
        f"- split: `{result.split_mode}` | embargo=`{result.embargo_days}d` | fit_scope=`{result.fit_scope}` | evaluation_scope=`{result.evaluation_scope}`",
        f"- trajectory_steps: `{result.trajectory_steps}` | fit_steps: `{result.fit_steps}` | evaluation_steps: `{result.evaluation_steps}`",
        f"- evaluation_window: `{result.evaluation_window_start}` -> `{result.evaluation_window_end}`",
        "",
        "## Leader Metrics",
        "",
        f"- candidate_recall_at_k: `{_fmt_pct(leader_eval.get('candidate_recall_at_k', 0.0))}`",
        f"- conviction_precision_at_1: `{_fmt_pct(leader_eval.get('conviction_precision_at_1', 0.0))}`",
        f"- ndcg_at_k: `{_fmt_num(leader_eval.get('ndcg_at_k', 0.0))}`",
        f"- hard_negative_survival_recall: `{_fmt_pct(leader_eval.get('hard_negative_survival_recall', 0.0))}`",
        f"- hard_negative_filter_rate: `{_fmt_pct(leader_eval.get('hard_negative_filter_rate', 0.0))}`",
        "",
        "## Signal Metrics",
        "",
        f"- leader_eval_top1_hit_rate: `{_fmt_pct(signal.get('leader_eval_top1_hit_rate', 0.0))}`",
        f"- leader_eval_top3_recall: `{_fmt_pct(signal.get('leader_eval_top3_recall', 0.0))}`",
        f"- leader_eval_ndcg_at_3: `{_fmt_num(signal.get('leader_eval_ndcg_at_3', 0.0))}`",
        f"- leader_eval_filter_precision: `{_fmt_pct(signal.get('leader_eval_filter_precision', 0.0))}`",
        f"- leader_eval_filter_recall: `{_fmt_pct(signal.get('leader_eval_filter_recall', 0.0))}`",
        f"- leader_eval_filter_pass_rate: `{_fmt_pct(signal.get('leader_eval_filter_pass_rate', 0.0))}`",
        f"- leader_eval_true_leader_survival_recall: `{_fmt_pct(signal.get('leader_eval_true_leader_survival_recall', 0.0))}`",
        f"- leader_eval_hard_negative_filter_rate: `{_fmt_pct(signal.get('leader_eval_hard_negative_filter_rate', 0.0))}`",
        f"- exit_eval_rank_corr: `{_fmt_num(signal.get('exit_eval_rank_corr', 0.0))}`",
        f"- exit_eval_precision: `{_fmt_pct(signal.get('exit_eval_precision', 0.0))}`",
        f"- exit_eval_recall: `{_fmt_pct(signal.get('exit_eval_recall', 0.0))}`",
        f"- exit_eval_accuracy: `{_fmt_pct(signal.get('exit_eval_accuracy', 0.0))}`",
        "",
        "## Label Coverage",
        "",
        f"- full leader rows: `{result.full_training_label_manifest.get('leader_row_count', 0)}` | full exit rows: `{result.full_training_label_manifest.get('exit_row_count', 0)}`",
        f"- fit leader rows: `{result.fit_training_label_manifest.get('leader_row_count', 0)}` | fit exit rows: `{result.fit_training_label_manifest.get('exit_row_count', 0)}`",
        f"- evaluation leader rows: `{result.evaluation_training_label_manifest.get('leader_row_count', 0)}` | evaluation exit rows: `{result.evaluation_training_label_manifest.get('exit_row_count', 0)}`",
        "",
        "## Top Leader Candidates",
        "",
        "| rank | symbol | theme | phase | role | candidate | conviction | negative | hard_negative |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, item in enumerate(result.leader_candidates, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(item.get("symbol", "")),
                    str(item.get("theme", "")),
                    str(item.get("theme_phase", "")),
                    str(item.get("role", "")),
                    _fmt_num(item.get("candidate_score", 0.0)),
                    _fmt_num(item.get("conviction_score", 0.0)),
                    _fmt_num(item.get("negative_score", 0.0)),
                    "Y" if bool(item.get("hard_negative", False)) else "N",
                ]
            )
            + " |"
        )

    if leader_coefs:
        lines.extend(
            [
                "",
                "## Leader Model Top Coefficients",
                "",
                "| feature | coef |",
                "| --- | --- |",
            ]
        )
        for name, value in leader_coefs:
            lines.append(f"| {name} | {_fmt_num(value)} |")

    if exit_coefs:
        lines.extend(
            [
                "",
                "## Exit Model Top Coefficients",
                "",
                "| feature | coef |",
                "| --- | --- |",
            ]
        )
        for name, value in exit_coefs:
            lines.append(f"| {name} | {_fmt_num(value)} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_v2_ranking_research_dashboard(out_path: str | Path, result: V2RankingResearchResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    leader_eval = dict(result.leader_manifest.get("evaluation", {}))
    signal = dict(result.signal_training_manifest)
    rows = []
    for idx, item in enumerate(result.leader_candidates, start=1):
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{escape(str(item.get('symbol', '')))}</td>"
            f"<td>{escape(str(item.get('theme', '')))}</td>"
            f"<td>{escape(str(item.get('theme_phase', '')))}</td>"
            f"<td>{escape(str(item.get('role', '')))}</td>"
            f"<td>{escape(_fmt_num(item.get('candidate_score', 0.0)))}</td>"
            f"<td>{escape(_fmt_num(item.get('conviction_score', 0.0)))}</td>"
            f"<td>{escape(_fmt_num(item.get('negative_score', 0.0)))}</td>"
            f"<td>{'Y' if bool(item.get('hard_negative', False)) else 'N'}</td>"
            "</tr>"
        )

    cards = [
        ("Candidate Recall@K", _fmt_pct(leader_eval.get("candidate_recall_at_k", 0.0))),
        ("Conviction Precision@1", _fmt_pct(leader_eval.get("conviction_precision_at_1", 0.0))),
        ("NDCG@K", _fmt_num(leader_eval.get("ndcg_at_k", 0.0))),
        ("Hard Negative Survival", _fmt_pct(leader_eval.get("hard_negative_survival_recall", 0.0))),
        ("Leader Top1 Hit", _fmt_pct(signal.get("leader_eval_top1_hit_rate", 0.0))),
        ("Leader Filter Recall", _fmt_pct(signal.get("leader_eval_filter_recall", 0.0))),
        ("Exit Rank Corr", _fmt_num(signal.get("exit_eval_rank_corr", 0.0))),
    ]
    card_html = "".join(
        [
            (
                "<div class='card'>"
                f"<div class='label'>{escape(label)}</div>"
                f"<div class='value'>{escape(value)}</div>"
                "</div>"
            )
            for label, value in cards
        ]
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 Ranking Research</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffaf2;
      --ink: #1f1a16;
      --muted: #7a6b5c;
      --accent: #b04b2f;
      --line: #e4d7c7;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: radial-gradient(circle at top, #fff5e6 0%, var(--bg) 55%, #efe6d8 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(176,75,47,0.12), rgba(34,114,101,0.10));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 16px 40px rgba(56, 38, 16, 0.08);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .hero p {{
      margin: 4px 0;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .section {{
      margin-top: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 12px 30px rgba(56, 38, 16, 0.05);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .mono {{
      font-family: Consolas, monospace;
    }}
    @media (max-width: 720px) {{
      .hero h1 {{
        font-size: 24px;
      }}
      .value {{
        font-size: 22px;
      }}
      th, td {{
        font-size: 12px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>V2 Ranking Research</h1>
      <p>strategy=<span class="mono">{escape(result.strategy_id)}</span> | as_of=<span class="mono">{escape(result.latest_state_date)}</span></p>
      <p>backend=<span class="mono">{escape(result.forecast_backend)}</span> | steps=<span class="mono">{result.trajectory_steps}</span> | eval=<span class="mono">{escape(result.evaluation_window_start)} -> {escape(result.evaluation_window_end)}</span></p>
      <p>fit_scope=<span class="mono">{escape(result.fit_scope)}</span> | evaluation_scope=<span class="mono">{escape(result.evaluation_scope)}</span></p>
      <div class="grid">{card_html}</div>
    </section>

    <section class="section">
      <h2>Top Leader Candidates</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>symbol</th><th>theme</th><th>phase</th><th>role</th><th>candidate</th><th>conviction</th><th>negative</th><th>hard_negative</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
