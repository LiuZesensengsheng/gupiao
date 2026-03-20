from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.application.v2_runtime_settings as runtime_settings
import src.application.v2_services as legacy_services
import src.application.v2_universe_generator as current_generator_module
from src.workflows import research_workflow


GeneratorFn = Callable[..., Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_git_generator(commit: str) -> GeneratorFn:
    result = subprocess.run(
        ["git", "show", f"{commit}:src/application/v2_universe_generator.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    module = ModuleType(f"dynamic_universe_{commit}")
    module.__file__ = f"<git:{commit}:src/application/v2_universe_generator.py>"
    exec(result.stdout, module.__dict__)
    generator_fn = module.__dict__.get("generate_dynamic_universe")
    if not callable(generator_fn):
        raise RuntimeError(f"{commit} does not expose generate_dynamic_universe")
    return generator_fn


@contextmanager
def _patched_generator(generator_fn: GeneratorFn):
    original_current = current_generator_module.generate_dynamic_universe
    original_runtime = runtime_settings.generate_dynamic_universe
    original_services = legacy_services.generate_dynamic_universe
    current_generator_module.generate_dynamic_universe = generator_fn
    runtime_settings.generate_dynamic_universe = generator_fn
    legacy_services.generate_dynamic_universe = generator_fn
    try:
        yield
    finally:
        current_generator_module.generate_dynamic_universe = original_current
        runtime_settings.generate_dynamic_universe = original_runtime
        legacy_services.generate_dynamic_universe = original_services


def _build_config_variant(
    *,
    base_config_path: Path,
    output_dir: Path,
    info_mode: str,
) -> Path:
    payload = _read_json(base_config_path)
    common = payload.get("common", {}) if isinstance(payload.get("common"), dict) else {}
    daily = payload.get("daily", {}) if isinstance(payload.get("daily"), dict) else {}
    common_copy = dict(common)
    daily_copy = dict(daily)

    if info_mode == "off":
        daily_copy["use_info_fusion"] = False
        daily_copy["use_learned_info_fusion"] = False
        daily_copy["info_shadow_only"] = False
        daily_copy["enable_insight_memory"] = False
    elif info_mode == "on":
        if "use_info_fusion" not in daily_copy:
            daily_copy["use_info_fusion"] = False
        if "use_learned_info_fusion" not in daily_copy:
            daily_copy["use_learned_info_fusion"] = True
        if "info_shadow_only" not in daily_copy:
            daily_copy["info_shadow_only"] = True
        if "enable_insight_memory" not in daily_copy:
            daily_copy["enable_insight_memory"] = True
    else:
        raise ValueError(f"unsupported info mode: {info_mode}")

    variant_path = output_dir / f"{base_config_path.stem}.info_{info_mode}.json"
    _write_json(
        variant_path,
        {
            "common": common_copy,
            "daily": daily_copy,
        },
    )
    return variant_path


def _summary_view(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_date": str(summary.get("start_date", "")),
        "end_date": str(summary.get("end_date", "")),
        "n_days": int(summary.get("n_days", 0) or 0),
        "annual_return": float(summary.get("annual_return", 0.0) or 0.0),
        "excess_annual_return": float(summary.get("excess_annual_return", 0.0) or 0.0),
        "information_ratio": float(summary.get("information_ratio", 0.0) or 0.0),
        "max_drawdown": float(summary.get("max_drawdown", 0.0) or 0.0),
        "excess_total_return": float(summary.get("excess_total_return", 0.0) or 0.0),
    }


def _load_reference_rows() -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    paths = {
        "formal_best_holdout_20260316_132232": ROOT / "artifacts" / "v2" / "swing_v2" / "20260316_132232" / "backtest_summary.json",
        "formal_current_holdout_20260319_100152": ROOT / "artifacts" / "v2" / "swing_v2" / "20260319_100152" / "backtest_summary.json",
        "fast_rule_20260318": ROOT / "reports" / "dynamic300_rule_fast_backtest_20260318.json",
    }
    for name, path in paths.items():
        payload = _read_json(path)
        if not payload:
            continue
        if name.startswith("fast_rule"):
            summary = _summary_view(dict(payload.get("summary", {})))
        else:
            summary = _summary_view(dict(payload.get("baseline", payload)))
        references.append({"name": name, "summary": summary, "path": str(path)})
    return references


def _run_single_combo(
    *,
    generator_mode: str,
    info_mode: str,
    base_config_path: Path,
    config_output_dir: Path,
    source: str | None,
    universe_limit: int,
    target_size: int,
    coarse_size: int,
    retrain_days: int,
    forecast_backend: str,
    training_window_days: int | None,
    split_mode: str,
    embargo_days: int,
    cache_root: str,
    refresh_cache: bool,
    v2_commit: str,
) -> dict[str, Any]:
    combo_name = f"{generator_mode}__info_{info_mode}"
    print(f"[ABLATION] start {combo_name}", flush=True)
    combo_cache_root = str((ROOT / str(cache_root) / "ablation_compare" / combo_name).resolve())
    config_path = _build_config_variant(
        base_config_path=base_config_path,
        output_dir=config_output_dir,
        info_mode=info_mode,
    )
    if generator_mode == "v2_leaders":
        generator_fn = _load_git_generator(v2_commit)
        effective_generator_version = f"git:{v2_commit}:dynamic_universe_v2_leaders"
    elif generator_mode == "v3_fresh_pool":
        generator_fn = current_generator_module.generate_dynamic_universe
        effective_generator_version = "current:dynamic_universe_v3_fresh_pool"
    else:
        raise ValueError(f"unsupported generator mode: {generator_mode}")

    workflow_kwargs = {
        "strategy_id": "swing_v2",
        "config_path": str(config_path),
        "source": source,
        "universe_limit": int(universe_limit),
        "dynamic_universe": True,
        "generator_target_size": int(target_size),
        "generator_coarse_size": int(coarse_size),
        "retrain_days": int(retrain_days),
        "forecast_backend": str(forecast_backend),
        "training_window_days": training_window_days,
        "split_mode": str(split_mode),
        "embargo_days": int(embargo_days),
        "cache_root": combo_cache_root,
        "refresh_cache": bool(refresh_cache),
        "skip_calibration": True,
        "skip_learning": True,
    }

    with _patched_generator(generator_fn):
        baseline, calibration, learning = legacy_services._run_v2_research_workflow_impl(**workflow_kwargs)
        trajectory = research_workflow.last_research_trajectory()
        full_cycle_summary = None
        if trajectory is not None:
            full_cycle = legacy_services.run_v2_backtest_live(
                strategy_id="swing_v2",
                config_path=str(config_path),
                source=source,
                universe_limit=int(universe_limit),
                dynamic_universe=True,
                generator_target_size=int(target_size),
                generator_coarse_size=int(coarse_size),
                retrain_days=int(retrain_days),
                forecast_backend=str(forecast_backend),
                training_window_days=training_window_days,
                cache_root=combo_cache_root,
                refresh_cache=False,
                trajectory=trajectory,
            )
            full_cycle_summary = _summary_view(asdict(full_cycle))

    result = {
        "combo": combo_name,
        "generator_mode": generator_mode,
        "info_mode": info_mode,
        "effective_generator_version": effective_generator_version,
        "config_path": str(config_path),
        "cache_root": combo_cache_root,
        "holdout": _summary_view(asdict(baseline)),
        "full_cycle": full_cycle_summary,
        "calibration_best_score": float(getattr(calibration, "best_score", 0.0)),
        "learning_model_present": bool(getattr(learning, "model", None)),
    }
    print(
        "[ABLATION] done "
        f"{combo_name} | holdout_excess_annual={result['holdout']['excess_annual_return']:.2%} "
        f"| holdout_ir={result['holdout']['information_ratio']:.3f}",
        flush=True,
    )
    return result


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Formal Generator x Info Ablation",
        "",
        f"- created_at: {payload.get('created_at', '')}",
        f"- base_config: {payload.get('base_config', '')}",
        f"- source: {payload.get('source', '')}",
        f"- training_window_days: {payload.get('training_window_days', '')}",
        f"- split_mode: {payload.get('split_mode', '')}",
        f"- embargo_days: {payload.get('embargo_days', '')}",
        "",
        "## Holdout Matrix",
        "",
        "| combo | generator | info | days | excess annual | IR | MDD | period |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("results", []):
        holdout = dict(row.get("holdout", {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("combo", "")),
                    str(row.get("generator_mode", "")),
                    str(row.get("info_mode", "")),
                    str(holdout.get("n_days", 0)),
                    f"{float(holdout.get('excess_annual_return', 0.0)):.2%}",
                    f"{float(holdout.get('information_ratio', 0.0)):.3f}",
                    f"{float(holdout.get('max_drawdown', 0.0)):.2%}",
                    f"{holdout.get('start_date', '')} -> {holdout.get('end_date', '')}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Full Cycle Matrix",
            "",
            "| combo | generator | info | days | excess annual | IR | MDD | period |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload.get("results", []):
        summary = dict(row.get("full_cycle") or {})
        if not summary:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("combo", "")),
                    str(row.get("generator_mode", "")),
                    str(row.get("info_mode", "")),
                    str(summary.get("n_days", 0)),
                    f"{float(summary.get('excess_annual_return', 0.0)):.2%}",
                    f"{float(summary.get('information_ratio', 0.0)):.3f}",
                    f"{float(summary.get('max_drawdown', 0.0)):.2%}",
                    f"{summary.get('start_date', '')} -> {summary.get('end_date', '')}",
                ]
            )
            + " |"
        )
    references = list(payload.get("references", []))
    if references:
        lines.extend(
            [
                "",
                "## Existing References",
                "",
                "| name | days | excess annual | IR | period |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in references:
            summary = dict(row.get("summary", {}))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("name", "")),
                        str(summary.get("n_days", 0)),
                        f"{float(summary.get('excess_annual_return', 0.0)):.2%}",
                        f"{float(summary.get('information_ratio', 0.0)):.3f}",
                        f"{summary.get('start_date', '')} -> {summary.get('end_date', '')}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run formal 2x2 generator/info ablation on the V2 research stack.")
    parser.add_argument("--base-config", default="config/api_generated300_train_5y.json")
    parser.add_argument("--source", default="local")
    parser.add_argument("--universe-limit", type=int, default=300)
    parser.add_argument("--target-size", type=int, default=300)
    parser.add_argument("--coarse-size", type=int, default=1000)
    parser.add_argument("--retrain-days", type=int, default=20)
    parser.add_argument("--forecast-backend", default="linear")
    parser.add_argument("--training-window-days", type=int, default=480)
    parser.add_argument("--split-mode", default="purged_wf")
    parser.add_argument("--embargo-days", type=int, default=20)
    parser.add_argument("--cache-root", default="artifacts/v2/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--v2-commit", default="01ce465")
    parser.add_argument(
        "--generators",
        default="v2_leaders,v3_fresh_pool",
        help="Comma separated generator modes",
    )
    parser.add_argument(
        "--info-modes",
        default="off,on",
        help="Comma separated info modes",
    )
    parser.add_argument("--output-dir", default="reports/ablation")
    args = parser.parse_args()

    base_config_path = (ROOT / str(args.base_config)).resolve()
    output_dir = (ROOT / str(args.output_dir)).resolve()
    config_output_dir = output_dir / "configs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"formal_generator_info_ablation_{timestamp}.json"
    md_path = output_dir / f"formal_generator_info_ablation_{timestamp}.md"

    generators = [item.strip() for item in str(args.generators).split(",") if item.strip()]
    info_modes = [item.strip() for item in str(args.info_modes).split(",") if item.strip()]

    results: list[dict[str, Any]] = []
    for generator_mode in generators:
        for info_mode in info_modes:
            results.append(
                _run_single_combo(
                    generator_mode=generator_mode,
                    info_mode=info_mode,
                    base_config_path=base_config_path,
                    config_output_dir=config_output_dir,
                    source=None if str(args.source).strip().lower() == "none" else str(args.source).strip(),
                    universe_limit=int(args.universe_limit),
                    target_size=int(args.target_size),
                    coarse_size=int(args.coarse_size),
                    retrain_days=int(args.retrain_days),
                    forecast_backend=str(args.forecast_backend),
                    training_window_days=(
                        None
                        if int(args.training_window_days) <= 0
                        else int(args.training_window_days)
                    ),
                    split_mode=str(args.split_mode),
                    embargo_days=int(args.embargo_days),
                    cache_root=str(args.cache_root),
                    refresh_cache=bool(args.refresh_cache),
                    v2_commit=str(args.v2_commit),
                )
            )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(base_config_path),
        "source": str(args.source),
        "training_window_days": int(args.training_window_days),
        "split_mode": str(args.split_mode),
        "embargo_days": int(args.embargo_days),
        "results": results,
        "references": _load_reference_rows(),
    }
    _write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"[ABLATION] report_json={json_path}", flush=True)
    print(f"[ABLATION] report_md={md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
