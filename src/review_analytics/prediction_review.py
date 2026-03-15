from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.application.v2_contracts import PredictionReviewState, PredictionReviewWindow


@dataclass(frozen=True)
class PredictionReviewDependencies:
    path_from_manifest_entry: Callable[..., Path | None]
    load_json_dict: Callable[[object], dict[str, object]]


def load_prediction_review_context(
    *,
    manifest: dict[str, object],
    manifest_path: Path | None,
    deps: PredictionReviewDependencies,
) -> tuple[PredictionReviewState, dict[str, dict[str, float]]]:
    if manifest_path is None:
        return PredictionReviewState(), {}
    backtest_path = deps.path_from_manifest_entry(manifest.get("backtest_summary"), run_dir=manifest_path.parent)
    payload = deps.load_json_dict(backtest_path)
    if not payload:
        return PredictionReviewState(), {}
    learned = payload.get("learned")
    baseline = payload.get("baseline")
    summary = learned if isinstance(learned, dict) else (baseline if isinstance(baseline, dict) else payload)
    raw_horizon_metrics = summary.get("horizon_metrics", {}) if isinstance(summary, dict) else {}
    calibration_priors = {
        str(key): {str(metric): float(value) for metric, value in raw.items()}
        for key, raw in raw_horizon_metrics.items()
        if isinstance(raw, dict)
    }
    if "10d" not in calibration_priors and ("5d" in calibration_priors or "20d" in calibration_priors):
        five = calibration_priors.get("5d", {})
        twenty = calibration_priors.get("20d", {})
        calibration_priors["10d"] = {
            "rank_ic": float(0.45 * float(five.get("rank_ic", 0.0)) + 0.55 * float(twenty.get("rank_ic", 0.0))),
            "top_k_hit_rate": float(
                0.45 * float(five.get("top_k_hit_rate", 0.0)) + 0.55 * float(twenty.get("top_k_hit_rate", 0.0))
            ),
            "top_bottom_spread": float(
                0.45 * float(five.get("top_bottom_spread", 0.0)) + 0.55 * float(twenty.get("top_bottom_spread", 0.0))
            ),
        }

    curve = [float(item) for item in summary.get("nav_curve", [])] if isinstance(summary, dict) else []
    excess_curve = [float(item) for item in summary.get("excess_nav_curve", [])] if isinstance(summary, dict) else []
    curve_dates = [str(item) for item in summary.get("curve_dates", [])] if isinstance(summary, dict) else []

    def _window_from_curve(window: int) -> PredictionReviewWindow:
        if window in {5, 20} and f"{window}d" in calibration_priors:
            metrics = calibration_priors[f"{window}d"]
            return PredictionReviewWindow(
                window_days=int(window),
                label=f"{window}日预测命中参考",
                hit_rate=float(metrics.get("top_k_hit_rate", 0.0)),
                avg_edge=float(metrics.get("top_bottom_spread", 0.0)),
                realized_return=0.0,
                sample_size=int(summary.get("n_days", 0)),
                note=f"研究期内 {window} 日横截面命中率与头尾价差。",
            )
        if len(curve) < max(2, window + 1):
            return PredictionReviewWindow(window_days=int(window), label=f"{window}日表现参考")
        recent_nav = curve[-(window + 1) :]
        base_nav = recent_nav[0]
        realized_return = 0.0 if abs(base_nav) <= 1e-12 else float(recent_nav[-1] / base_nav - 1.0)
        daily_rets = [
            float(nxt / prev - 1.0)
            for prev, nxt in zip(recent_nav[:-1], recent_nav[1:])
            if abs(prev) > 1e-12
        ]
        hit_rate = float(sum(1 for item in daily_rets if item > 0.0) / max(1, len(daily_rets)))
        avg_edge = 0.0
        if len(excess_curve) >= len(curve):
            recent_excess = excess_curve[-(window + 1) :]
            base_excess = recent_excess[0]
            if abs(base_excess) > 1e-12:
                avg_edge = float(recent_excess[-1] / base_excess - 1.0)
        return PredictionReviewWindow(
            window_days=int(window),
            label=f"{window}日策略表现",
            hit_rate=hit_rate,
            avg_edge=avg_edge,
            realized_return=realized_return,
            sample_size=int(len(daily_rets)),
            note=f"截至 {(curve_dates[-1] if curve_dates else '最近一期')} 的滚动净值表现。",
        )

    review = PredictionReviewState(
        windows={
            "5d": _window_from_curve(5),
            "20d": _window_from_curve(20),
            "60d": _window_from_curve(60),
        },
        notes=[f"复盘参考来自研究 run_id={str(manifest.get('run_id', '')).strip() or 'NA'}"],
    )
    return review, calibration_priors
