from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.application.v2_contracts import InfoItem
from src.review_analytics.info_manifest import (
    InfoManifestDependencies,
    build_info_manifest_payload,
)
from src.review_analytics.prediction_review import (
    PredictionReviewDependencies,
    load_prediction_review_context,
)


def test_prediction_review_context_builds_derived_10d_window(tmp_path: Path) -> None:
    backtest_path = tmp_path / "backtest_summary.json"
    payload = {
        "learned": {
            "n_days": 42,
            "horizon_metrics": {
                "5d": {"rank_ic": 0.08, "top_k_hit_rate": 0.61, "top_bottom_spread": 0.05},
                "20d": {"rank_ic": 0.12, "top_k_hit_rate": 0.65, "top_bottom_spread": 0.08},
            },
            "nav_curve": [1.0, 1.02, 1.03, 1.05, 1.08, 1.10],
            "excess_nav_curve": [1.0, 1.01, 1.02, 1.025, 1.04, 1.05],
            "curve_dates": ["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14", "2026-03-17"],
        }
    }
    backtest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = {"backtest_summary": "backtest_summary.json", "run_id": "research_20260315"}
    deps = PredictionReviewDependencies(
        path_from_manifest_entry=lambda entry, run_dir: None if not entry else Path(run_dir) / str(entry),
        load_json_dict=lambda path_like: json.loads(Path(path_like).read_text(encoding="utf-8")),
    )

    review, calibration_priors = load_prediction_review_context(
        manifest=manifest,
        manifest_path=tmp_path / "research_manifest.json",
        deps=deps,
    )

    assert "10d" in calibration_priors
    assert calibration_priors["10d"]["rank_ic"] == pytest.approx(0.102)
    assert review.windows["5d"].sample_size == 42
    assert review.notes[0].endswith("research_20260315")


def test_info_manifest_payload_captures_counts_and_hash(tmp_path: Path) -> None:
    info_file = tmp_path / "info.json"
    info_file.write_text("{}", encoding="utf-8")
    info_items = [
        InfoItem(
            date="2026-03-15",
            target_type="stock",
            target="AAA",
            horizon="20d",
            direction="up",
            info_type="announcement",
            title="ann",
            source_subset="announcements",
        ),
        InfoItem(
            date="2026-03-16",
            target_type="market",
            target="all",
            horizon="5d",
            direction="up",
            info_type="news",
            title="news",
            source_subset="market_news",
        ),
    ]
    deps = InfoManifestDependencies(
        sha256_file=lambda path_like: "file_hash" if Path(path_like).exists() else "",
        stable_json_hash=lambda payload: f"json:{len(payload)}",
    )

    payload = build_info_manifest_payload(
        settings={
            "info_shadow_only": False,
            "info_source_mode": "layered",
            "info_types": ["news", "announcement"],
            "info_subsets": ["market_news", "announcements"],
            "announcement_event_tags": ["earnings"],
            "info_cutoff_time": "15:00:00",
        },
        info_file=str(info_file),
        info_items=info_items,
        as_of_date=pd.Timestamp("2026-03-16"),
        config_hash="cfg_hash",
        shadow_enabled=True,
        shadow_report={"coverage_summary": {"market_coverage_ratio": 0.7, "stock_coverage_ratio": 0.5}},
        deps=deps,
    )

    assert payload["info_hash"] == "file_hash"
    assert payload["announcement_count"] == 1
    assert payload["market_news_count"] == 1
    assert payload["publish_timestamp_count"] == 0
    assert payload["publish_timestamp_coverage_ratio"] == 0.0
    assert payload["date_window"] == {"start": "2026-03-15", "end": "2026-03-16"}
    assert payload["market_coverage_ratio"] == 0.7
    assert payload["stock_coverage_ratio"] == 0.5
    assert payload["info_cutoff_time"] == "15:00:00"
