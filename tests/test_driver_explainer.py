from __future__ import annotations

from src.interfaces.presenters.driver_explainer import format_driver_list, format_driver_text


def test_driver_explainer_formats_new_chart_pattern_features() -> None:
    assert "上影线占比" in format_driver_text("upper_shadow_ratio_1(+0.62)")
    assert "20日窄幅排名" in format_driver_text("narrow_range_rank_20(-0.44)")
    assert "突破20日前高" in format_driver_text("breakout_above_20_high(+)")
    assert "放量突破强度" in format_driver_text("volume_breakout_ratio(+1.10)")


def test_driver_explainer_formats_driver_list_with_chart_pattern_features() -> None:
    text = format_driver_list(
        [
            "upper_shadow_ratio_1(-0.40)",
            "range_contraction_5(+0.75)",
            "breakout_above_20_high(+)",
        ]
    )

    assert "上影线占比" in text
    assert "5日振幅收缩" in text
    assert "突破20日前高" in text


def test_driver_explainer_formats_interaction_chart_features() -> None:
    assert "缩量后突破强度" in format_driver_text("squeeze_breakout_score(+0.88)")
    assert "突破质量分" in format_driver_text("breakout_quality_score(+0.74)")
    assert "冲高回落风险" in format_driver_text("exhaustion_reversal_risk(-0.52)")
    assert "回踩修复强度" in format_driver_text("pullback_reclaim_score(+0.41)")
