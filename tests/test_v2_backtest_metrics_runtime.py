from __future__ import annotations

import warnings

import pandas as pd

from src.application.v2_backtest_metrics_runtime import panel_slice_metrics


def test_panel_slice_metrics_silences_constant_input_warning() -> None:
    scored_rows = pd.DataFrame(
        {
            "score": [0.5, 0.5, 0.5, 0.5],
            "realized_ret_20d": [0.01, 0.02, 0.03, 0.04],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rank_ic, top_decile_return, top_bottom_spread, top_k_hit_rate = panel_slice_metrics(scored_rows)

    assert rank_ic == 0.0
    assert top_decile_return == 0.01
    assert top_bottom_spread == -0.03
    assert top_k_hit_rate == 1.0
    assert caught == []
