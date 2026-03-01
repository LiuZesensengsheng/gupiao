from __future__ import annotations

import pandas as pd

from src.infrastructure.cross_section_forecast import forecast_cross_section_state


def _make_market_context_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    base = pd.DataFrame({"date": dates})
    base["idx_cyb_trend_20_60"] = 0.08
    base["idx_sh_trend_20_60"] = 0.02
    base["idx_cyb_ret_20"] = 0.10
    base["idx_sh_ret_20"] = 0.03
    base["mrg_mkt_fin_net_buy_z20"] = 0.40
    base["mrg_mkt_fin_sec_spread_chg5"] = 0.25
    base["breadth_up_down_diff"] = 0.18
    base["breadth_up_ratio"] = 0.58
    base["breadth_down_ratio"] = 0.27
    base["breadth_limit_spread"] = 0.03
    base["breadth_limit_up_ratio"] = 0.04
    base["breadth_amount_z20"] = 0.15
    return base


def test_forecast_cross_section_state_returns_bounded_metrics() -> None:
    record = forecast_cross_section_state(_make_market_context_frame())

    assert str(record.as_of_date.date()) == "2024-04-19"
    assert -1.0 <= record.large_vs_small_bias <= 1.0
    assert -1.0 <= record.growth_vs_value_bias <= 1.0
    assert -1.0 <= record.fund_flow_strength <= 1.0
    assert -1.0 <= record.margin_risk_on_score <= 1.0
    assert -1.0 <= record.breadth_strength <= 1.0
    assert 0.0 <= record.leader_participation <= 1.0
    assert 0.0 <= record.weak_stock_ratio <= 1.0


def test_forecast_cross_section_state_reflects_positive_flow_and_breadth() -> None:
    record = forecast_cross_section_state(_make_market_context_frame())

    assert record.large_vs_small_bias > 0.0
    assert record.growth_vs_value_bias > 0.0
    assert record.fund_flow_strength > 0.0
    assert record.breadth_strength > 0.0
