from __future__ import annotations

import pandas as pd

from src.application.v2_contracts import InfoItem
from src.infrastructure.v2_info_fusion import build_info_state_maps


def test_build_info_state_maps_ignores_items_not_yet_published() -> None:
    market_state, sector_states, stock_states = build_info_state_maps(
        info_items=[
            InfoItem(
                date="2026-03-12",
                publish_datetime="2026-03-13T08:00:00",
                target_type="stock",
                target="AAA",
                horizon="mid",
                direction="bullish",
                info_type="news",
                title="future item",
                source_weight=0.85,
            ),
            InfoItem(
                date="2026-03-12",
                publish_datetime="2026-03-12T09:30:00",
                target_type="stock",
                target="AAA",
                horizon="mid",
                direction="bullish",
                info_type="news",
                title="current item",
                source_weight=0.85,
            ),
        ],
        as_of_date=pd.Timestamp("2026-03-12"),
        stock_symbols=["AAA"],
        sector_map={"AAA": "chips"},
        market_to_stock_carry=0.35,
        info_half_life_days=10.0,
        market_info_strength=1.0,
        stock_info_strength=1.0,
    )

    assert market_state.item_count == 0
    assert sector_states["chips"].item_count == 1
    assert stock_states["AAA"].item_count == 1
