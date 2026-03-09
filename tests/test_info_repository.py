from __future__ import annotations

import pandas as pd

from src.infrastructure.info_repository import load_v2_info_items


def test_load_v2_info_items_supports_defaults_and_dedupes_by_confidence(tmp_path) -> None:
    info_path = tmp_path / "info.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title,confidence,source_url,event_tag",
                "2026-03-01,stock,600160.SH,short,bullish,old row,0.6,https://a,contract_win",
                "2026-03-01,stock,600160.SH,short,bullish,old row,0.9,https://a,contract_win",
                "2026-03-01,market,MARKET,mid,bearish,macro risk,0.8,https://b,regulatory_negative",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-05"),
        lookback_days=10,
    )

    assert len(items) == 2
    stock_item = next(item for item in items if item.target_type == "stock")
    market_item = next(item for item in items if item.target_type == "market")
    assert stock_item.info_type == "news"
    assert stock_item.confidence == 0.9
    assert stock_item.source_weight == 0.85
    assert market_item.event_tag == "regulatory_negative"


def test_load_v2_info_items_filters_by_info_type(tmp_path) -> None:
    info_path = tmp_path / "info.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,info_type,title",
                "2026-03-01,stock,600160.SH,short,bullish,news,news row",
                "2026-03-01,stock,600160.SH,mid,bullish,announcement,announcement row",
                "2026-03-01,stock,600160.SH,mid,bullish,research,research row",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-05"),
        lookback_days=10,
        info_types=("announcement", "research"),
    )

    assert {item.info_type for item in items} == {"announcement", "research"}
