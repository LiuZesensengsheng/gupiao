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
                "date,target_type,target,horizon,direction,info_type,title,event_tag",
                "2026-03-01,stock,600160.SH,short,bullish,news,news row,",
                "2026-03-01,stock,600160.SH,mid,bullish,announcement,业绩预增公告,earnings_positive",
                "2026-03-01,stock,600160.SH,mid,bullish,research,research row,",
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


def test_load_v2_info_items_keeps_publish_datetime_when_available(tmp_path) -> None:
    info_path = tmp_path / "info.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,publish_datetime,target_type,target,horizon,direction,title",
                "2026-03-01,2026-03-01T09:35:00,stock,600160.SH,short,bullish,open news",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-01"),
        lookback_days=5,
    )

    assert len(items) == 1
    assert items[0].publish_datetime == "2026-03-01T09:35:00"


def test_load_v2_info_items_filters_rows_published_after_as_of_cutoff(tmp_path) -> None:
    info_path = tmp_path / "info.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,publish_datetime,target_type,target,horizon,direction,title",
                "2026-03-01,2026-03-02T09:35:00,stock,600160.SH,short,bullish,future publish",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-01"),
        lookback_days=5,
    )

    assert items == []


def test_load_v2_info_items_derives_publish_datetime_from_eastmoney_source_url(tmp_path) -> None:
    info_path = tmp_path / "info.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,info_type,title,source_url,event_tag",
                "2026-03-02,stock,600160.SH,mid,bullish,announcement,业绩预增公告,https://data.eastmoney.com/notices/detail/600160/AN202603011642298303.html,earnings_positive",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-02"),
        lookback_days=5,
    )

    assert len(items) == 1
    assert items[0].publish_datetime == "2026-03-01T16:42:29"


def test_load_v2_info_items_respects_cutoff_time_for_same_day_timestamped_rows(tmp_path) -> None:
    info_path = tmp_path / "info.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,publish_datetime,target_type,target,horizon,direction,title",
                "2026-03-01,2026-03-01T16:42:00,stock,600160.SH,mid,bullish,after close notice",
            ]
        ),
        encoding="utf-8",
    )

    strict_items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-01"),
        lookback_days=5,
        cutoff_time="15:00:00",
    )
    relaxed_items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-03-01"),
        lookback_days=5,
        cutoff_time="23:59:59",
    )

    assert strict_items == []
    assert len(relaxed_items) == 1


def test_load_v2_info_items_infers_announcement_and_event_tag_from_legacy_notice(tmp_path) -> None:
    info_path = tmp_path / "legacy_notice.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title,source_url",
                "2026-02-01,stock,603619.SH,mid,bearish,中曼石油关于股东减持的公告,https://data.eastmoney.com/notices/detail/603619/AN202601011234.html",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-02-10"),
        lookback_days=20,
    )

    assert len(items) == 1
    assert items[0].info_type == "announcement"
    assert items[0].event_tag == "share_reduction"


def test_load_v2_info_items_infers_research_type_from_legacy_title(tmp_path) -> None:
    info_path = tmp_path / "legacy_research.csv"
    info_path.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title,source_url",
                "2026-02-01,stock,603619.SH,mid,bullish,国金证券股份有限公司关于中曼石油签署协议的核查意见,https://example.com/report",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_path,
        as_of_date=pd.Timestamp("2026-02-10"),
        lookback_days=20,
    )

    assert len(items) == 1
    assert items[0].info_type == "research"


def test_load_v2_info_items_reads_layered_directory_and_tracks_source_subset(tmp_path) -> None:
    info_dir = tmp_path / "info_parts"
    (info_dir / "market_news").mkdir(parents=True, exist_ok=True)
    (info_dir / "announcements").mkdir(parents=True, exist_ok=True)
    (info_dir / "research").mkdir(parents=True, exist_ok=True)
    (info_dir / "market_news" / "market.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title",
                "2026-02-01,market,MARKET,mid,bullish,macro support",
            ]
        ),
        encoding="utf-8",
    )
    (info_dir / "announcements" / "ann.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title,event_tag",
                "2026-02-01,stock,603619.SH,mid,bullish,中曼石油关于控股股东增持的公告,share_increase",
            ]
        ),
        encoding="utf-8",
    )
    (info_dir / "research" / "research.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title",
                "2026-02-01,stock,603619.SH,mid,bullish,机构首次覆盖并给出买入评级",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_dir,
        as_of_date=pd.Timestamp("2026-02-10"),
        lookback_days=20,
        source_mode="layered",
    )

    assert len(items) == 3
    assert {item.source_subset for item in items} == {"market_news", "announcements", "research"}
    assert {item.info_type for item in items} == {"news", "announcement", "research"}


def test_load_v2_info_items_allows_file_info_type_to_override_directory_default(tmp_path) -> None:
    info_dir = tmp_path / "info_parts"
    (info_dir / "market_news").mkdir(parents=True, exist_ok=True)
    (info_dir / "market_news" / "override.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,info_type,title",
                "2026-02-01,stock,603619.SH,mid,bullish,research,券商深度覆盖报告",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_dir,
        as_of_date=pd.Timestamp("2026-02-10"),
        lookback_days=20,
        source_mode="layered",
    )

    assert len(items) == 1
    assert items[0].source_subset == "market_news"
    assert items[0].info_type == "research"


def test_load_v2_info_items_filters_weak_announcements_by_event_tag(tmp_path) -> None:
    info_dir = tmp_path / "info_parts"
    (info_dir / "announcements").mkdir(parents=True, exist_ok=True)
    (info_dir / "announcements" / "ann.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title,event_tag",
                "2026-02-01,stock,600160.SH,mid,neutral,巨化股份董事会会议决议公告,",
                "2026-02-01,stock,600160.SH,mid,bullish,巨化股份2024年度业绩预增公告,earnings_positive",
            ]
        ),
        encoding="utf-8",
    )

    items = load_v2_info_items(
        info_dir,
        as_of_date=pd.Timestamp("2026-02-10"),
        lookback_days=20,
        source_mode="layered",
    )

    assert len(items) == 1
    assert items[0].title == "巨化股份2024年度业绩预增公告"
