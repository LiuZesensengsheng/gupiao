from __future__ import annotations

import re
from typing import Sequence


_NUMERIC_DRIVER_RE = re.compile(r"^(?P<name>.+)\((?P<val>[+-]?\d+(?:\.\d+)?)\)$")
_SIGN_DRIVER_RE = re.compile(r"^(?P<name>.+)\((?P<sign>[+-])\)$")

_BASE_LABELS = {
    "ret_1": "1日涨跌幅",
    "ret_5": "5日涨跌幅",
    "ret_20": "20日涨跌幅",
    "trend_5_20": "短中期趋势(5/20)",
    "trend_20_60": "中长期趋势(20/60)",
    "volatility_20": "20日波动率",
    "volatility_60": "60日波动率",
    "drawdown_20": "20日回撤",
    "price_pos_20": "20日价格位置",
    "vol_ratio_20": "20日量比",
    "vol_conc_5_20": "量能集中度(5/20)",
    "obv_z_20": "OBV强弱",
    "amihud_20": "流动性冲击(Amihud)",
    "atr_14": "14日真实波幅",
    "gap_1": "隔夜跳空",
    "bear_body_1": "当日阴线实体",
    "upper_shadow_ratio_1": "上影线占比",
    "lower_shadow_ratio_1": "下影线占比",
    "body_ratio_1": "实体占比",
    "up_streak_3": "近3日连阳强度",
    "down_streak_3": "近3日连阴强度",
    "narrow_range_rank_20": "20日窄幅排名",
    "range_contraction_5": "5日振幅收缩",
    "breakout_above_20_high": "突破20日前高",
    "breakdown_below_20_low": "跌破20日前低",
    "distance_to_20d_high": "距20日前高偏离",
    "distance_to_20d_low": "距20日前低偏离",
    "volume_breakout_ratio": "放量突破强度",
    "hvbd_recent_5": "近5日高位巨量阴线",
}

_BREADTH_LABELS = {
    "breadth_up_ratio": "上涨家数占比",
    "breadth_down_ratio": "下跌家数占比",
    "breadth_up_down_diff": "涨跌差",
    "breadth_limit_up_ratio": "涨停占比",
    "breadth_limit_down_ratio": "跌停占比",
    "breadth_limit_spread": "涨跌停差",
    "breadth_amount_z20": "市场成交额强弱",
    "breadth_coverage": "样本覆盖数",
}

_MARGIN_SUFFIX_LABELS = {
    "fin_balance_z20": "融资余额强弱",
    "fin_balance_chg5": "融资余额5日变化",
    "sec_balance_z20": "融券余额强弱",
    "fin_net_buy_z20": "融资净买入强弱",
    "sec_net_sell_z20": "融券净卖出强弱",
    "fin_sec_spread_chg5": "融资-融券差变化",
}

_INDEX_PREFIX_LABELS = {
    "idx_sh": "上证指数",
    "idx_sz": "深证成指",
    "idx_cyb": "创业板指",
}


def _strength(abs_val: float | None) -> str:
    if abs_val is None:
        return "未知"
    if abs_val >= 1.2:
        return "强"
    if abs_val >= 0.6:
        return "中"
    return "弱"


def _to_label(name: str) -> str:
    if name in _BASE_LABELS:
        return _BASE_LABELS[name]
    if name in _BREADTH_LABELS:
        return _BREADTH_LABELS[name]

    if name.startswith("mkt_"):
        base = name[4:]
        if base in _BASE_LABELS:
            return f"大盘{_BASE_LABELS[base]}"
        return f"大盘{base.replace('_', ' ')}"

    for idx_prefix, idx_label in _INDEX_PREFIX_LABELS.items():
        pref = f"{idx_prefix}_"
        if name.startswith(pref):
            suffix = name[len(pref) :]
            core = _BASE_LABELS.get(suffix, suffix.replace("_", " "))
            return f"{idx_label}{core}"

    if name.startswith("mrg_mkt_"):
        suffix = name[len("mrg_mkt_") :]
        return f"市场两融{_MARGIN_SUFFIX_LABELS.get(suffix, suffix.replace('_', ' '))}"
    if name.startswith("mrg_stk_"):
        suffix = name[len("mrg_stk_") :]
        return f"个股两融{_MARGIN_SUFFIX_LABELS.get(suffix, suffix.replace('_', ' '))}"

    return name.replace("_", " ")


def format_driver_text(raw_driver: str) -> str:
    text = str(raw_driver).strip()
    if not text:
        return "NA"

    m_num = _NUMERIC_DRIVER_RE.match(text)
    if m_num:
        name = _to_label(m_num.group("name").strip())
        val = float(m_num.group("val"))
        direction = "推涨" if val >= 0 else "压制"
        return f"{name}:{direction}({_strength(abs(val))}, 贡献{val:+.2f})"

    m_sign = _SIGN_DRIVER_RE.match(text)
    if m_sign:
        name = _to_label(m_sign.group("name").strip())
        direction = "推涨" if m_sign.group("sign") == "+" else "压制"
        return f"{name}:{direction}"

    return _to_label(text)


def format_driver_list(drivers: Sequence[str]) -> str:
    if not drivers:
        return "NA"
    return "；".join(format_driver_text(x) for x in drivers)
