from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.domain.entities import Security
from src.domain.symbols import normalize_symbol


def load_watchlist(path: str | Path) -> Tuple[Security, List[Security], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    market = payload["market_index"]
    market_security = Security(symbol=market["symbol"], name=market["name"], sector=market.get("sector", "大盘"))

    stocks: List[Security] = []
    sector_map: Dict[str, str] = {}
    for item in payload["stocks"]:
        symbol = normalize_symbol(item["symbol"]).symbol
        sector = item.get("sector", "其他")
        stocks.append(Security(symbol=symbol, name=item["name"], sector=sector))
        sector_map[symbol] = sector
    return market_security, stocks, sector_map

