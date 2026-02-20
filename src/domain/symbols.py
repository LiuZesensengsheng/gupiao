from __future__ import annotations

from dataclasses import dataclass


class SymbolError(ValueError):
    """Raised when a symbol cannot be normalized."""


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    code: str
    exchange: str

    @property
    def secid(self) -> str:
        market = 1 if self.exchange == "SH" else 0
        return f"{market}.{self.code}"


def normalize_symbol(symbol: str) -> SymbolInfo:
    text = symbol.strip().upper()
    if not text:
        raise SymbolError("Empty symbol.")
    if text == "MARKET":
        return SymbolInfo(symbol="MARKET", code="MARKET", exchange="MARKET")

    if "." in text:
        code, exchange = text.split(".", 1)
    else:
        code = text
        if not code.isdigit() or len(code) != 6:
            raise SymbolError(f"Unsupported symbol format: {symbol}")
        exchange = "SH" if code.startswith(("5", "6", "9")) else "SZ"

    exchange = exchange.upper()
    if not code.isdigit() or len(code) != 6:
        raise SymbolError(f"Unsupported symbol code: {symbol}")
    if exchange not in {"SH", "SZ"}:
        raise SymbolError(f"Unsupported exchange in symbol: {symbol}")
    return SymbolInfo(symbol=f"{code}.{exchange}", code=code, exchange=exchange)

