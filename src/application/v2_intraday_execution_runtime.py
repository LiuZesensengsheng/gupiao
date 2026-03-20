from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.symbols import normalize_symbol
from src.infrastructure.market_data import (
    DataError,
    EASTMONEY_KLINE_URL,
    _HTTP_SESSION,
    _post_tushare_sdk_http,
)


_INTRADAY_AUTO_SOURCE_CHAIN = ("tushare", "eastmoney", "local")
_INTRADAY_TIMEFRAME_ALIASES = {
    "1": "1m",
    "1m": "1m",
    "1min": "1m",
    "5": "5m",
    "5m": "5m",
    "5min": "5m",
    "15": "15m",
    "15m": "15m",
    "15min": "15m",
    "30": "30m",
    "30m": "30m",
    "30min": "30m",
    "60": "60m",
    "60m": "60m",
    "60min": "60m",
}
_INTRADAY_TUSHARE_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
}
_INTRADAY_EASTMONEY_KLT = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
}


@dataclass(frozen=True)
class IntradayExecutionAssessment:
    symbol: str
    signal: str
    timeframe: str
    data_date: str
    stop_price: float
    take_profit_price: float
    vwap_gap: float
    drawdown_from_high: float
    break_state: str
    reason: str


def _normalize_timeframe(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return str(_INTRADAY_TIMEFRAME_ALIASES.get(text, text))


def _parse_boolish(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if out == out else float(default)


def _unique_timeframes(primary: object, secondary: object) -> list[str]:
    out: list[str] = []
    for value in [primary, secondary]:
        text = _normalize_timeframe(value)
        if text and text not in out:
            out.append(text)
    return out or ["15m"]


def _parse_source_chain(value: object) -> list[str]:
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return list(_INTRADAY_AUTO_SOURCE_CHAIN)
    out: list[str] = []
    for item in text.split(","):
        token = str(item).strip().lower()
        if not token:
            continue
        if token == "auto":
            for default_item in _INTRADAY_AUTO_SOURCE_CHAIN:
                if default_item not in out:
                    out.append(default_item)
            continue
        if token in {"tushare", "eastmoney", "local"} and token not in out:
            out.append(token)
    return out or list(_INTRADAY_AUTO_SOURCE_CHAIN)


def _candidate_paths(*, data_dir: Path, symbol: str, timeframe: str) -> list[Path]:
    suffixes = [".parquet", ".csv", ".json"]
    paths: list[Path] = []
    for suffix in suffixes:
        paths.extend(
            [
                data_dir / timeframe / f"{symbol}{suffix}",
                data_dir / f"{symbol}_{timeframe}{suffix}",
                data_dir / symbol / f"{timeframe}{suffix}",
                data_dir / symbol / timeframe / f"bars{suffix}",
            ]
        )
    return paths


def _canonical_intraday_cache_path(*, data_dir: Path, symbol: str, timeframe: str) -> Path:
    return data_dir / timeframe / f"{symbol}.csv"


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return pd.DataFrame(payload["items"])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_datetime_column(frame: pd.DataFrame) -> pd.Series:
    cols = {str(col).strip().lower(): col for col in frame.columns}
    for key in ["datetime", "timestamp", "trade_time", "dt", "time"]:
        raw = cols.get(key)
        if raw is not None:
            return pd.to_datetime(frame[raw], errors="coerce")
    date_col = cols.get("date")
    time_col = cols.get("time")
    if date_col is not None and time_col is not None:
        return pd.to_datetime(
            frame[date_col].astype(str).str.strip() + " " + frame[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    if date_col is not None:
        return pd.to_datetime(frame[date_col], errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, index=frame.index)
    return pd.Series(pd.NaT, index=frame.index)


def _normalize_intraday_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "symbol"])
    cols = {str(col).strip().lower(): col for col in frame.columns}
    out = pd.DataFrame(index=frame.index)
    out["datetime"] = _resolve_datetime_column(frame)
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    volume_col = cols.get("volume") or cols.get("vol") or cols.get("trade_vol")
    amount_col = cols.get("amount") or cols.get("trade_amount")
    symbol_col = cols.get("symbol") or cols.get("ts_code")
    if close_col is None:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "symbol"])
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce")
    out["open"] = pd.to_numeric(frame[open_col], errors="coerce") if open_col is not None else out["close"]
    out["high"] = pd.to_numeric(frame[high_col], errors="coerce") if high_col is not None else out["close"]
    out["low"] = pd.to_numeric(frame[low_col], errors="coerce") if low_col is not None else out["close"]
    if volume_col is not None:
        out["volume"] = pd.to_numeric(frame[volume_col], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0
    if amount_col is not None:
        out["amount"] = pd.to_numeric(frame[amount_col], errors="coerce")
    else:
        out["amount"] = out["close"] * out["volume"]
    if symbol_col is not None:
        out["symbol"] = frame[symbol_col].astype(str).str.strip()
    else:
        out["symbol"] = ""
    out = out.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)
    if out.empty:
        return out
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"]).clip(lower=out["close"])
    out["low"] = out["low"].fillna(out["close"]).clip(upper=out["close"])
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    out["amount"] = out["amount"].fillna(out["close"] * out["volume"]).clip(lower=0.0)
    return out


def _merge_intraday_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return _normalize_intraday_frame(right)
    if right.empty:
        return _normalize_intraday_frame(left)
    combined = pd.concat([left, right], ignore_index=True)
    normalized = _normalize_intraday_frame(combined)
    if normalized.empty:
        return normalized
    return normalized.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)


def _target_session_day(as_of_date: str) -> pd.Timestamp:
    target = pd.Timestamp(as_of_date).normalize() if str(as_of_date).strip() else pd.Timestamp.today().normalize()
    return target


def _select_intraday_session(
    *,
    frame: pd.DataFrame,
    as_of_date: str,
    lookback_bars: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    target_day = _target_session_day(as_of_date)
    session_frame = frame.loc[frame["datetime"].dt.normalize() <= target_day].copy()
    if session_frame.empty:
        return session_frame
    latest_day = session_frame["datetime"].dt.normalize().max()
    session_frame = session_frame.loc[session_frame["datetime"].dt.normalize() == latest_day].copy()
    if session_frame.empty:
        return session_frame
    return session_frame.tail(max(8, int(lookback_bars))).reset_index(drop=True)


def _load_intraday_frame_from_disk(
    *,
    data_dir: Path,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    for path in _candidate_paths(data_dir=data_dir, symbol=symbol, timeframe=timeframe):
        if not path.exists():
            continue
        try:
            return _normalize_intraday_frame(_load_frame(path))
        except Exception:
            continue
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "symbol"])


def _write_intraday_frame(
    *,
    data_dir: Path,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
) -> None:
    path = _canonical_intraday_cache_path(data_dir=data_dir, symbol=symbol, timeframe=timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _cache_needs_refresh(
    *,
    frame: pd.DataFrame,
    as_of_date: str,
) -> bool:
    if frame.empty:
        return True
    latest_day = frame["datetime"].dt.normalize().max()
    target_day = _target_session_day(as_of_date)
    return bool(latest_day < target_day)


def _fetch_tushare_intraday(
    *,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: int,
) -> pd.DataFrame:
    info = normalize_symbol(symbol)
    freq = _INTRADAY_TUSHARE_FREQ.get(timeframe)
    if not freq:
        raise DataError(f"{symbol}: unsupported tushare intraday timeframe {timeframe}")
    raw = _post_tushare_sdk_http(
        api_name="stk_mins",
        params={
            "ts_code": info.symbol,
            "freq": freq,
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        },
        fields="ts_code,trade_time,open,close,high,low,vol,amount",
        timeout=max(1, int(timeout)),
    )
    if raw is None or raw.empty:
        raise DataError(f"{symbol}: tushare returned no intraday bars for {timeframe}")
    frame = pd.DataFrame(
        {
            "datetime": raw.get("trade_time", raw.get("datetime")),
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "close": raw.get("close"),
            "volume": raw.get("vol", raw.get("volume")),
            "amount": raw.get("amount"),
            "symbol": info.symbol,
        }
    )
    normalized = _normalize_intraday_frame(frame)
    if normalized.empty:
        raise DataError(f"{symbol}: tushare intraday bars are empty after normalization")
    return normalized


def _fetch_eastmoney_intraday(
    *,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: int,
) -> pd.DataFrame:
    info = normalize_symbol(symbol)
    klt = _INTRADAY_EASTMONEY_KLT.get(timeframe)
    if not klt:
        raise DataError(f"{symbol}: unsupported eastmoney intraday timeframe {timeframe}")
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": klt,
        "fqt": "1",
        "secid": info.secid,
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        response = _HTTP_SESSION.get(EASTMONEY_KLINE_URL, params=params, headers=headers, timeout=max(1, int(timeout)))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise DataError(f"{symbol}: eastmoney intraday request failed: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    lines = data.get("klines") if isinstance(data, dict) else None
    if not lines:
        raise DataError(f"{symbol}: eastmoney returned no intraday bars for {timeframe}")
    rows: list[dict[str, object]] = []
    for line in lines:
        parts = str(line).split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "datetime": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6] if len(parts) > 6 else None,
                "symbol": info.symbol,
            }
        )
    normalized = _normalize_intraday_frame(pd.DataFrame(rows))
    if normalized.empty:
        raise DataError(f"{symbol}: eastmoney intraday bars are empty after normalization")
    return normalized


def _fetch_intraday_frame(
    *,
    symbol: str,
    timeframe: str,
    as_of_date: str,
    source: str,
    fetch_lookback_days: int,
    timeout: int,
) -> pd.DataFrame:
    target_day = _target_session_day(as_of_date)
    start = target_day - pd.Timedelta(days=max(1, int(fetch_lookback_days)))
    end = target_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if source == "tushare":
        return _fetch_tushare_intraday(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            timeout=timeout,
        )
    if source == "eastmoney":
        return _fetch_eastmoney_intraday(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            timeout=timeout,
        )
    raise DataError(f"{symbol}: unsupported intraday source {source}")


def _load_intraday_session(
    *,
    data_dir: Path,
    symbol: str,
    timeframe: str,
    as_of_date: str,
    lookback_bars: int,
    auto_fetch: bool,
    source_chain: list[str],
    fetch_lookback_days: int,
    fetch_timeout_seconds: int,
) -> pd.DataFrame:
    timeframe = _normalize_timeframe(timeframe)
    disk_frame = _load_intraday_frame_from_disk(
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
    )
    if auto_fetch and _cache_needs_refresh(frame=disk_frame, as_of_date=as_of_date):
        remote_errors: list[str] = []
        for source in source_chain:
            if source == "local":
                continue
            try:
                remote = _fetch_intraday_frame(
                    symbol=symbol,
                    timeframe=timeframe,
                    as_of_date=as_of_date,
                    source=source,
                    fetch_lookback_days=fetch_lookback_days,
                    timeout=fetch_timeout_seconds,
                )
            except Exception as exc:
                remote_errors.append(f"{source}: {exc}")
                continue
            if not remote.empty:
                merged = _merge_intraday_frames(disk_frame, remote)
                try:
                    _write_intraday_frame(
                        data_dir=data_dir,
                        symbol=symbol,
                        timeframe=timeframe,
                        frame=merged,
                    )
                except Exception:
                    pass
                disk_frame = merged
                break
        if disk_frame.empty and remote_errors:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "symbol"])
    session = _select_intraday_session(
        frame=disk_frame,
        as_of_date=as_of_date,
        lookback_bars=lookback_bars,
    )
    return session if not session.empty else pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "symbol"])


def _bounded_support(*, last_close: float, session_low: float, ema_slow: float, vwap: float) -> float:
    return min(last_close, max(session_low, min(ema_slow, vwap)))


def _bounded_resistance(*, last_close: float, session_high: float, ema_slow: float, vwap: float) -> float:
    return max(last_close, min(session_high, max(ema_slow, vwap)))


def _assess_single_timeframe(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
) -> IntradayExecutionAssessment | None:
    if frame.empty or len(frame) < 8:
        return None
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    last_close = float(close.iloc[-1])
    session_high = float(high.max())
    session_low = float(low.min())
    if not (last_close == last_close and session_high == session_high and session_low == session_low):
        return None
    session_range = max(session_high - session_low, 1e-9)
    close_pos = float((last_close - session_low) / session_range)
    volume_sum = float(volume.sum())
    if volume_sum > 0.0:
        vwap = float((close * volume).sum() / volume_sum)
    else:
        vwap = float(close.mean())
    fast_span = max(3, min(5, len(close)))
    slow_span = max(6, min(12, len(close)))
    ema_fast = float(close.ewm(span=fast_span, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=slow_span, adjust=False).mean().iloc[-1])
    vwap_gap = float((last_close - vwap) / max(abs(vwap), 1e-9))
    ema_gap = float((ema_fast - ema_slow) / max(abs(ema_slow), 1e-9))
    drawdown_from_high = float((session_high - last_close) / max(abs(session_high), 1e-9))
    support = _bounded_support(last_close=last_close, session_low=session_low, ema_slow=ema_slow, vwap=vwap)
    resistance = _bounded_resistance(last_close=last_close, session_high=session_high, ema_slow=ema_slow, vwap=vwap)
    signal = "hold_neutral"
    break_state = "range"
    reason = (
        f"{timeframe} close {last_close:.2f}, VWAP gap {vwap_gap:+.1%}, "
        f"drawdown {drawdown_from_high:.1%}, close-pos {close_pos:.0%}"
    )
    stop_price = support
    take_profit_price = max(last_close, session_high)
    if close_pos <= 0.28 and vwap_gap <= -0.008 and ema_gap <= -0.0015:
        signal = "exit_on_weak_rebound"
        break_state = "breakdown"
        stop_price = min(last_close, max(session_low, support))
        take_profit_price = max(last_close, min(session_high, max(vwap, ema_slow)))
        reason = (
            f"{timeframe} breakdown: close near session low, below VWAP {abs(vwap_gap):.1%}, "
            f"fast EMA trails slow EMA"
        )
    elif drawdown_from_high >= 0.025 and vwap_gap < -0.004 and ema_gap <= 0.0:
        signal = "reduce_on_bounce"
        break_state = "failed_rebound"
        stop_price = support
        take_profit_price = max(last_close, resistance)
        reason = (
            f"{timeframe} weak rebound: drawdown {drawdown_from_high:.1%} from high and "
            f"close still below VWAP {abs(vwap_gap):.1%}"
        )
    elif close_pos >= 0.68 and vwap_gap >= 0.002 and ema_gap >= 0.001 and drawdown_from_high <= 0.015:
        signal = "hold_strong"
        break_state = "trend_intact"
        stop_price = support
        take_profit_price = max(session_high, last_close * 1.015)
        reason = (
            f"{timeframe} trend intact: close above VWAP {vwap_gap:.1%}, "
            f"drawdown only {drawdown_from_high:.1%}"
        )
    return IntradayExecutionAssessment(
        symbol=symbol,
        signal=signal,
        timeframe=timeframe,
        data_date=str(frame["datetime"].iloc[-1].date()),
        stop_price=float(stop_price),
        take_profit_price=float(take_profit_price),
        vwap_gap=float(vwap_gap),
        drawdown_from_high=float(drawdown_from_high),
        break_state=break_state,
        reason=reason,
    )


def _signal_priority(signal: str) -> int:
    order = {
        "exit_on_weak_rebound": 4,
        "reduce_on_bounce": 3,
        "hold_neutral": 2,
        "hold_strong": 1,
    }
    return int(order.get(str(signal), 0))


def _pick_best_assessment(items: list[IntradayExecutionAssessment]) -> IntradayExecutionAssessment | None:
    if not items:
        return None
    ranked = sorted(
        items,
        key=lambda item: (
            _signal_priority(item.signal),
            abs(float(item.vwap_gap)),
            float(item.drawdown_from_high),
        ),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) == 1:
        return best
    confirmations = [item for item in ranked[1:] if item.signal == best.signal]
    if confirmations:
        confirm = confirmations[0]
        return IntradayExecutionAssessment(
            symbol=best.symbol,
            signal=best.signal,
            timeframe=best.timeframe,
            data_date=best.data_date,
            stop_price=best.stop_price,
            take_profit_price=best.take_profit_price,
            vwap_gap=best.vwap_gap,
            drawdown_from_high=best.drawdown_from_high,
            break_state=best.break_state,
            reason=f"{best.reason}; {confirm.timeframe} confirms {confirm.signal}",
        )
    return best


def build_intraday_execution_overlay(
    *,
    settings: dict[str, object] | None,
    symbols: list[str],
    as_of_date: str = "",
) -> dict[str, IntradayExecutionAssessment]:
    cfg = dict(settings or {})
    if not _parse_boolish(cfg.get("enable_intraday_execution_overlay", True), True):
        return {}
    data_dir = Path(str(cfg.get("intraday_data_dir", "data/intraday"))).expanduser()
    auto_fetch = _parse_boolish(cfg.get("intraday_auto_fetch", True), True)
    if not data_dir.exists() and not auto_fetch:
        return {}
    lookback_bars = max(8, int(cfg.get("intraday_lookback_bars", 64) or 64))
    symbol_limit = max(0, int(cfg.get("intraday_symbol_limit", 20) or 20))
    fetch_lookback_days = max(1, int(cfg.get("intraday_fetch_lookback_days", 5) or 5))
    fetch_timeout_seconds = max(3, int(cfg.get("intraday_fetch_timeout_seconds", 12) or 12))
    source_chain = _parse_source_chain(cfg.get("intraday_source", "auto"))
    timeframes = _unique_timeframes(
        cfg.get("intraday_primary_timeframe", "15m"),
        cfg.get("intraday_secondary_timeframe", "5m"),
    )
    selected_symbols: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip()
        if not symbol or symbol in seen:
            continue
        selected_symbols.append(symbol)
        seen.add(symbol)
        if symbol_limit and len(selected_symbols) >= symbol_limit:
            break
    overlay: dict[str, IntradayExecutionAssessment] = {}
    for symbol in selected_symbols:
        candidates: list[IntradayExecutionAssessment] = []
        for timeframe in timeframes:
            frame = _load_intraday_session(
                data_dir=data_dir,
                symbol=symbol,
                timeframe=timeframe,
                as_of_date=as_of_date,
                lookback_bars=lookback_bars,
                auto_fetch=auto_fetch,
                source_chain=source_chain,
                fetch_lookback_days=fetch_lookback_days,
                fetch_timeout_seconds=fetch_timeout_seconds,
            )
            assessment = _assess_single_timeframe(
                symbol=symbol,
                timeframe=timeframe,
                frame=frame,
            )
            if assessment is not None:
                candidates.append(assessment)
        best = _pick_best_assessment(candidates)
        if best is not None:
            overlay[symbol] = best
    return overlay
