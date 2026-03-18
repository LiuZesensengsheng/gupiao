from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

from src.application.v2_contracts import CompositeState, InfoAggregateState, InfoItem, StockForecastState, V2BacktestSummary


def stable_json_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path_like: object) -> str:
    path = Path(str(path_like))
    if not path.exists():
        return ""
    h = hashlib.sha256()
    if path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = str(file.relative_to(path)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            with file.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        return h.hexdigest()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compose_shadow_stock_score(
    *,
    stock: StockForecastState,
    info_state: InfoAggregateState,
) -> float:
    return float(
        0.15 * float(info_state.shadow_prob_1d)
        + 0.25 * float(info_state.shadow_prob_5d)
        + 0.60 * float(info_state.shadow_prob_20d)
        + 0.08 * float(stock.tradeability_score)
        - 0.08 * float(info_state.negative_event_risk)
    )


def build_sector_map_from_state(state: CompositeState) -> dict[str, str]:
    return {str(stock.symbol): str(stock.sector) for stock in state.stocks}


def _write_console_line(text: str) -> None:
    stream = sys.stdout
    line = f"{text}\n"
    try:
        stream.write(line)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_line)
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def emit_progress(stage: str, message: str) -> None:
    _write_console_line(f"[V2][{stage}] {message}")


def trajectory_step_count(trajectory: object) -> int:
    steps = getattr(trajectory, "steps", [])
    try:
        return int(len(steps))
    except TypeError:
        return 0


def format_elapsed(seconds: float) -> str:
    secs = max(0, int(round(float(seconds))))
    minutes, remain = divmod(secs, 60)
    if minutes <= 0:
        return f"{remain}s"
    return f"{minutes}m{remain:02d}s"


def policy_objective_score(summary: V2BacktestSummary) -> float:
    excess_alpha = float(summary.excess_annual_return)
    ir_term = float(summary.information_ratio)
    drawdown_penalty = 0.80 * abs(float(summary.max_drawdown))
    turnover_penalty = 0.40 * float(summary.avg_turnover)
    cost_penalty = 0.60 * float(summary.total_cost)
    score = excess_alpha + ir_term - drawdown_penalty - turnover_penalty - cost_penalty
    return float(score)


def filter_info_items_by_source_subset(items: Iterable[InfoItem], subset: str) -> list[InfoItem]:
    target = str(subset).strip()
    if not target:
        return list(items)
    return [item for item in items if str(getattr(item, "source_subset", "")) == target]
