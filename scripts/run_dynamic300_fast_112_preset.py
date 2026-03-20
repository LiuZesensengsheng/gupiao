from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_dynamic300_rule_plan import run_fast_backtest


_PRESET_NAME = "dynamic300_fast_112"
_PRESET_PARAMS: dict[str, Any] = {
    "config_path": "config/api.json",
    "lookback_trading_days": 120,
    "rebalance_interval": 10,
    "top_n": 4,
    "max_per_theme": 1,
    "benchmark_symbol": "000300.SH",
    "commission_bps": 1.5,
    "slippage_bps": 2.0,
    "workers": 4,
}


def _write_latest_alias(*, source_path: Path, target_name: str) -> Path:
    target_path = ROOT / "reports" / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    return target_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the historical 112% fast dynamic300 preset.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--config", default=str(_PRESET_PARAMS["config_path"]))
    parser.add_argument("--workers", type=int, default=int(_PRESET_PARAMS["workers"]))
    args = parser.parse_args()

    outputs = run_fast_backtest(
        config_path=str(args.config),
        end_date=str(args.end_date),
        lookback_trading_days=int(_PRESET_PARAMS["lookback_trading_days"]),
        rebalance_interval=int(_PRESET_PARAMS["rebalance_interval"]),
        top_n=int(_PRESET_PARAMS["top_n"]),
        max_per_theme=int(_PRESET_PARAMS["max_per_theme"]),
        benchmark_symbol=str(_PRESET_PARAMS["benchmark_symbol"]),
        commission_bps=float(_PRESET_PARAMS["commission_bps"]),
        slippage_bps=float(_PRESET_PARAMS["slippage_bps"]),
        workers=max(1, int(args.workers)),
    )

    json_path = Path(outputs["json"])
    markdown_path = Path(outputs["markdown"])
    result = json.loads(json_path.read_text(encoding="utf-8"))
    result["preset"] = {
        "name": _PRESET_NAME,
        "params": {
            **_PRESET_PARAMS,
            "config_path": str(args.config),
            "end_date": str(args.end_date),
            "workers": max(1, int(args.workers)),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_json = _write_latest_alias(
        source_path=json_path,
        target_name="dynamic300_fast_112_latest.json",
    )
    latest_markdown = _write_latest_alias(
        source_path=markdown_path,
        target_name="dynamic300_fast_112_latest.md",
    )

    print(f"[fast-112] preset={_PRESET_NAME}")
    print(f"[fast-112] json={json_path.resolve()}")
    print(f"[fast-112] markdown={markdown_path.resolve()}")
    print(f"[fast-112] latest_json={latest_json}")
    print(f"[fast-112] latest_markdown={latest_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
