from __future__ import annotations

import argparse
import json

from src.application.v2_services import run_daily_v2, summarize_daily_run
from src.application.v2_workflow import (
    build_daily_run_blueprint,
    build_research_run_blueprint,
    describe_v2_stack,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant architecture V2 CLI scaffold")
    sub = parser.add_subparsers(dest="task", required=True)

    sub.add_parser("describe", help="Print the V2 architecture summary")

    daily = sub.add_parser("daily-run", help="Print the V2 production workflow stages")
    daily.add_argument("--strategy", default="swing_v2", help="Strategy snapshot id")

    research = sub.add_parser("research-run", help="Print the V2 research workflow stages")
    research.add_argument("--strategy", default="swing_v2", help="Target strategy id")

    return parser


def _print_blueprint(title: str, strategy: str, stages: list[tuple[str, str, str]]) -> None:
    print(f"[V2] {title}: strategy={strategy}")
    for idx, (name, purpose, produces) in enumerate(stages, start=1):
        print(f"{idx}. {name}")
        print(f"   purpose: {purpose}")
        print(f"   produces: {produces}")


def main() -> int:
    args = build_parser().parse_args()

    if args.task == "describe":
        print(describe_v2_stack())
        print("Use `research-run` to build artifacts and `daily-run` to consume published strategy snapshots.")
        return 0

    if args.task == "daily-run":
        bp = build_daily_run_blueprint()
        _print_blueprint(
            bp.name,
            str(args.strategy),
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        result = run_daily_v2(strategy_id=str(args.strategy))
        print("[V2] daily-run summary:")
        print(json.dumps(summarize_daily_run(result), ensure_ascii=False, indent=2))
        return 0

    if args.task == "research-run":
        bp = build_research_run_blueprint()
        _print_blueprint(
            bp.name,
            str(args.strategy),
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        print("[V2] scaffold only: wire dataset, training, and backtest jobs into these stages next.")
        return 0

    raise ValueError(f"Unsupported task: {args.task}")
