#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ai_simulate.simulator import run_meta_analysis_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run meta + hook analysis for an experiment config")
    parser.add_argument("--experiment", required=True, help="Path to an experiment config JSON file")
    parser.add_argument("--phase", default="prefill", choices=["prefill"], help="Analysis phase to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_meta_analysis_experiment(Path(args.experiment), phase=args.phase)
    summary = result["analysis"]["summary"]
    print(f"Experiment: {result['experiment_name']}")
    print(f"Output: {result['output_path']}")
    print(
        "Captured ops: {op_count}, Total FLOPs: {total_flops:.2f}, Total predicted time (s): {total_time:.6e}".format(
            op_count=summary["captured_op_count"],
            total_flops=summary["total_flops"],
            total_time=summary["total_predicted_time_s"],
        )
    )


if __name__ == "__main__":
    main()
