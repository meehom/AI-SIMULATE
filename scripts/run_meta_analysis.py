#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ai_simulate.simulator import run_end_to_end_experiment, run_meta_analysis_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run meta + hook analysis for an experiment config")
    parser.add_argument("--experiment", required=True, help="Path to an experiment config JSON file")
    parser.add_argument(
        "--phase",
        default="prefill",
        choices=["prefill", "decode", "end_to_end"],
        help="Inference phase or combined experiment summary to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "end_to_end":
        summary = run_end_to_end_experiment(Path(args.experiment))
        metrics = summary["request_level_metrics"]
        print(f"Experiment: {summary['experiment_name']}")
        print("Phase: end_to_end")
        print(f"Summary: {summary['summary_output_path']}")
        print(
            "TTFT={ttft:.6e}s, ITL={itl:.6e}s, Total={total:.6e}s, Decode throughput={decode_tp:.3f} tok/s, Request throughput={req_tp:.3f} tok/s".format(
                ttft=metrics["time_to_first_token_s"],
                itl=metrics["inter_token_latency_s"],
                total=metrics["request_total_time_s"],
                decode_tp=metrics["decode_throughput_tokens_per_s"],
                req_tp=metrics["request_throughput_tokens_per_s"],
            )
        )
        return

    result = run_meta_analysis_experiment(Path(args.experiment), phase=args.phase)
    summary = result["analysis"]["summary"]
    print(f"Experiment: {result['experiment_name']}")
    print(f"Phase: {args.phase}")
    print(f"Output: {result['output_path']}")
    print(f"CSV: {result['op_csv_path']}")
    print(f"Trace: {result['trace_output_path']}")
    if args.phase == "decode":
        decode_estimate = result["analysis"]["decode_estimate"]
        print(
            "Decode estimate: per-step={per_step:.6e}s, total={total:.6e}s over {steps} steps".format(
                per_step=decode_estimate["per_step_predicted_time_s"],
                total=decode_estimate["estimated_total_decode_time_s"],
                steps=decode_estimate["estimated_output_steps"],
            )
        )
    print(
        "Captured ops: {op_count}, Total FLOPs: {total_flops:.2f}, Total predicted time (s): {total_time:.6e}".format(
            op_count=summary["captured_op_count"],
            total_flops=summary["total_flops"],
            total_time=summary["total_predicted_time_s"],
        )
    )


if __name__ == "__main__":
    main()
