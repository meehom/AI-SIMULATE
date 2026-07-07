#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

EXPERIMENT=configs/experiments/gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512.json

PYTHONPATH=src python scripts/run_meta_analysis.py --experiment "$EXPERIMENT" --phase prefill
PYTHONPATH=src python scripts/run_meta_analysis.py --experiment "$EXPERIMENT" --phase decode
