from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ai_simulate.analysis import analyze_model_with_meta_hooks
from ai_simulate.core import ResolvedExperiment, load_experiment
from ai_simulate.workload import build_deepseek_v3_proxy


class UnsupportedModelError(ValueError):
    """Raised when no proxy implementation exists for a requested model."""


def _select_proxy_builder(model_name: str):
    normalized = model_name.strip().lower()
    if normalized == "deepseek v3":
        return build_deepseek_v3_proxy
    raise UnsupportedModelError(f"No proxy implementation registered for model: {model_name}")


def _resolved_paths_payload(resolved_experiment: ResolvedExperiment) -> Dict[str, str]:
    return {
        "experiment": str(resolved_experiment.experiment_path),
        "chip_config": str(resolved_experiment.chip_config_path),
        "topo_config": str(resolved_experiment.topo_config_path),
        "workload_config": str(resolved_experiment.workload_config_path),
        "strategy_config": str(resolved_experiment.strategy_config_path),
    }


def _build_result_payload(
    resolved_experiment: ResolvedExperiment,
    analysis_phase: str,
    analysis_result: Dict[str, Any],
    proxy_input_spec: Any,
) -> Dict[str, Any]:
    analysis_payload = {
        **analysis_result,
        "model_proxy_type": "deepseek_v3_mla_moe_proxy",
        "proxy_input_spec": {
            "shape": proxy_input_spec.shape,
            "dtype": proxy_input_spec.dtype,
            "input_kind": proxy_input_spec.input_kind,
            "hidden_size": proxy_input_spec.hidden_size,
            "intermediate_size": proxy_input_spec.intermediate_size,
            "activation": proxy_input_spec.activation,
            "num_layers": proxy_input_spec.num_layers,
            "analysis_phase": proxy_input_spec.analysis_phase,
            "kv_cache_seq_len": proxy_input_spec.kv_cache_seq_len,
        },
        "recorded_output_seq_len": resolved_experiment.workload_config["output_seq_len"],
    }
    if analysis_phase == "decode":
        per_step_time = analysis_result["summary"]["total_predicted_time_s"]
        output_seq_len = int(resolved_experiment.workload_config["output_seq_len"])
        analysis_payload["decode_estimate"] = {
            "per_step_predicted_time_s": per_step_time,
            "estimated_total_decode_time_s": per_step_time * output_seq_len,
            "estimated_output_steps": output_seq_len,
        }

    return {
        "experiment_name": resolved_experiment.experiment_config["name"],
        "experiment_path": str(resolved_experiment.experiment_path),
        "resolved_config_paths": _resolved_paths_payload(resolved_experiment),
        "system_config": {
            "chip": resolved_experiment.chip_config,
            "topology": resolved_experiment.topo_config,
        },
        "workload_config": resolved_experiment.workload_config,
        "strategy_config": resolved_experiment.strategy_config,
        "analysis": analysis_payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "This prototype uses a structural DeepSeek V3 MLA + MoE proxy.",
            f"Analysis phase is restricted to {analysis_phase} semantics in v1.",
            "Current roofline timing is based on intercepted built-in PyTorch ops under dispatch capture.",
        ],
    }


def _shape_string(tensors: list[Dict[str, Any]]) -> str:
    if not tensors:
        return ""
    return json.dumps(tensors[0]["shape"])


def _bottleneck_label(raw: str) -> str:
    return "compute_bound" if raw == "compute" else "memory_bound"


def _write_operator_csv(output_dir: Path, analysis_result: Dict[str, Any], phase: str) -> Path:
    csv_path = output_dir / f"op_trace_{phase}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "op_index",
            "op_name",
            "op_kind",
            "op_time",
            "analysis",
            "flops",
            "memory_bytes_total",
            "arithmetic_intensity",
            "input_shape",
            "output_shape",
        ])
        for op in analysis_result["ops"]:
            metrics = op["metrics"]
            writer.writerow(
                [
                    op["op_index"],
                    op["op_name"],
                    op["op_kind"],
                    metrics["predicted_time_s"],
                    _bottleneck_label(metrics["bottleneck"]),
                    metrics["flops"],
                    metrics["memory_bytes_total"],
                    metrics["arithmetic_intensity"],
                    _shape_string(op.get("input_tensors", [])),
                    _shape_string(op.get("output_tensors", [])),
                ]
            )
    return csv_path


def _trace_shape(tensors: list[Dict[str, Any]]) -> list[int] | None:
    if not tensors:
        return None
    return tensors[0].get("shape")


def _write_chrome_trace(output_dir: Path, result_payload: Dict[str, Any], phase: str) -> Path:
    trace_path = output_dir / f"chrome_trace_{phase}.json"
    analysis = result_payload["analysis"]
    trace_events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": result_payload["experiment_name"]},
        },
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": f"predicted_op_timeline_{phase}"},
        },
    ]

    cursor_us = 0.0
    for op in analysis["ops"]:
        metrics = op["metrics"]
        raw_dur_us = float(metrics["predicted_time_s"]) * 1e6
        dur_us = raw_dur_us if raw_dur_us > 0 else 1.0
        trace_events.append(
            {
                "name": op["op_name"],
                "cat": "operator",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": cursor_us,
                "dur": dur_us,
                "args": {
                    "phase": phase,
                    "op_index": op["op_index"],
                    "op_kind": op["op_kind"],
                    "predicted_time_s": metrics["predicted_time_s"],
                    "compute_time_s": metrics["compute_time_s"],
                    "memory_time_s": metrics["memory_time_s"],
                    "bottleneck": _bottleneck_label(metrics["bottleneck"]),
                    "flops": metrics["flops"],
                    "memory_bytes_total": metrics["memory_bytes_total"],
                    "arithmetic_intensity": metrics["arithmetic_intensity"],
                    "input_shape": _trace_shape(op.get("input_tensors", [])),
                    "output_shape": _trace_shape(op.get("output_tensors", [])),
                    "local_input_shape": _trace_shape(op.get("local_input_tensors", [])),
                    "local_output_shape": _trace_shape(op.get("local_output_tensors", [])),
                    "precision_context": op.get("precision_context", {}),
                    "parallelism": op.get("parallelism", {}),
                    "attrs": op.get("attrs", {}),
                },
            }
        )
        cursor_us += dur_us

    trace_payload = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
    }
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(trace_payload, handle, indent=2, ensure_ascii=False)
    return trace_path


def run_meta_analysis_experiment(experiment_path: str | Path, phase: str = "prefill") -> Dict[str, Any]:
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"Unsupported analysis phase for v1: {phase}")

    resolved_experiment = load_experiment(experiment_path)
    proxy_builder = _select_proxy_builder(resolved_experiment.workload_config["model_name"])
    model, proxy_input_spec = proxy_builder(
        resolved_experiment.workload_config,
        phase=phase,
    )
    analysis_result = analyze_model_with_meta_hooks(
        model=model,
        input_shape=proxy_input_spec.shape,
        chip_config=resolved_experiment.chip_config,
        logical_precision=resolved_experiment.workload_config["precision"],
        analysis_phase=phase,
        strategy_config=resolved_experiment.strategy_config,
    )

    result_payload = _build_result_payload(
        resolved_experiment=resolved_experiment,
        analysis_phase=phase,
        analysis_result=analysis_result,
        proxy_input_spec=proxy_input_spec,
    )

    output_dir = resolved_experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"meta_hook_analysis_{phase}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    csv_path = _write_operator_csv(output_dir, analysis_result, phase)
    trace_path = _write_chrome_trace(output_dir, result_payload, phase)

    result_payload["output_path"] = str(output_path)
    result_payload["op_csv_path"] = str(csv_path)
    result_payload["trace_output_path"] = str(trace_path)
    return result_payload
