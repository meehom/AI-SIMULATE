from __future__ import annotations

from typing import Any, Dict, List

from ai_simulate.core.op_record import OpRecord
from ai_simulate.roofline.op_costs import get_flops, get_memory


def build_roofline_context(chip_config: Dict[str, Any], precision: str) -> Dict[str, float | str]:
    precision_key = precision.lower()
    peak_compute_tflops = float(chip_config["precision_performance"][f"{precision_key}_tflops"])
    peak_memory_tb_per_s = float(chip_config["memory"]["bandwidth_tb_per_s"])
    peak_compute_flops_per_s = peak_compute_tflops * 1e12
    peak_memory_bytes_per_s = peak_memory_tb_per_s * 1e12
    ridge_point_flops_per_byte = peak_compute_flops_per_s / peak_memory_bytes_per_s
    return {
        "precision": precision_key,
        "peak_compute_tflops": peak_compute_tflops,
        "peak_memory_tb_per_s": peak_memory_tb_per_s,
        "peak_compute_flops_per_s": peak_compute_flops_per_s,
        "peak_memory_bytes_per_s": peak_memory_bytes_per_s,
        "ridge_point_flops_per_byte": ridge_point_flops_per_byte,
    }


def analyze_ops_with_roofline(
    primitive_ops: List[OpRecord],
    chip_config: Dict[str, Any],
    precision: str,
) -> Dict[str, Any]:
    roofline_context = build_roofline_context(chip_config, precision)
    peak_compute_flops_per_s = float(roofline_context["peak_compute_flops_per_s"])
    peak_memory_bytes_per_s = float(roofline_context["peak_memory_bytes_per_s"])

    ops_payload: List[Dict[str, Any]] = []
    total_flops = 0.0
    total_memory_bytes = 0
    total_predicted_time_s = 0.0

    for op_record in primitive_ops:
        flops = get_flops(op_record)
        memory = get_memory(op_record)
        memory_total = memory.total_bytes
        arithmetic_intensity = flops / memory_total if memory_total else 0.0
        compute_time_s = flops / peak_compute_flops_per_s if flops else 0.0
        memory_time_s = memory_total / peak_memory_bytes_per_s if memory_total else 0.0
        predicted_time_s = max(compute_time_s, memory_time_s)
        bottleneck = "compute" if compute_time_s >= memory_time_s else "memory"

        op_record.metrics = {
            "flops": flops,
            "memory_bytes_read": memory.read_bytes,
            "memory_bytes_written": memory.write_bytes,
            "memory_bytes_total": memory_total,
            "arithmetic_intensity": arithmetic_intensity,
            "compute_time_s": compute_time_s,
            "memory_time_s": memory_time_s,
            "predicted_time_s": predicted_time_s,
            "bottleneck": bottleneck,
            "estimator_status": "ok",
        }
        ops_payload.append(op_record.to_dict())

        total_flops += flops
        total_memory_bytes += memory_total
        total_predicted_time_s += predicted_time_s

    return {
        "roofline_context": roofline_context,
        "ops": ops_payload,
        "summary": {
            "captured_op_count": len(primitive_ops),
            "unsupported_op_count": 0,
            "total_flops": total_flops,
            "total_memory_bytes": total_memory_bytes,
            "total_predicted_time_s": total_predicted_time_s,
        },
    }
