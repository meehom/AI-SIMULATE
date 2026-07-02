from __future__ import annotations

from typing import Any, Dict, List

from ai_simulate.core.op_record import OpRecord
from ai_simulate.roofline.estimator import analyze_ops_with_roofline
from ai_simulate.workload import capture_model_ops


def analyze_model_with_meta_hooks(
    model: Any,
    input_shape: List[int],
    chip_config: Dict[str, Any],
    logical_precision: str,
    analysis_phase: str,
    strategy_config: Dict[str, Any],
) -> Dict[str, Any]:
    op_records: List[OpRecord] = capture_model_ops(
        model=model,
        input_shape=input_shape,
        precision=logical_precision,
        strategy_config=strategy_config,
    )
    result = analyze_ops_with_roofline(
        primitive_ops=op_records,
        chip_config=chip_config,
        precision=logical_precision,
    )
    return {
        "phase": analysis_phase,
        "logical_precision": logical_precision,
        "roofline_context": result["roofline_context"],
        "ops": result["ops"],
        "summary": result["summary"],
    }
