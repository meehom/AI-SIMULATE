from __future__ import annotations

from typing import Dict

from .cost_model import estimate_fc2_flops, estimate_fc2_memory


CUSTOM_ESTIMATORS = {
    "custom.fc2.default": {
        "get_flops": estimate_fc2_flops,
        "get_memory": estimate_fc2_memory,
    }
}


def has_custom_estimator(op_name: str) -> bool:
    return op_name in CUSTOM_ESTIMATORS


def get_custom_estimator(op_name: str) -> Dict[str, object]:
    if op_name not in CUSTOM_ESTIMATORS:
        raise ValueError(f"No custom estimator registered for {op_name}")
    return CUSTOM_ESTIMATORS[op_name]
