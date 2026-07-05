from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Dict

from ai_simulate.core.op_record import OpRecord, bytes_per_precision, shape_numel
from ai_simulate.custom import get_custom_estimator


ACTIVATION_FLOP_COST = {
    "aten.gelu.default": 8.0,
}


@dataclass(frozen=True)
class MemoryStats:
    read_bytes: int
    write_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


@dataclass(frozen=True)
class OperatorSpec:
    get_flops: Callable[[OpRecord], float]
    get_memory: Callable[[OpRecord], MemoryStats]


REGISTRY: Dict[str, OperatorSpec] = {}


class UnsupportedEstimatorError(ValueError):
    """Raised when no cost estimator is registered for an intercepted op."""


def register_operator(op_name: str, spec: OperatorSpec) -> None:
    REGISTRY[op_name] = spec


def get_flops(op_record: OpRecord) -> float:
    if op_record.op_name.startswith("custom."):
        estimator = get_custom_estimator(op_record.op_name)
        return float(estimator["get_flops"](op_record))
    if op_record.op_name not in REGISTRY:
        raise UnsupportedEstimatorError(f"No FLOPs estimator registered for {op_record.op_name}")
    return REGISTRY[op_record.op_name].get_flops(op_record)


def get_memory(op_record: OpRecord) -> MemoryStats:
    if op_record.op_name.startswith("custom."):
        estimator = get_custom_estimator(op_record.op_name)
        stats = estimator["get_memory"](op_record)
        return MemoryStats(read_bytes=stats.read_bytes, write_bytes=stats.write_bytes)
    if op_record.op_name not in REGISTRY:
        raise UnsupportedEstimatorError(f"No memory estimator registered for {op_record.op_name}")
    return REGISTRY[op_record.op_name].get_memory(op_record)


def _tensor_bytes(numel: int, precision: str) -> int:
    return int(numel) * bytes_per_precision(precision)


def _addmm_flops(op_record: OpRecord) -> float:
    activations = op_record.local_input_tensors[1]
    weights = op_record.local_input_tensors[2]
    batch_tokens = int(prod(activations.shape[:-1]))
    in_features = int(activations.shape[-1])
    out_features = int(weights.shape[0])
    return float(2 * batch_tokens * in_features * out_features)


def _addmm_memory(op_record: OpRecord) -> MemoryStats:
    precision = op_record.precision_context["storage_precision"]
    bias, activations, weights = op_record.local_input_tensors
    output = op_record.local_output_tensors[0]
    read_bytes = _tensor_bytes(bias.numel, precision)
    read_bytes += _tensor_bytes(activations.numel, precision)
    read_bytes += _tensor_bytes(weights.numel, precision)
    write_bytes = _tensor_bytes(output.numel, precision)
    return MemoryStats(read_bytes=read_bytes, write_bytes=write_bytes)


def _layer_norm_flops(op_record: OpRecord) -> float:
    output = op_record.local_output_tensors[0]
    return float(7.0 * output.numel)


def _layer_norm_memory(op_record: OpRecord) -> MemoryStats:
    precision = op_record.precision_context["storage_precision"]
    input_tensor = op_record.local_input_tensors[0]
    weight = op_record.local_input_tensors[1]
    bias = op_record.local_input_tensors[2]
    output = op_record.local_output_tensors[0]
    read_bytes = _tensor_bytes(input_tensor.numel, precision)
    read_bytes += _tensor_bytes(weight.numel, precision)
    read_bytes += _tensor_bytes(bias.numel, precision)
    write_bytes = _tensor_bytes(output.numel, precision)
    return MemoryStats(read_bytes=read_bytes, write_bytes=write_bytes)


def _gelu_flops(op_record: OpRecord) -> float:
    output = op_record.local_output_tensors[0]
    return float(ACTIVATION_FLOP_COST[op_record.op_name] * output.numel)


def _gelu_memory(op_record: OpRecord) -> MemoryStats:
    precision = op_record.precision_context["storage_precision"]
    input_tensor = op_record.local_input_tensors[0]
    output = op_record.local_output_tensors[0]
    read_bytes = _tensor_bytes(input_tensor.numel, precision)
    write_bytes = _tensor_bytes(output.numel, precision)
    return MemoryStats(read_bytes=read_bytes, write_bytes=write_bytes)


register_operator("aten.addmm.default", OperatorSpec(get_flops=_addmm_flops, get_memory=_addmm_memory))
register_operator(
    "aten.native_layer_norm.default",
    OperatorSpec(get_flops=_layer_norm_flops, get_memory=_layer_norm_memory),
)
register_operator("aten.gelu.default", OperatorSpec(get_flops=_gelu_flops, get_memory=_gelu_memory))
