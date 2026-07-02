from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Dict

from ai_simulate.core.op_record import OpRecord, bytes_per_precision, shape_numel


ACTIVATION_FLOP_COST = {
    "gelu": 8.0,
    "relu": 1.0,
}

NORM_FLOP_COST = {
    "layer_norm": 7.0,
    "rms_norm": 5.0,
}


@dataclass(frozen=True)
class MemoryStats:
    read_bytes: int
    write_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    def to_dict(self) -> Dict[str, int]:
        return {
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class OperatorSpec:
    get_flops: Callable[[OpRecord], float]
    get_memory: Callable[[OpRecord], MemoryStats]


REGISTRY: Dict[str, OperatorSpec] = {}


def register_operator(op_type: str, spec: OperatorSpec) -> None:
    REGISTRY[op_type] = spec


def get_flops(op_record: OpRecord) -> float:
    return REGISTRY[op_record.op_type].get_flops(op_record)


def get_memory(op_record: OpRecord) -> MemoryStats:
    return REGISTRY[op_record.op_type].get_memory(op_record)


def _tensor_bytes(shape: list[int] | None, precision: str) -> int:
    return shape_numel(shape) * bytes_per_precision(precision)


def _linear_flops(op_record: OpRecord) -> float:
    input_shape = op_record.local_shapes["input"]
    output_shape = op_record.local_shapes["output"]
    batch_tokens = int(prod(input_shape[:-1]))
    in_features = int(input_shape[-1])
    out_features = int(output_shape[-1])
    return float(2 * batch_tokens * in_features * out_features)


def _linear_memory(op_record: OpRecord) -> MemoryStats:
    precision = op_record.precision
    read_bytes = _tensor_bytes(op_record.local_shapes.get("input"), precision)
    read_bytes += _tensor_bytes(op_record.local_shapes.get("weight"), precision)
    if "bias" in op_record.local_shapes:
        read_bytes += _tensor_bytes(op_record.local_shapes.get("bias"), precision)
    write_bytes = _tensor_bytes(op_record.local_shapes.get("output"), precision)
    return MemoryStats(read_bytes=read_bytes, write_bytes=write_bytes)


def _norm_flops(op_record: OpRecord) -> float:
    output_shape = op_record.local_shapes["output"]
    op_type = op_record.op_type
    flop_cost = NORM_FLOP_COST[op_type]
    return float(flop_cost * shape_numel(output_shape))


def _norm_memory(op_record: OpRecord) -> MemoryStats:
    precision = op_record.precision
    read_bytes = _tensor_bytes(op_record.local_shapes.get("input"), precision)
    read_bytes += _tensor_bytes(op_record.local_shapes.get("weight"), precision)
    read_bytes += _tensor_bytes(op_record.local_shapes.get("bias"), precision)
    write_bytes = _tensor_bytes(op_record.local_shapes.get("output"), precision)
    return MemoryStats(read_bytes=read_bytes, write_bytes=write_bytes)


def _activation_flops(op_record: OpRecord) -> float:
    activation_kind = str(op_record.attrs.get("activation_kind", "gelu")).lower()
    if activation_kind not in ACTIVATION_FLOP_COST:
        raise ValueError(f"Unsupported activation kind: {activation_kind}")
    return float(ACTIVATION_FLOP_COST[activation_kind] * shape_numel(op_record.local_shapes["output"]))


def _activation_memory(op_record: OpRecord) -> MemoryStats:
    precision = op_record.precision
    read_bytes = _tensor_bytes(op_record.local_shapes.get("input"), precision)
    write_bytes = _tensor_bytes(op_record.local_shapes.get("output"), precision)
    return MemoryStats(read_bytes=read_bytes, write_bytes=write_bytes)


register_operator("linear", OperatorSpec(get_flops=_linear_flops, get_memory=_linear_memory))
register_operator("layer_norm", OperatorSpec(get_flops=_norm_flops, get_memory=_norm_memory))
register_operator("activation", OperatorSpec(get_flops=_activation_flops, get_memory=_activation_memory))
