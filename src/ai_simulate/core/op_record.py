from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Dict, List


BYTES_PER_PRECISION = {
    "fp8": 1,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}


@dataclass
class TensorMetadata:
    shape: List[int]
    dtype: str
    numel: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "numel": self.numel,
            "device": self.device,
        }


@dataclass
class OpRecord:
    op_index: int
    op_name: str
    op_kind: str = "builtin"
    module_path: str | None = None
    precision_context: Dict[str, str] = field(default_factory=dict)
    input_tensors: List[TensorMetadata] = field(default_factory=list)
    output_tensors: List[TensorMetadata] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    parallelism: Dict[str, Any] = field(default_factory=dict)
    local_input_tensors: List[TensorMetadata] = field(default_factory=list)
    local_output_tensors: List[TensorMetadata] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_index": self.op_index,
            "op_name": self.op_name,
            "op_kind": self.op_kind,
            "module_path": self.module_path,
            "precision_context": self.precision_context,
            "input_tensors": [tensor.to_dict() for tensor in self.input_tensors],
            "output_tensors": [tensor.to_dict() for tensor in self.output_tensors],
            "attrs": self.attrs,
            "parallelism": self.parallelism,
            "local_input_tensors": [tensor.to_dict() for tensor in self.local_input_tensors],
            "local_output_tensors": [tensor.to_dict() for tensor in self.local_output_tensors],
            "metrics": self.metrics,
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


def shape_numel(shape: List[int] | None) -> int:
    if not shape:
        return 0
    return int(prod(shape))


def bytes_per_precision(precision: str) -> int:
    precision_key = precision.lower()
    if precision_key not in BYTES_PER_PRECISION:
        raise ValueError(f"Unsupported logical precision: {precision}")
    return BYTES_PER_PRECISION[precision_key]
