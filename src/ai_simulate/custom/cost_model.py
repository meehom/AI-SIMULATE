from __future__ import annotations

from dataclasses import dataclass

from ai_simulate.core.op_record import OpRecord, bytes_per_precision, shape_numel


@dataclass(frozen=True)
class CostStats:
    flops: float
    read_bytes: int
    write_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


def estimate_fc2_flops(op: OpRecord) -> float:
    x = op.local_input_tensors[0]
    weight = op.local_input_tensors[1]
    batch_tokens = shape_numel(x.shape[:-1])
    in_features = x.shape[-1]
    out_features = weight.shape[0]
    return float(2 * batch_tokens * in_features * out_features)


def estimate_fc2_memory(op: OpRecord) -> CostStats:
    precision = op.precision_context["storage_precision"]
    x, weight, bias = op.local_input_tensors
    output = op.local_output_tensors[0]
    read_bytes = x.numel * bytes_per_precision(precision)
    read_bytes += weight.numel * bytes_per_precision(precision)
    read_bytes += bias.numel * bytes_per_precision(precision)
    write_bytes = output.numel * bytes_per_precision(precision)
    return CostStats(flops=estimate_fc2_flops(op), read_bytes=read_bytes, write_bytes=write_bytes)
