from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ai_simulate.core.op_record import OpRecord, TensorMetadata, shape_numel


SUPPORTED_CAPTURE_OPS = {
    "aten.native_layer_norm.default",
    "aten.addmm.default",
    "aten.gelu.default",
}

IGNORED_CAPTURE_OPS = {
    "aten.view.default",
    "aten.t.default",
}


@dataclass
class CapturedOpEvent:
    op_name: str
    input_tensors: List[TensorMetadata]
    output_tensors: List[TensorMetadata]
    attrs: Dict[str, Any]
    module_path: str | None = None


class UnsupportedCapturedOpError(ValueError):
    """Raised when dispatch capture sees an operator without estimator coverage."""


class TorchOpCaptureMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.events: List[CapturedOpEvent] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        op_name = str(func)

        if op_name in SUPPORTED_CAPTURE_OPS:
            self.events.append(
                CapturedOpEvent(
                    op_name=op_name,
                    input_tensors=_extract_tensor_metadata(args),
                    output_tensors=_extract_output_metadata(output),
                    attrs=_extract_relevant_attrs(op_name, args, kwargs),
                )
            )
        elif op_name not in IGNORED_CAPTURE_OPS:
            raise UnsupportedCapturedOpError(f"Unsupported captured operator: {op_name}")

        return output


def _extract_tensor_metadata(values: Iterable[Any]) -> List[TensorMetadata]:
    tensors: List[TensorMetadata] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(_tensor_metadata(value))
    return tensors


def _extract_output_metadata(output: Any) -> List[TensorMetadata]:
    if isinstance(output, torch.Tensor):
        return [_tensor_metadata(output)]
    if isinstance(output, tuple):
        return [_tensor_metadata(value) for value in output if isinstance(value, torch.Tensor)]
    return []


def _tensor_metadata(tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        numel=shape_numel(list(tensor.shape)),
        device=str(tensor.device),
    )


def _extract_relevant_attrs(op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if op_name == "aten.native_layer_norm.default":
        normalized_shape = list(args[1]) if len(args) > 1 else []
        eps = args[4] if len(args) > 4 else kwargs.get("eps")
        return {"normalized_shape": normalized_shape, "eps": eps}
    return {}


def _precision_context(precision: str) -> Dict[str, str]:
    precision_key = precision.lower()
    accum_precision = "fp16" if precision_key == "fp8" else precision_key
    return {
        "storage_precision": precision_key,
        "compute_precision": precision_key,
        "accum_precision": accum_precision,
    }


def _parallelism_payload(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tp_degree": int(strategy_config["tp_degree"]),
        "pp_degree": int(strategy_config["pp_degree"]),
        "dp_degree": int(strategy_config["dp_degree"]),
    }


def _tp_scaled_shape(shape: List[int], tp_degree: int, mode: str) -> List[int]:
    scaled = list(shape)
    if not scaled or tp_degree == 1:
        return scaled
    if mode == "column":
        if scaled[-1] % tp_degree != 0:
            raise ValueError(f"Shape {shape} last dim must be divisible by tp_degree={tp_degree}")
        scaled[-1] //= tp_degree
        return scaled
    if mode == "row":
        if scaled[-1] % tp_degree != 0:
            raise ValueError(f"Shape {shape} last dim must be divisible by tp_degree={tp_degree}")
        scaled[-1] //= tp_degree
        return scaled
    return scaled


def _localize_event(
    event: CapturedOpEvent,
    strategy_config: Dict[str, Any],
) -> tuple[List[TensorMetadata], List[TensorMetadata], Dict[str, Any]]:
    tp_degree = int(strategy_config["tp_degree"])
    tp_mode = "replicated"

    if event.op_name == "aten.addmm.default":
        bias, activations, weights = event.input_tensors
        if event.output_tensors[0].shape[-1] >= activations.shape[-1]:
            tp_mode = "column"
            local_inputs = [
                TensorMetadata(shape=[bias.shape[0] // tp_degree], dtype=bias.dtype, numel=bias.numel // tp_degree, device=bias.device),
                activations,
                TensorMetadata(
                    shape=[weights.shape[0], weights.shape[1] // tp_degree],
                    dtype=weights.dtype,
                    numel=weights.numel // tp_degree,
                    device=weights.device,
                ),
            ]
            local_outputs = [
                TensorMetadata(
                    shape=_tp_scaled_shape(event.output_tensors[0].shape, tp_degree, "column"),
                    dtype=event.output_tensors[0].dtype,
                    numel=event.output_tensors[0].numel // tp_degree,
                    device=event.output_tensors[0].device,
                )
            ]
        else:
            tp_mode = "row"
            local_inputs = [
                bias,
                TensorMetadata(
                    shape=_tp_scaled_shape(activations.shape, tp_degree, "row"),
                    dtype=activations.dtype,
                    numel=activations.numel // tp_degree,
                    device=activations.device,
                ),
                TensorMetadata(
                    shape=[weights.shape[0] // tp_degree, weights.shape[1]],
                    dtype=weights.dtype,
                    numel=weights.numel // tp_degree,
                    device=weights.device,
                ),
            ]
            local_outputs = event.output_tensors
        parallelism = {**_parallelism_payload(strategy_config), "tp_mode": tp_mode}
        return local_inputs, local_outputs, parallelism

    parallelism = {**_parallelism_payload(strategy_config), "tp_mode": tp_mode}
    return event.input_tensors, event.output_tensors, parallelism


def capture_model_ops(
    model: torch.nn.Module,
    input_shape: List[int],
    precision: str,
    strategy_config: Dict[str, Any],
) -> List[OpRecord]:
    with torch.device("meta"):
        meta_model = model.to(device="meta")
        meta_input = torch.randn(*input_shape, device="meta")
    capture_mode = TorchOpCaptureMode()
    with capture_mode:
        meta_model(meta_input)

    precision_context = _precision_context(precision)
    op_records: List[OpRecord] = []
    for op_index, event in enumerate(capture_mode.events):
        local_input_tensors, local_output_tensors, parallelism = _localize_event(event, strategy_config)
        op_records.append(
            OpRecord(
                op_index=op_index,
                op_name=event.op_name,
                module_path=event.module_path,
                precision_context=precision_context,
                input_tensors=event.input_tensors,
                output_tensors=event.output_tensors,
                attrs=event.attrs,
                parallelism=parallelism,
                local_input_tensors=local_input_tensors,
                local_output_tensors=local_output_tensors,
            )
        )
    return op_records
