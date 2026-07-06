from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ai_simulate.core.op_record import OpRecord, TensorMetadata, shape_numel
from ai_simulate.custom import has_custom_estimator


SUPPORTED_CAPTURE_OPS = {
    "aten.embedding.default",
    "aten.native_layer_norm.default",
    "aten.addmm.default",
    "aten.gelu.default",
    "aten.silu.default",
    "aten.bmm.default",
    "aten.div.Tensor",
    "aten._softmax.default",
    "aten.add.Tensor",
    "aten.topk.default",
    "aten.zeros_like.default",
    "aten.scatter.src",
    "aten.stack.default",
    "aten.unsqueeze.default",
    "aten.mul.Tensor",
    "aten.sum.dim_IntList",
    "custom.fc2.default",
}

IGNORED_CAPTURE_OPS = {
    "aten.view.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.expand.default",
    "aten._unsafe_view.default",
    "aten.detach.default",
    "aten.clone.default",
}


class UnsupportedCapturedOpError(ValueError):
    """Raised when dispatch capture sees an operator without estimator coverage."""


class TorchOpCaptureMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.events: List[Dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        op_name = str(func)

        if op_name in SUPPORTED_CAPTURE_OPS:
            self.events.append(
                {
                    "op_name": op_name,
                    "input_tensors": _extract_tensor_metadata(args),
                    "output_tensors": _extract_output_metadata(output),
                    "attrs": _extract_relevant_attrs(op_name, args, kwargs),
                }
            )
        elif op_name not in IGNORED_CAPTURE_OPS:
            raise UnsupportedCapturedOpError(f"Unsupported captured operator: {op_name}")

        return output


def _extract_tensor_metadata(values: List[Any]) -> List[TensorMetadata]:
    tensors: List[TensorMetadata] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(_tensor_metadata(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    tensors.append(_tensor_metadata(item))
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


def _extract_relevant_attrs(op_name: str, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if op_name == "aten.native_layer_norm.default":
        normalized_shape = list(args[1]) if len(args) > 1 else []
        eps = args[4] if len(args) > 4 else kwargs.get("eps")
        return {"normalized_shape": normalized_shape, "eps": eps}
    if op_name == "aten._softmax.default":
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        return {"dim": dim}
    if op_name == "aten.topk.default":
        k = args[1] if len(args) > 1 else kwargs.get("k")
        return {"k": k}
    if op_name == "aten.scatter.src":
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        return {"dim": dim}
    if op_name == "aten.stack.default":
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        return {"dim": dim}
    if op_name == "aten.sum.dim_IntList":
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        return {"dim": list(dims) if isinstance(dims, (list, tuple)) else dims}
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


def _scaled_last_dim(shape: List[int], tp_degree: int) -> List[int]:
    scaled = list(shape)
    if not scaled or tp_degree == 1:
        return scaled
    if scaled[-1] % tp_degree != 0:
        raise ValueError(f"Shape {shape} last dim must be divisible by tp_degree={tp_degree}")
    scaled[-1] //= tp_degree
    return scaled


def _localize_builtin_addmm(event: Dict[str, Any], strategy_config: Dict[str, Any]) -> tuple[List[TensorMetadata], List[TensorMetadata], Dict[str, Any]]:
    tp_degree = int(strategy_config["tp_degree"])
    if len(event["input_tensors"]) != 3:
        raise ValueError(f"Expected 3 tensor inputs for addmm-like op, got {len(event['input_tensors'])}")

    bias, activations, weights = event["input_tensors"]
    output = event["output_tensors"][0]

    if output.shape[-1] >= activations.shape[-1]:
        local_inputs = [
            TensorMetadata(
                shape=[bias.shape[0] // tp_degree],
                dtype=bias.dtype,
                numel=bias.numel // tp_degree,
                device=bias.device,
            ),
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
                shape=_scaled_last_dim(output.shape, tp_degree),
                dtype=output.dtype,
                numel=output.numel // tp_degree,
                device=output.device,
            )
        ]
        tp_mode = "column"
    else:
        local_inputs = [
            bias,
            TensorMetadata(
                shape=_scaled_last_dim(activations.shape, tp_degree),
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
        local_outputs = [output]
        tp_mode = "row"

    parallelism = {**_parallelism_payload(strategy_config), "tp_mode": tp_mode}
    return local_inputs, local_outputs, parallelism


def _localize_custom_fc2(event: Dict[str, Any], strategy_config: Dict[str, Any]) -> tuple[List[TensorMetadata], List[TensorMetadata], Dict[str, Any]]:
    tp_degree = int(strategy_config["tp_degree"])
    if len(event["input_tensors"]) != 3:
        raise ValueError(f"Expected 3 tensor inputs for custom.fc2-like op, got {len(event['input_tensors'])}")

    x, weight, bias = event["input_tensors"]
    output = event["output_tensors"][0]
    if len(weight.shape) != 2:
        raise ValueError(f"Expected 2D weight tensor for custom.fc2, got shape={weight.shape}")

    local_inputs = [
        TensorMetadata(
            shape=_scaled_last_dim(x.shape, tp_degree),
            dtype=x.dtype,
            numel=x.numel // tp_degree,
            device=x.device,
        ),
        TensorMetadata(
            shape=[weight.shape[0], weight.shape[1] // tp_degree],
            dtype=weight.dtype,
            numel=weight.numel // tp_degree,
            device=weight.device,
        ),
        bias,
    ]
    local_outputs = [output]
    parallelism = {**_parallelism_payload(strategy_config), "tp_mode": "row"}
    return local_inputs, local_outputs, parallelism


def _localize_event(event: Dict[str, Any], strategy_config: Dict[str, Any]) -> tuple[List[TensorMetadata], List[TensorMetadata], Dict[str, Any]]:
    if event["op_name"] == "aten.addmm.default":
        return _localize_builtin_addmm(event, strategy_config)
    if event["op_name"] == "custom.fc2.default":
        return _localize_custom_fc2(event, strategy_config)
    parallelism = {**_parallelism_payload(strategy_config), "tp_mode": "replicated"}
    return event["input_tensors"], event["output_tensors"], parallelism


def capture_model_ops(
    model: torch.nn.Module,
    input_shape: List[int],
    precision: str,
    strategy_config: Dict[str, Any],
) -> List[OpRecord]:
    with torch.device("meta"):
        meta_model = model.to(device="meta")
        meta_input = torch.zeros(*input_shape, dtype=torch.long, device="meta")
    capture_mode = TorchOpCaptureMode()
    with capture_mode:
        meta_model(meta_input)

    precision_context = _precision_context(precision)
    op_records: List[OpRecord] = []
    for op_index, event in enumerate(capture_mode.events):
        local_input_tensors, local_output_tensors, parallelism = _localize_event(event, strategy_config)
        op_kind = "custom" if event["op_name"].startswith("custom.") or has_custom_estimator(event["op_name"]) else "builtin"
        op_records.append(
            OpRecord(
                op_index=op_index,
                op_name=event["op_name"],
                op_kind=op_kind,
                module_path=None,
                precision_context=precision_context,
                input_tensors=event["input_tensors"],
                output_tensors=event["output_tensors"],
                attrs=event["attrs"],
                parallelism=parallelism,
                local_input_tensors=local_input_tensors,
                local_output_tensors=local_output_tensors,
            )
        )
    return op_records
