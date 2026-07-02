from __future__ import annotations

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
        "analysis": {
            **analysis_result,
            "model_proxy_type": "deepseek_v3_2layer_mlp",
            "proxy_input_spec": {
                "shape": proxy_input_spec.shape,
                "hidden_size": proxy_input_spec.hidden_size,
                "intermediate_size": proxy_input_spec.intermediate_size,
                "activation": proxy_input_spec.activation,
                "analysis_phase": proxy_input_spec.analysis_phase,
            },
            "recorded_output_seq_len": resolved_experiment.workload_config["output_seq_len"],
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "This prototype uses a 2-layer MLP as a stand-in for DeepSeek V3.",
            f"Analysis phase is restricted to {analysis_phase} semantics in v1.",
            "Current roofline timing is based on intercepted built-in PyTorch ops under dispatch capture.",
        ],
    }


def run_meta_analysis_experiment(experiment_path: str | Path, phase: str = "prefill") -> Dict[str, Any]:
    if phase != "prefill":
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
    output_path = output_dir / "meta_hook_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    result_payload["output_path"] = str(output_path)
    return result_payload
