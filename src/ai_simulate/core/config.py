from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ResolvedExperiment:
    experiment_path: Path
    repo_root: Path
    configs_root: Path
    output_dir: Path
    experiment_config: Dict[str, Any]
    chip_config_path: Path
    topo_config_path: Path
    workload_config_path: Path
    strategy_config_path: Path
    chip_config: Dict[str, Any]
    topo_config: Dict[str, Any]
    workload_config: Dict[str, Any]
    strategy_config: Dict[str, Any]


class ConfigError(ValueError):
    """Raised when a config file is missing or malformed."""


REQUIRED_WORKLOAD_KEYS = {
    "name",
    "model_name",
    "mode",
    "precision",
    "global_batch_size",
    "input_seq_len",
    "output_seq_len",
}

REQUIRED_STRATEGY_KEYS = {
    "name",
    "tp_degree",
    "pp_degree",
    "dp_degree",
    "gpu_count_used",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def find_configs_root(experiment_path: Path) -> Path:
    for candidate in [experiment_path.parent, *experiment_path.parents]:
        if candidate.name == "configs":
            return candidate
    raise ConfigError(f"Could not find configs root from experiment path: {experiment_path}")


def resolve_config_path(configs_root: Path, relative_ref: str) -> Path:
    resolved_path = (configs_root / relative_ref).resolve()
    if not resolved_path.exists():
        raise ConfigError(f"Referenced config does not exist: {relative_ref}")
    return resolved_path


def _require_keys(config_name: str, payload: Dict[str, Any], required_keys: set[str]) -> None:
    missing = sorted(required_keys - payload.keys())
    if missing:
        raise ConfigError(f"{config_name} missing required keys: {', '.join(missing)}")


def _require_nested_value(payload: Dict[str, Any], key_path: str) -> Any:
    current: Any = payload
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ConfigError(f"Missing required config path: {key_path}")
        current = current[key]
    return current


def _validate_strategy(strategy_config: Dict[str, Any], topo_config: Dict[str, Any]) -> None:
    tp_degree = int(strategy_config["tp_degree"])
    pp_degree = int(strategy_config["pp_degree"])
    dp_degree = int(strategy_config["dp_degree"])
    gpu_count_used = int(strategy_config["gpu_count_used"])
    topo_gpu_count = int(topo_config["gpu_count"])

    for name, value in {
        "tp_degree": tp_degree,
        "pp_degree": pp_degree,
        "dp_degree": dp_degree,
        "gpu_count_used": gpu_count_used,
    }.items():
        if value <= 0:
            raise ConfigError(f"{name} must be positive, got {value}")

    if tp_degree * pp_degree * dp_degree != gpu_count_used:
        raise ConfigError(
            "Strategy degrees must multiply to gpu_count_used: "
            f"{tp_degree} * {pp_degree} * {dp_degree} != {gpu_count_used}"
        )

    if gpu_count_used > topo_gpu_count:
        raise ConfigError(
            f"Strategy uses {gpu_count_used} GPUs, but topology only provides {topo_gpu_count}"
        )


def load_experiment(experiment_path: str | Path) -> ResolvedExperiment:
    experiment_path = Path(experiment_path).resolve()
    if not experiment_path.exists():
        raise ConfigError(f"Experiment config not found: {experiment_path}")

    configs_root = find_configs_root(experiment_path)
    repo_root = configs_root.parent
    experiment_config = read_json(experiment_path)

    chip_config_ref = _require_nested_value(experiment_config, "system.chip_config")
    topo_config_ref = _require_nested_value(experiment_config, "system.topo_config")
    workload_config_ref = _require_nested_value(experiment_config, "workload_config")
    strategy_config_ref = _require_nested_value(experiment_config, "strategy_config")

    chip_config_path = resolve_config_path(configs_root, chip_config_ref)
    topo_config_path = resolve_config_path(configs_root, topo_config_ref)
    workload_config_path = resolve_config_path(configs_root, workload_config_ref)
    strategy_config_path = resolve_config_path(configs_root, strategy_config_ref)

    chip_config = read_json(chip_config_path)
    topo_config = read_json(topo_config_path)
    workload_config = read_json(workload_config_path)
    strategy_config = read_json(strategy_config_path)

    _require_keys("workload_config", workload_config, REQUIRED_WORKLOAD_KEYS)
    _require_keys("strategy_config", strategy_config, REQUIRED_STRATEGY_KEYS)
    _require_nested_value(chip_config, "precision_performance.fp8_tflops")
    _require_nested_value(chip_config, "memory.bandwidth_tb_per_s")
    _require_nested_value(topo_config, "gpu_count")
    _require_nested_value(topo_config, "interconnect_bandwidth_gb_per_s")
    _require_nested_value(topo_config, "latency_us")
    _validate_strategy(strategy_config, topo_config)

    output_dir_ref = experiment_config.get("output_dir", f"results/{experiment_config['name']}")
    output_dir = (repo_root / output_dir_ref).resolve()

    return ResolvedExperiment(
        experiment_path=experiment_path,
        repo_root=repo_root,
        configs_root=configs_root,
        output_dir=output_dir,
        experiment_config=experiment_config,
        chip_config_path=chip_config_path,
        topo_config_path=topo_config_path,
        workload_config_path=workload_config_path,
        strategy_config_path=strategy_config_path,
        chip_config=chip_config,
        topo_config=topo_config,
        workload_config=workload_config,
        strategy_config=strategy_config,
    )
