from pathlib import Path

from ai_simulate.core import load_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = (
    REPO_ROOT
    / "configs/experiments/gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512.json"
)


def test_load_experiment_resolves_config_chain() -> None:
    resolved = load_experiment(EXPERIMENT_PATH)

    assert resolved.experiment_config["name"] == "gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512"
    assert resolved.chip_config["name"] == "NVIDIA GB300"
    assert resolved.topo_config["gpu_count"] == 72
    assert resolved.workload_config["model_name"] == "DeepSeek V3"
    assert resolved.strategy_config["gpu_count_used"] == 72
    assert resolved.output_dir.name == "gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512"
