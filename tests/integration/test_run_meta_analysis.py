import json
from pathlib import Path

from ai_simulate.simulator import run_meta_analysis_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = (
    REPO_ROOT
    / "configs/experiments/gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512.json"
)


def test_run_meta_analysis_experiment_writes_result_file() -> None:
    result = run_meta_analysis_experiment(EXPERIMENT_PATH, phase="prefill")

    output_path = Path(result["output_path"])
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512"
    assert payload["analysis"]["model_proxy_type"] == "deepseek_v3_2layer_mlp"
    assert payload["analysis"]["summary"]["captured_op_count"] == 4
    assert payload["analysis"]["ops"][0]["op_name"] == "aten.native_layer_norm.default"
    assert payload["analysis"]["ops"][1]["op_name"] == "aten.addmm.default"
    assert payload["analysis"]["ops"][1]["metrics"]["predicted_time_s"] > 0
