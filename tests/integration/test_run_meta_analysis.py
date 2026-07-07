import csv
import json
from pathlib import Path

from ai_simulate.simulator import run_meta_analysis_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = (
    REPO_ROOT
    / "configs/experiments/gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512.json"
)


def _assert_common_result(result, phase, expected_op_count):
    output_path = Path(result["output_path"])
    csv_path = Path(result["op_csv_path"])
    trace_path = Path(result["trace_output_path"])
    assert output_path.exists()
    assert csv_path.exists()
    assert trace_path.exists()
    assert output_path.name == f"meta_hook_analysis_{phase}.json"
    assert csv_path.name == f"op_trace_{phase}.csv"
    assert trace_path.name == f"chrome_trace_{phase}.json"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["analysis"]["model_proxy_type"] == "deepseek_v3_mla_moe_proxy"
    assert payload["analysis"]["proxy_input_spec"]["analysis_phase"] == phase
    assert payload["analysis"]["summary"]["captured_op_count"] == expected_op_count
    assert payload["analysis"]["ops"][0]["op_name"] == "aten.embedding.default"
    assert payload["analysis"]["ops"][-1]["op_name"] == "aten.native_layer_norm.default"
    assert any(op["op_name"] == "custom.fc2.default" and op["op_kind"] == "custom" for op in payload["analysis"]["ops"])

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_op_count

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    op_events = [event for event in trace_payload["traceEvents"] if event.get("ph") == "X"]
    assert len(op_events) == expected_op_count
    assert op_events[0]["ts"] == 0
    assert all(event["dur"] > 0 for event in op_events)
    return payload


def test_run_prefill_experiment_writes_result_file() -> None:
    result = run_meta_analysis_experiment(EXPERIMENT_PATH, phase="prefill")
    payload = _assert_common_result(result, "prefill", 83)
    op_names = [op["op_name"] for op in payload["analysis"]["ops"]]
    assert op_names.count("aten.zeros.default") == 0
    assert payload["analysis"]["proxy_input_spec"]["kv_cache_seq_len"] is None


def test_run_decode_experiment_writes_result_file() -> None:
    result = run_meta_analysis_experiment(EXPERIMENT_PATH, phase="decode")
    payload = _assert_common_result(result, "decode", 89)
    op_names = [op["op_name"] for op in payload["analysis"]["ops"]]
    assert op_names.count("aten.zeros.default") == 2
    assert op_names.count("aten.cat.default") == 4
    assert op_names.count("aten.arange.start") == 2
    assert payload["analysis"]["proxy_input_spec"]["shape"] == [1, 1]
    assert payload["analysis"]["proxy_input_spec"]["kv_cache_seq_len"] == 4352
    assert payload["analysis"]["decode_estimate"]["estimated_output_steps"] == 512
    assert payload["analysis"]["decode_estimate"]["estimated_total_decode_time_s"] > payload["analysis"]["decode_estimate"]["per_step_predicted_time_s"]
