import csv
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
    csv_path = Path(result["op_csv_path"])
    trace_path = Path(result["trace_output_path"])
    assert output_path.exists()
    assert csv_path.exists()
    assert trace_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512"
    assert payload["analysis"]["model_proxy_type"] == "deepseek_v3_2layer_mlp"
    assert payload["analysis"]["summary"]["captured_op_count"] == 4
    assert payload["analysis"]["ops"][0]["op_name"] == "aten.native_layer_norm.default"
    assert payload["analysis"]["ops"][1]["op_name"] == "aten.addmm.default"
    assert payload["analysis"]["ops"][3]["op_kind"] == "custom"

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["op_name"] == "aten.native_layer_norm.default"
    assert rows[0]["analysis"] in {"memory_bound", "compute_bound"}
    assert rows[0]["flops"]
    assert rows[0]["memory_bytes_total"]
    assert rows[0]["arithmetic_intensity"]
    assert rows[0]["input_shape"]
    assert rows[0]["output_shape"]
    assert rows[0]["op_time"]

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["displayTimeUnit"] == "ms"
    assert "traceEvents" in trace_payload
    trace_events = trace_payload["traceEvents"]
    assert len(trace_events) == 6
    op_events = [event for event in trace_events if event.get("ph") == "X"]
    assert len(op_events) == 4
    assert [event["name"] for event in op_events] == [
        "aten.native_layer_norm.default",
        "aten.addmm.default",
        "aten.gelu.default",
        "custom.fc2.default",
    ]
    assert op_events[0]["ts"] == 0
    assert all(event["dur"] > 0 for event in op_events)
    assert all(op_events[index]["ts"] <= op_events[index + 1]["ts"] for index in range(len(op_events) - 1))
