from ai_simulate.analysis import analyze_model_with_meta_hooks
from ai_simulate.workload import build_deepseek_v3_proxy


def test_meta_hook_analyzer_captures_expected_aten_ops() -> None:
    workload_config = {
        "name": "deepseek_v3_proxy_test",
        "model_name": "DeepSeek V3",
        "mode": "inference",
        "precision": "fp8",
        "global_batch_size": 1,
        "input_seq_len": 16,
        "output_seq_len": 4,
        "proxy_hidden_size": 32,
        "proxy_intermediate_size": 64,
    }
    strategy_config = {
        "name": "proxy_strategy_test",
        "tp_degree": 8,
        "pp_degree": 1,
        "dp_degree": 1,
        "gpu_count_used": 8,
    }

    model, input_spec = build_deepseek_v3_proxy(workload_config)
    chip_config = {
        "name": "Test GPU",
        "precision_performance": {"fp8_tflops": 1000},
        "memory": {"bandwidth_tb_per_s": 1.0},
    }
    result = analyze_model_with_meta_hooks(
        model=model,
        input_shape=input_spec.shape,
        chip_config=chip_config,
        logical_precision=workload_config["precision"],
        analysis_phase="prefill",
        strategy_config=strategy_config,
    )

    op_names = [op["op_name"] for op in result["ops"]]
    assert op_names == [
        "aten.native_layer_norm.default",
        "aten.addmm.default",
        "aten.gelu.default",
        "aten.addmm.default",
    ]
    assert result["summary"]["captured_op_count"] == 4
    assert result["ops"][1]["local_output_tensors"][0]["shape"] == [16, 8]
    assert result["ops"][3]["local_input_tensors"][1]["shape"] == [16, 8]
    assert result["ops"][1]["metrics"]["flops"] > 0
    assert result["ops"][1]["metrics"]["predicted_time_s"] > 0
