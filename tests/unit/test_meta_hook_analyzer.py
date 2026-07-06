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
        "proxy_vocab_size": 100,
        "proxy_num_layers": 1,
        "proxy_hidden_size": 32,
        "proxy_intermediate_size": 64,
        "proxy_num_attention_heads": 4,
        "proxy_use_moe": True,
        "proxy_num_experts": 4,
        "proxy_top_k": 2,
        "proxy_expert_intermediate_size": 64,
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
    assert op_names[0] == "aten.embedding.default"
    assert "aten.bmm.default" in op_names
    assert "aten._softmax.default" in op_names
    assert "aten.topk.default" in op_names
    assert "aten.scatter.src" in op_names
    assert "aten.unsqueeze.default" in op_names
    assert "aten.mul.Tensor" in op_names
    assert "aten.sum.dim_IntList" in op_names
    assert op_names.count("custom.fc2.default") == 4
    assert result["summary"]["captured_op_count"] == len(op_names)
    assert any(op["op_kind"] == "custom" for op in result["ops"])
