from ai_simulate.analysis import analyze_model_with_meta_hooks
from ai_simulate.workload import build_deepseek_v3_proxy


def _build_workload_config():
    return {
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
        "proxy_attention_impl": "mla",
        "proxy_mla_q_lora_rank": 8,
        "proxy_mla_kv_lora_rank": 8,
        "proxy_mla_qk_nope_head_dim": 4,
        "proxy_mla_qk_rope_head_dim": 4,
        "proxy_mla_v_head_dim": 8,
        "proxy_rope_theta": 10000.0,
    }


def _build_strategy_config():
    return {
        "name": "proxy_strategy_test",
        "tp_degree": 8,
        "pp_degree": 1,
        "dp_degree": 1,
        "gpu_count_used": 8,
    }


def _build_chip_config():
    return {
        "name": "Test GPU",
        "precision_performance": {"fp8_tflops": 1000},
        "memory": {"bandwidth_tb_per_s": 1.0},
    }


def test_meta_hook_analyzer_captures_expected_prefill_ops() -> None:
    workload_config = _build_workload_config()
    strategy_config = _build_strategy_config()
    model, input_spec = build_deepseek_v3_proxy(workload_config, phase="prefill")
    result = analyze_model_with_meta_hooks(
        model=model,
        input_shape=input_spec.shape,
        chip_config=_build_chip_config(),
        logical_precision=workload_config["precision"],
        analysis_phase="prefill",
        strategy_config=strategy_config,
    )

    op_names = [op["op_name"] for op in result["ops"]]
    assert op_names[0] == "aten.embedding.default"
    assert op_names.count("custom.fc2.default") == 2
    assert result["summary"]["captured_op_count"] == len(op_names) == 83


def test_meta_hook_analyzer_captures_expected_decode_ops() -> None:
    workload_config = _build_workload_config()
    strategy_config = _build_strategy_config()
    model, input_spec = build_deepseek_v3_proxy(workload_config, phase="decode")
    result = analyze_model_with_meta_hooks(
        model=model,
        input_shape=input_spec.shape,
        chip_config=_build_chip_config(),
        logical_precision=workload_config["precision"],
        analysis_phase="decode",
        strategy_config=strategy_config,
    )

    op_names = [op["op_name"] for op in result["ops"]]
    assert op_names[0] == "aten.embedding.default"
    assert op_names.count("aten.zeros.default") == 2
    assert op_names.count("aten.cat.default") == 4
    assert op_names.count("aten.arange.start") == 2
    assert result["summary"]["captured_op_count"] == len(op_names) == 89
