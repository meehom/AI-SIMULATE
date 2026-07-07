from ai_simulate.workload import build_deepseek_v3_proxy, capture_model_ops


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
        "proxy_decode_kv_cache_len": 16,
    }


def _build_strategy_config():
    return {
        "name": "proxy_strategy_test",
        "tp_degree": 8,
        "pp_degree": 1,
        "dp_degree": 1,
        "gpu_count_used": 8,
    }


def test_torch_capture_records_expected_prefill_ops() -> None:
    workload_config = _build_workload_config()
    strategy_config = _build_strategy_config()
    model, input_spec = build_deepseek_v3_proxy(workload_config, phase="prefill")
    records = capture_model_ops(model, input_spec.shape, workload_config["precision"], strategy_config)

    op_names = [record.op_name for record in records]
    assert op_names[0] == "aten.embedding.default"
    assert op_names.count("aten.addmm.default") == 13
    assert op_names.count("aten.cat.default") == 2
    assert op_names.count("custom.fc2.default") == 2
    assert op_names[-1] == "aten.native_layer_norm.default"


def test_torch_capture_records_expected_decode_ops() -> None:
    workload_config = _build_workload_config()
    strategy_config = _build_strategy_config()
    model, input_spec = build_deepseek_v3_proxy(workload_config, phase="decode")
    records = capture_model_ops(model, input_spec.shape, workload_config["precision"], strategy_config)

    op_names = [record.op_name for record in records]
    assert op_names[0] == "aten.embedding.default"
    # MLA decode absorbs v_proj/o_proj into matmuls, so fewer addmm than prefill.
    assert op_names.count("aten.addmm.default") == 10
    # KV cache holds the compressed latent + rope branch only (2 zeros allocations).
    assert op_names.count("aten.zeros.default") == 2
    assert op_names.count("aten.cat.default") == 4
    assert op_names.count("aten.arange.start") == 2
    assert op_names.count("custom.fc2.default") == 2
    assert op_names[-1] == "aten.native_layer_norm.default"
