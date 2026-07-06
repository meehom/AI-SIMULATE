from ai_simulate.workload import build_deepseek_v3_proxy, capture_model_ops


def test_torch_capture_records_expected_supported_ops() -> None:
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
        "proxy_attention_impl": "mla",
        "proxy_mla_kv_lora_rank": 8,
        "proxy_mla_qk_nope_head_dim": 4,
        "proxy_mla_qk_rope_head_dim": 4,
        "proxy_mla_v_head_dim": 8,
    }
    strategy_config = {
        "name": "proxy_strategy_test",
        "tp_degree": 8,
        "pp_degree": 1,
        "dp_degree": 1,
        "gpu_count_used": 8,
    }

    model, input_spec = build_deepseek_v3_proxy(workload_config)
    records = capture_model_ops(model, input_spec.shape, workload_config["precision"], strategy_config)

    op_names = [record.op_name for record in records]
    assert op_names[0] == "aten.embedding.default"
    assert op_names.count("aten.addmm.default") == 16
    assert op_names.count("aten.bmm.default") == 3
    assert op_names.count("aten._softmax.default") == 2
    assert op_names.count("aten.add.Tensor") == 3
    assert op_names.count("aten.silu.default") == 4
    assert op_names.count("aten.mul.Tensor") == 5
    assert "aten.topk.default" in op_names
    assert "aten.zeros_like.default" in op_names
    assert "aten.scatter.src" in op_names
    assert "aten.stack.default" in op_names
    assert "aten.unsqueeze.default" in op_names
    assert "aten.sum.dim_IntList" in op_names
    assert op_names.count("custom.fc2.default") == 4
    assert op_names[-1] == "aten.native_layer_norm.default"
    assert records[0].output_tensors[0].shape == [1, 16, 32]
    assert any(record.op_kind == "custom" for record in records)
