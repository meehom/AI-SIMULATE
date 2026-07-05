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
    assert op_names == [
        "aten.embedding.default",
        "aten.native_layer_norm.default",
        "aten.addmm.default",
        "aten.addmm.default",
        "aten.addmm.default",
        "aten.bmm.default",
        "aten.div.Tensor",
        "aten._softmax.default",
        "aten.bmm.default",
        "aten.addmm.default",
        "aten.add.Tensor",
        "aten.native_layer_norm.default",
        "aten.addmm.default",
        "aten.gelu.default",
        "custom.fc2.default",
        "aten.add.Tensor",
        "aten.native_layer_norm.default",
    ]
    assert records[0].output_tensors[0].shape == [1, 16, 32]
    assert records[0].op_kind == "builtin"
    assert records[14].op_kind == "custom"
    assert records[15].op_name == "aten.add.Tensor"
