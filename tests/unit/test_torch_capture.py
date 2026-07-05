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
    records = capture_model_ops(model, input_spec.shape, workload_config["precision"], strategy_config)

    assert [record.op_name for record in records] == [
        "aten.native_layer_norm.default",
        "aten.addmm.default",
        "aten.gelu.default",
        "custom.fc2.default",
    ]
    assert records[0].output_tensors[0].shape == [1, 16, 32]
    assert records[1].local_output_tensors[0].shape == [16, 8]
    assert records[3].op_kind == "custom"
