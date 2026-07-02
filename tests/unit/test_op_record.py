from ai_simulate.core.op_record import OpRecord, TensorMetadata, bytes_per_precision, shape_numel


def test_shape_numel_and_bytes_per_precision() -> None:
    assert shape_numel([2, 3, 4]) == 24
    assert shape_numel([]) == 0
    assert bytes_per_precision("fp8") == 1
    assert bytes_per_precision("fp16") == 2
    assert bytes_per_precision("fp32") == 4


def test_op_record_to_dict() -> None:
    op = OpRecord(
        op_index=0,
        op_name="aten.addmm.default",
        module_path="fc1",
        precision_context={"storage_precision": "fp8", "compute_precision": "fp8", "accum_precision": "fp16"},
        input_tensors=[TensorMetadata(shape=[1, 2, 3], dtype="float32", numel=6, device="meta")],
        output_tensors=[TensorMetadata(shape=[1, 2, 4], dtype="float32", numel=8, device="meta")],
    )
    payload = op.to_dict()
    assert payload["op_name"] == "aten.addmm.default"
    assert payload["module_path"] == "fc1"
    assert payload["output_tensors"][0]["shape"] == [1, 2, 4]
