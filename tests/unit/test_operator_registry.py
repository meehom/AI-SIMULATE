from ai_simulate.core.op_record import OpRecord, TensorMetadata
from ai_simulate.roofline.op_costs import get_flops, get_memory


def test_addmm_registry_metrics() -> None:
    op = OpRecord(
        op_index=0,
        op_name="aten.addmm.default",
        op_kind="builtin",
        module_path=None,
        precision_context={"storage_precision": "fp8", "compute_precision": "fp8", "accum_precision": "fp16"},
        input_tensors=[],
        output_tensors=[],
        local_input_tensors=[
            TensorMetadata(shape=[8], dtype="float32", numel=8, device="meta"),
            TensorMetadata(shape=[16, 32], dtype="float32", numel=16 * 32, device="meta"),
            TensorMetadata(shape=[32, 8], dtype="float32", numel=32 * 8, device="meta"),
        ],
        local_output_tensors=[
            TensorMetadata(shape=[16, 8], dtype="float32", numel=16 * 8, device="meta"),
        ],
    )

    flops = get_flops(op)
    memory = get_memory(op)

    assert flops == 2 * 16 * 32 * 32
    assert memory.read_bytes == 8 + (16 * 32) + (32 * 8)
    assert memory.write_bytes == 16 * 8
    assert memory.total_bytes == memory.read_bytes + memory.write_bytes


def test_gelu_registry_metrics() -> None:
    op = OpRecord(
        op_index=1,
        op_name="aten.gelu.default",
        op_kind="builtin",
        module_path=None,
        precision_context={"storage_precision": "fp8", "compute_precision": "fp8", "accum_precision": "fp16"},
        input_tensors=[],
        output_tensors=[],
        local_input_tensors=[
            TensorMetadata(shape=[1, 16, 8], dtype="float32", numel=1 * 16 * 8, device="meta"),
        ],
        local_output_tensors=[
            TensorMetadata(shape=[1, 16, 8], dtype="float32", numel=1 * 16 * 8, device="meta"),
        ],
    )
    flops = get_flops(op)
    memory = get_memory(op)
    assert flops == 8.0 * (1 * 16 * 8)
    assert memory.total_bytes == (1 * 16 * 8) * 2
