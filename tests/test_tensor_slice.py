"""End-to-end tensor and accelerator placeholder slice tests."""

from __future__ import annotations

import struct

from perf_modeling.config import AcceleratorConfig, CoreConfig, DRAMConfig, ScalarUnitConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


def pack_int32(values: list[int]) -> bytes:
    """Pack a list of int32 values into little-endian bytes."""
    return struct.pack("<" + "i" * len(values), *values)


def unpack_int32(blob: bytes) -> tuple[int, ...]:
    """Unpack little-endian int32 values from bytes."""
    return struct.unpack("<" + "i" * (len(blob) // 4), blob)


class TestTensorSlice:
    """Exercise the first accelerator-side tensor vertical slice."""

    def make_config(self) -> AcceleratorConfig:
        """Construct a compact configuration for tensor-slice tests."""
        return AcceleratorConfig(
            core=CoreConfig(
                scalar=ScalarUnitConfig(pipeline_depth=1, queue_depth=4),
            ),
            dram=DRAMConfig(
                capacity_bytes=1 << 20,
                read_latency_cycles=3,
                write_latency_cycles=3,
                bytes_per_cycle=16,
            ),
        )

    def test_tensor_vertical_slice_executes_load_compute_store(self) -> None:
        """Tensor loads, vector add, matmul, and stores should execute through the engine."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(2, 2), dtype="int32")
            .emit_tensor_load(dest_tensor=1, address=0x240, shape=(2, 2), dtype="int32")
            .emit_vector_add(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype="int32")
            .emit_matmul(dest_tensor=3, lhs_tensor=0, rhs_tensor=1, acc_dtype="int32", out_dtype="int32")
            .emit_tensor_store(source_tensor=2, address=0x280)
            .emit_tensor_store(source_tensor=3, address=scratchpad_base)
            .emit("ebreak")
            .build(name="tensor-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, 6, 7, 8]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (6, 8, 10, 12)
        assert tuple(engine.state.tensor_regs.read(3).payload.reshape(-1).tolist()) == (19, 22, 43, 50)
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (6, 8, 10, 12)
        assert unpack_int32(engine.state.scratchpad.read(0, 16)) == (19, 22, 43, 50)
        assert stats["vector.issued_ops"] == 1
        assert stats["mxu.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 4
        assert stats["dram.bytes_read"] == 32
        assert stats["dram.bytes_written"] == 16
        assert stats["scratchpad.bytes_written"] == 16
