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


def pack_float32(values: list[float]) -> bytes:
    """Pack a list of float32 values into little-endian bytes."""
    return struct.pack("<" + "f" * len(values), *values)


def unpack_float32(blob: bytes) -> tuple[float, ...]:
    """Unpack little-endian float32 values from bytes."""
    return struct.unpack("<" + "f" * (len(blob) // 4), blob)


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

    def test_tensor_vertical_slice_executes_float32_load_compute_store(self) -> None:
        """Tensor loads, vector add, matmul, and stores should also support float32 payloads."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(2, 2), dtype="float32")
            .emit_tensor_load(dest_tensor=1, address=0x240, shape=(2, 2), dtype="float32")
            .emit_vector_add(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype="float32")
            .emit_matmul(dest_tensor=3, lhs_tensor=0, rhs_tensor=1, acc_dtype="float32", out_dtype="float32")
            .emit_tensor_store(source_tensor=2, address=0x280)
            .emit_tensor_store(source_tensor=3, address=scratchpad_base)
            .emit("ebreak")
            .build(name="tensor-float32-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 1.0, 1.5, 2.0]))
        engine.state.dram.write(0x240, pack_float32([2.0, 0.5, 1.0, 1.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (2.5, 1.5, 2.5, 3.5)
        assert tuple(engine.state.tensor_regs.read(3).payload.reshape(-1).tolist()) == (2.0, 1.75, 5.0, 3.75)
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (2.5, 1.5, 2.5, 3.5)
        assert unpack_float32(engine.state.scratchpad.read(0, 16)) == (2.0, 1.75, 5.0, 3.75)
        assert stats["vector.issued_ops"] == 1
        assert stats["mxu.issued_ops"] == 1

    def test_tensor_vertical_slice_executes_vector_mul_load_compute_store(self) -> None:
        """Tensor loads, vector multiply, and stores should execute through the engine."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="int32")
            .emit_tensor_load(dest_tensor=1, address=0x240, shape=(4,), dtype="int32")
            .emit_vector_mul(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype="int32")
            .emit_tensor_store(source_tensor=2, address=0x280)
            .emit("ebreak")
            .build(name="tensor-vector-mul-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, -1, 0, 2]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (5, -2, 0, 8)
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (5, -2, 0, 8)
        assert stats["vector.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 3

    def test_tensor_vertical_slice_executes_float32_vector_mul_load_compute_store(self) -> None:
        """Vector multiply should also support float32 payloads."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="float32")
            .emit_tensor_load(dest_tensor=1, address=0x240, shape=(4,), dtype="float32")
            .emit_vector_mul(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype="float32")
            .emit_tensor_store(source_tensor=2, address=0x280)
            .emit("ebreak")
            .build(name="tensor-float32-vector-mul-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 1.5, -2.0, 3.0]))
        engine.state.dram.write(0x240, pack_float32([2.0, -1.0, 0.25, 0.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (1.0, -1.5, -0.5, 1.5)
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (1.0, -1.5, -0.5, 1.5)
        assert stats["vector.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 3

    def test_tensor_vertical_slice_executes_vector_max_load_compute_store(self) -> None:
        """Tensor loads, vector max, and stores should execute through the engine."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="int32")
            .emit_tensor_load(dest_tensor=1, address=0x240, shape=(4,), dtype="int32")
            .emit_vector_max(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype="int32")
            .emit_tensor_store(source_tensor=2, address=0x280)
            .emit("ebreak")
            .build(name="tensor-vector-max-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([1, 7, -3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, 2, 0, 6]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (5, 7, 0, 6)
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (5, 7, 0, 6)
        assert stats["vector.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 3

    def test_tensor_vertical_slice_executes_float32_vector_max_load_compute_store(self) -> None:
        """Vector max should also support float32 payloads."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="float32")
            .emit_tensor_load(dest_tensor=1, address=0x240, shape=(4,), dtype="float32")
            .emit_vector_max(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype="float32")
            .emit_tensor_store(source_tensor=2, address=0x280)
            .emit("ebreak")
            .build(name="tensor-float32-vector-max-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 3.0, -2.0, 1.25]))
        engine.state.dram.write(0x240, pack_float32([1.0, 2.5, -1.0, 1.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (1.0, 3.0, -1.0, 1.5)
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (1.0, 3.0, -1.0, 1.5)
        assert stats["vector.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 3

    def test_tensor_vertical_slice_executes_vector_relu_and_reduce_sum(self) -> None:
        """Tensor loads, vector ReLU, vector reduce-sum, and stores should execute through the engine."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="int32")
            .emit_vector_relu(dest_tensor=1, source_tensor=0, out_dtype="int32")
            .emit_vector_reduce_sum(dest_tensor=2, source_tensor=1, out_dtype="int32")
            .emit_tensor_store(source_tensor=1, address=0x240)
            .emit_tensor_store(source_tensor=2, address=scratchpad_base)
            .emit("ebreak")
            .build(name="tensor-relu-reduce-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([-3, 0, 5, -1]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(1).payload.reshape(-1).tolist()) == (0, 0, 5, 0)
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (5,)
        assert unpack_int32(engine.state.dram.read(0x240, 16)) == (0, 0, 5, 0)
        assert unpack_int32(engine.state.scratchpad.read(0, 4)) == (5,)
        assert stats["vector.issued_ops"] == 2
        assert stats["load_store.issued_ops"] >= 3

    def test_tensor_vertical_slice_executes_float32_vector_relu_and_reduce_sum(self) -> None:
        """Tensor ReLU and reduction should also support float32 payloads."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="float32")
            .emit_vector_relu(dest_tensor=1, source_tensor=0, out_dtype="float32")
            .emit_vector_reduce_sum(dest_tensor=2, source_tensor=1, out_dtype="float32")
            .emit_tensor_store(source_tensor=1, address=0x240)
            .emit_tensor_store(source_tensor=2, address=scratchpad_base)
            .emit("ebreak")
            .build(name="tensor-float32-relu-reduce-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([-1.5, 0.0, 2.5, -0.25]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(1).payload.reshape(-1).tolist()) == (0.0, 0.0, 2.5, 0.0)
        assert tuple(engine.state.tensor_regs.read(2).payload.reshape(-1).tolist()) == (2.5,)
        assert unpack_float32(engine.state.dram.read(0x240, 16)) == (0.0, 0.0, 2.5, 0.0)
        assert unpack_float32(engine.state.scratchpad.read(0, 4)) == (2.5,)
        assert stats["vector.issued_ops"] == 2
        assert stats["load_store.issued_ops"] >= 3

    def test_tensor_vertical_slice_executes_vector_reduce_max(self) -> None:
        """Tensor reduce-max should execute through the engine for int32 payloads."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="int32")
            .emit_vector_reduce_max(dest_tensor=1, source_tensor=0, out_dtype="int32")
            .emit_tensor_store(source_tensor=1, address=scratchpad_base)
            .emit("ebreak")
            .build(name="tensor-reduce-max-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([-3, 7, 5, -1]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(1).payload.reshape(-1).tolist()) == (7,)
        assert unpack_int32(engine.state.scratchpad.read(0, 4)) == (7,)
        assert stats["vector.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 2

    def test_tensor_vertical_slice_executes_float32_vector_reduce_max(self) -> None:
        """Tensor reduce-max should also support float32 payloads."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_tensor_load(dest_tensor=0, address=0x200, shape=(4,), dtype="float32")
            .emit_vector_reduce_max(dest_tensor=1, source_tensor=0, out_dtype="float32")
            .emit_tensor_store(source_tensor=1, address=scratchpad_base)
            .emit("ebreak")
            .build(name="tensor-float32-reduce-max-vertical-slice")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([-1.5, 0.5, 2.5, -0.25]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert tuple(engine.state.tensor_regs.read(1).payload.reshape(-1).tolist()) == (2.5,)
        assert unpack_float32(engine.state.scratchpad.read(0, 4)) == (2.5,)
        assert stats["vector.issued_ops"] == 1
        assert stats["load_store.issued_ops"] >= 2
