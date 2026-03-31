"""Workload-builder smoke tests for tensor programs."""

from __future__ import annotations

import struct

from perf_modeling.config import AcceleratorConfig, CoreConfig, DRAMConfig, ScalarUnitConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder
from perf_modeling.workloads.kernels import KernelProblem


def pack_int32(values: list[int]) -> bytes:
    """Pack a list of int32 values into little-endian bytes."""
    return struct.pack("<" + "i" * len(values), *values)


def unpack_int32(blob: bytes) -> tuple[int, ...]:
    """Unpack little-endian int32 values from bytes."""
    return struct.unpack("<" + "i" * (len(blob) // 4), blob)


class TestWorkloadBuilders:
    """Exercise builder-generated tensor smoke-test programs end to end."""

    def make_config(self) -> AcceleratorConfig:
        """Construct a compact configuration for builder smoke tests."""
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

    def test_vector_add_smoke_builder_executes_end_to_end(self) -> None:
        """The vector-add builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="vector-add",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_add_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, 6, 7, 8]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (6, 8, 10, 12)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_matmul_smoke_builder_executes_end_to_end(self) -> None:
        """The matmul builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="matmul",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        program = ProgramBuilder(base_address=0x1000).build_matmul_smoke_test(
            problem=problem,
            lhs_address=0x300,
            rhs_address=0x340,
            output_address=0x380,
            acc_dtype="int32",
            out_dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x300, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x340, pack_int32([5, 6, 7, 8]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x380, 16)) == (19, 22, 43, 50)
        assert stats["mxu.issued_ops"] == 1
        assert stats["stall_fence"] >= 1
