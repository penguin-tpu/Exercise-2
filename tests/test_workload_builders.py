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


def pack_float32(values: list[float]) -> bytes:
    """Pack a list of float32 values into little-endian bytes."""
    return struct.pack("<" + "f" * len(values), *values)


def unpack_float32(blob: bytes) -> tuple[float, ...]:
    """Unpack little-endian float32 values from bytes."""
    return struct.unpack("<" + "f" * (len(blob) // 4), blob)


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

    def test_vector_add_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The vector-add builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="vector-add-float32",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_add_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 1.0, 1.5, 2.0]))
        engine.state.dram.write(0x240, pack_float32([2.0, 0.5, 1.0, 1.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (2.5, 1.5, 2.5, 3.5)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_mul_smoke_builder_executes_end_to_end(self) -> None:
        """The vector-multiply builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="vector-mul",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_mul_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, -1, 0, 2]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (5, -2, 0, 8)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_mul_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The vector-multiply builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="vector-mul-float32",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_mul_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 1.5, -2.0, 3.0]))
        engine.state.dram.write(0x240, pack_float32([2.0, -1.0, 0.25, 0.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (1.0, -1.5, -0.5, 1.5)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_max_smoke_builder_executes_end_to_end(self) -> None:
        """The vector-max builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="vector-max",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_max_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([1, 7, -3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, 2, 0, 6]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (5, 7, 0, 6)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_max_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The vector-max builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="vector-max-float32",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_max_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 3.0, -2.0, 1.25]))
        engine.state.dram.write(0x240, pack_float32([1.0, 2.5, -1.0, 1.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (1.0, 3.0, -1.0, 1.5)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_relu_smoke_builder_executes_end_to_end(self) -> None:
        """The vector-ReLU builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="vector-relu",
            input_shapes=((4,),),
            output_shape=(4,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_relu_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([-3, 0, 5, -1]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x240, 16)) == (0, 0, 5, 0)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_relu_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The vector-ReLU builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="vector-relu-float32",
            input_shapes=((4,),),
            output_shape=(4,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_relu_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([-1.5, 0.0, 2.5, -0.25]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x240, 16)) == (0.0, 0.0, 2.5, 0.0)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_reduce_sum_smoke_builder_executes_end_to_end(self) -> None:
        """The vector reduce-sum builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="vector-reduce-sum",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_reduce_sum_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([2, 4, 6, 8]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x240, 4)) == (20,)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_reduce_sum_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The vector reduce-sum builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="vector-reduce-sum-float32",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_reduce_sum_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([0.25, 0.75, 1.25, 1.75]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x240, 4)) == (4.0,)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_reduce_max_smoke_builder_executes_end_to_end(self) -> None:
        """The vector reduce-max builder should emit a runnable tensor microbenchmark."""
        problem = KernelProblem(
            name="vector-reduce-max",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_reduce_max_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="int32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_int32([-3, 7, 5, -1]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x240, 4)) == (7,)
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_vector_reduce_max_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The vector reduce-max builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="vector-reduce-max-float32",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        program = ProgramBuilder(base_address=0x1000).build_vector_reduce_max_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x200, pack_float32([-1.5, 0.5, 2.5, -0.25]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x240, 4)) == (2.5,)
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

    def test_matmul_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The matmul builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="matmul-float32",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        program = ProgramBuilder(base_address=0x1000).build_matmul_smoke_test(
            problem=problem,
            lhs_address=0x300,
            rhs_address=0x340,
            output_address=0x380,
            acc_dtype="float32",
            out_dtype="float32",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x300, pack_float32([0.5, 1.0, 1.5, 2.0]))
        engine.state.dram.write(0x340, pack_float32([2.0, 0.5, 1.0, 1.5]))

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x380, 16)) == (2.0, 1.75, 5.0, 3.75)
        assert stats["mxu.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_add_smoke_builder_executes_end_to_end(self) -> None:
        """The staged vector-add builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-vector-add",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_add_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, 6, 7, 8]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (6, 8, 10, 12)
        assert stats["dma.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_add_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged vector-add builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-vector-add-float32",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_add_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 1.0, 1.5, 2.0]))
        engine.state.dram.write(0x240, pack_float32([2.0, 0.5, 1.0, 1.5]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (2.5, 1.5, 2.5, 3.5)
        assert stats["dma.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_mul_smoke_builder_executes_end_to_end(self) -> None:
        """The staged vector-multiply builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-vector-mul",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_mul_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, -1, 0, 2]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (5, -2, 0, 8)
        assert stats["dma.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_mul_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged vector-multiply builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-vector-mul-float32",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_mul_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 1.5, -2.0, 3.0]))
        engine.state.dram.write(0x240, pack_float32([2.0, -1.0, 0.25, 0.5]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (1.0, -1.5, -0.5, 1.5)
        assert stats["dma.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_max_smoke_builder_executes_end_to_end(self) -> None:
        """The staged vector-max builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-vector-max",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_max_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_int32([1, 7, -3, 4]))
        engine.state.dram.write(0x240, pack_int32([5, 2, 0, 6]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x280, 16)) == (5, 7, 0, 6)
        assert stats["dma.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_max_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged vector-max builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-vector-max-float32",
            input_shapes=((4,), (4,)),
            output_shape=(4,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_max_smoke_test(
            problem=problem,
            lhs_address=0x200,
            rhs_address=0x240,
            output_address=0x280,
            dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_float32([0.5, 3.0, -2.0, 1.25]))
        engine.state.dram.write(0x240, pack_float32([1.0, 2.5, -1.0, 1.5]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x280, 16)) == (1.0, 3.0, -1.0, 1.5)
        assert stats["dma.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_relu_smoke_builder_executes_end_to_end(self) -> None:
        """The staged vector-ReLU builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-vector-relu",
            input_shapes=((4,),),
            output_shape=(4,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_relu_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_int32([-3, 0, 5, -1]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x240, 16)) == (0, 0, 5, 0)
        assert stats["dma.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_relu_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged vector-ReLU builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-vector-relu-float32",
            input_shapes=((4,),),
            output_shape=(4,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_relu_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_float32([-1.5, 0.0, 2.5, -0.25]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x240, 16)) == (0.0, 0.0, 2.5, 0.0)
        assert stats["dma.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_reduce_sum_smoke_builder_executes_end_to_end(self) -> None:
        """The staged vector reduce-sum builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-vector-reduce-sum",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_reduce_sum_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_int32([2, 4, 6, 8]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x240, 4)) == (20,)
        assert stats["dma.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_reduce_sum_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged vector reduce-sum builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-vector-reduce-sum-float32",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_reduce_sum_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_float32([0.25, 0.75, 1.25, 1.75]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x240, 4)) == (4.0,)
        assert stats["dma.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_reduce_max_smoke_builder_executes_end_to_end(self) -> None:
        """The staged vector reduce-max builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-vector-reduce-max",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_reduce_max_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_int32([-3, 7, 5, -1]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x240, 4)) == (7,)
        assert stats["dma.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_vector_reduce_max_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged vector reduce-max builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-vector-reduce-max-float32",
            input_shapes=((4,),),
            output_shape=(1,),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_vector_reduce_max_smoke_test(
            problem=problem,
            input_address=0x200,
            output_address=0x240,
            dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x200, pack_float32([-1.5, 0.5, 2.5, -0.25]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x240, 4)) == (2.5,)
        assert stats["dma.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_matmul_smoke_builder_executes_end_to_end(self) -> None:
        """The staged matmul builder should DMA through scratchpad before compute."""
        problem = KernelProblem(
            name="staged-matmul",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_matmul_smoke_test(
            problem=problem,
            lhs_address=0x300,
            rhs_address=0x340,
            output_address=0x380,
            acc_dtype="int32",
            out_dtype="int32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x300, pack_int32([1, 2, 3, 4]))
        engine.state.dram.write(0x340, pack_int32([5, 6, 7, 8]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_int32(engine.state.dram.read(0x380, 16)) == (19, 22, 43, 50)
        assert stats["dma.issued_ops"] == 3
        assert stats["mxu.issued_ops"] == 1
        assert stats["stall_fence"] >= 1

    def test_staged_matmul_smoke_builder_executes_float32_end_to_end(self) -> None:
        """The staged matmul builder should support float32 tensor payloads."""
        problem = KernelProblem(
            name="staged-matmul-float32",
            input_shapes=((2, 2), (2, 2)),
            output_shape=(2, 2),
        )
        config = self.make_config()
        program = ProgramBuilder(base_address=0x1000).build_staged_matmul_smoke_test(
            problem=problem,
            lhs_address=0x300,
            rhs_address=0x340,
            output_address=0x380,
            acc_dtype="float32",
            out_dtype="float32",
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x300, pack_float32([0.5, 1.0, 1.5, 2.0]))
        engine.state.dram.write(0x340, pack_float32([2.0, 0.5, 1.0, 1.5]))

        stats = engine.run(max_cycles=400).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(0x380, 16)) == (2.0, 1.75, 5.0, 3.75)
        assert stats["dma.issued_ops"] == 3
        assert stats["mxu.issued_ops"] == 1
        assert stats["stall_fence"] >= 1
