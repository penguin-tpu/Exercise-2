"""Program builders for common kernels and synthetic tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.isa.instruction import Instruction
from perf_modeling.program import Program
from perf_modeling.workloads.kernels import KernelProblem

DTYPE_BYTES = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "float16": 2,
    "float32": 4,
}


def _tensor_payload_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Return the byte size of one tensor payload."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported staged workload dtype {dtype!r}.")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * DTYPE_BYTES[dtype]


@dataclass
class ProgramBuilder:
    """Helper used to assemble synthetic programs for validation."""

    base_address: int = 0
    instructions: list[Instruction] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)

    def emit(
        self,
        opcode: str,
        operands: tuple[object, ...] = (),
        metadata: dict[str, object] | None = None,
    ) -> "ProgramBuilder":
        """Append one instruction to the in-progress program."""
        self.instructions.append(
            Instruction(
                opcode=opcode,
                operands=operands,
                metadata=dict(metadata) if metadata is not None else {},
            )
        )
        return self

    def emit_tensor_load(
        self,
        dest_tensor: int,
        address: int,
        shape: tuple[int, ...],
        dtype: str,
    ) -> "ProgramBuilder":
        """Append a tensor load from memory into one tensor register."""
        return self.emit(
            "tload",
            metadata={
                "address": address,
                "shape": shape,
                "dtype": dtype,
                "dest_tensors": (dest_tensor,),
            },
        )

    def emit_dma_copy(
        self,
        source_address: int,
        dest_address: int,
        num_bytes: int,
    ) -> "ProgramBuilder":
        """Append an asynchronous DMA copy between two memory addresses."""
        return self.emit(
            "dma_copy",
            metadata={
                "source_address": source_address,
                "dest_address": dest_address,
                "num_bytes": num_bytes,
            },
        )

    def emit_tensor_store(self, source_tensor: int, address: int) -> "ProgramBuilder":
        """Append a tensor store from one tensor register into memory."""
        return self.emit(
            "tstore",
            metadata={
                "address": address,
                "source_tensors": (source_tensor,),
            },
        )

    def emit_vector_add(
        self,
        dest_tensor: int,
        lhs_tensor: int,
        rhs_tensor: int,
        out_dtype: str,
    ) -> "ProgramBuilder":
        """Append a vector elementwise add across two tensor registers."""
        return self.emit(
            "vadd",
            metadata={
                "source_tensors": (lhs_tensor, rhs_tensor),
                "dest_tensors": (dest_tensor,),
                "out_dtype": out_dtype,
            },
        )

    def emit_vector_mul(
        self,
        dest_tensor: int,
        lhs_tensor: int,
        rhs_tensor: int,
        out_dtype: str,
    ) -> "ProgramBuilder":
        """Append a vector elementwise multiply across two tensor registers."""
        return self.emit(
            "vmul",
            metadata={
                "source_tensors": (lhs_tensor, rhs_tensor),
                "dest_tensors": (dest_tensor,),
                "out_dtype": out_dtype,
            },
        )

    def emit_vector_max(
        self,
        dest_tensor: int,
        lhs_tensor: int,
        rhs_tensor: int,
        out_dtype: str,
    ) -> "ProgramBuilder":
        """Append a vector elementwise maximum across two tensor registers."""
        return self.emit(
            "vmax",
            metadata={
                "source_tensors": (lhs_tensor, rhs_tensor),
                "dest_tensors": (dest_tensor,),
                "out_dtype": out_dtype,
            },
        )

    def emit_vector_relu(
        self,
        dest_tensor: int,
        source_tensor: int,
        out_dtype: str,
    ) -> "ProgramBuilder":
        """Append a vector ReLU over one tensor register."""
        return self.emit(
            "vrelu",
            metadata={
                "source_tensors": (source_tensor,),
                "dest_tensors": (dest_tensor,),
                "out_dtype": out_dtype,
            },
        )

    def emit_vector_reduce_sum(
        self,
        dest_tensor: int,
        source_tensor: int,
        out_dtype: str,
        output_shape: tuple[int, ...] = (1,),
    ) -> "ProgramBuilder":
        """Append a vector reduce-sum over one tensor register."""
        return self.emit(
            "vreduce_sum",
            metadata={
                "source_tensors": (source_tensor,),
                "dest_tensors": (dest_tensor,),
                "out_dtype": out_dtype,
                "output_shape": output_shape,
            },
        )

    def emit_vector_reduce_max(
        self,
        dest_tensor: int,
        source_tensor: int,
        out_dtype: str,
        output_shape: tuple[int, ...] = (1,),
    ) -> "ProgramBuilder":
        """Append a vector reduce-max over one tensor register."""
        return self.emit(
            "vreduce_max",
            metadata={
                "source_tensors": (source_tensor,),
                "dest_tensors": (dest_tensor,),
                "out_dtype": out_dtype,
                "output_shape": output_shape,
            },
        )

    def emit_matmul(
        self,
        dest_tensor: int,
        lhs_tensor: int,
        rhs_tensor: int,
        acc_dtype: str,
        out_dtype: str,
    ) -> "ProgramBuilder":
        """Append one tensor matmul on the MXU."""
        return self.emit(
            "matmul",
            metadata={
                "source_tensors": (lhs_tensor, rhs_tensor),
                "dest_tensors": (dest_tensor,),
                "acc_dtype": acc_dtype,
                "out_dtype": out_dtype,
            },
        )

    def label(self, name: str) -> "ProgramBuilder":
        """Bind the current program counter to a textual label."""
        self.labels[name] = self.base_address + len(self.instructions) * 4
        return self

    def build(self, name: str = "generated") -> Program:
        """Freeze the accumulated instructions into a program object."""
        instruction_map: dict[int, Instruction] = {}
        for index, instruction in enumerate(self.instructions):
            pc = self.base_address + index * 4
            metadata = dict(instruction.metadata)
            metadata.setdefault("pc", pc)
            instruction_map[pc] = Instruction(
                opcode=instruction.opcode,
                operands=instruction.operands,
                metadata=metadata,
            )
        return Program(
            instructions=instruction_map,
            entry_point=self.base_address,
            labels=dict(self.labels),
            name=name,
        )

    def build_dma_smoke_test(self, problem: KernelProblem) -> Program:
        """Construct a basic DMA copy plus fence microbenchmark program."""
        output_elements = 1
        for dimension in problem.output_shape:
            output_elements *= dimension
        payload_bytes = max(4, output_elements * 4)
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=0x300,
                dest_address=0x2000_0000,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-dma-smoke")
        )

    def build_vector_add_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        dtype: str,
    ) -> Program:
        """Construct a tensor-load, vector-add, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Vector-add smoke tests expect exactly two input tensors.")
        if problem.input_shapes[0] != problem.input_shapes[1]:
            raise ValueError("Vector-add smoke tests expect matching input shapes.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-add smoke-test output shape must match the input tensors.")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=lhs_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_address, shape=problem.input_shapes[1], dtype=dtype)
            .emit_vector_add(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype=dtype)
            .emit_tensor_store(source_tensor=2, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-add-smoke")
        )

    def build_vector_relu_smoke_test(
        self,
        problem: KernelProblem,
        input_address: int,
        output_address: int,
        dtype: str,
    ) -> Program:
        """Construct a tensor-load, vector-ReLU, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 1:
            raise ValueError("Vector-ReLU smoke tests expect exactly one input tensor.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-ReLU smoke-test output shape must match the input tensor.")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=input_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_vector_relu(dest_tensor=1, source_tensor=0, out_dtype=dtype)
            .emit_tensor_store(source_tensor=1, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-relu-smoke")
        )

    def build_vector_mul_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        dtype: str,
    ) -> Program:
        """Construct a tensor-load, vector-multiply, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Vector-multiply smoke tests expect exactly two input tensors.")
        if problem.input_shapes[0] != problem.input_shapes[1]:
            raise ValueError("Vector-multiply smoke tests expect matching input shapes.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-multiply smoke-test output shape must match the input tensors.")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=lhs_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_address, shape=problem.input_shapes[1], dtype=dtype)
            .emit_vector_mul(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype=dtype)
            .emit_tensor_store(source_tensor=2, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-mul-smoke")
        )

    def build_vector_max_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        dtype: str,
    ) -> Program:
        """Construct a tensor-load, vector-max, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Vector-max smoke tests expect exactly two input tensors.")
        if problem.input_shapes[0] != problem.input_shapes[1]:
            raise ValueError("Vector-max smoke tests expect matching input shapes.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-max smoke-test output shape must match the input tensors.")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=lhs_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_address, shape=problem.input_shapes[1], dtype=dtype)
            .emit_vector_max(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype=dtype)
            .emit_tensor_store(source_tensor=2, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-max-smoke")
        )

    def build_vector_reduce_sum_smoke_test(
        self,
        problem: KernelProblem,
        input_address: int,
        output_address: int,
        dtype: str,
    ) -> Program:
        """Construct a tensor-load, vector-reduce-sum, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 1:
            raise ValueError("Vector reduce-sum smoke tests expect exactly one input tensor.")
        if problem.output_shape != (1,):
            raise ValueError("Vector reduce-sum smoke-test output shape must be (1,).")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=input_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_vector_reduce_sum(dest_tensor=1, source_tensor=0, out_dtype=dtype, output_shape=problem.output_shape)
            .emit_tensor_store(source_tensor=1, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-reduce-sum-smoke")
        )

    def build_vector_reduce_max_smoke_test(
        self,
        problem: KernelProblem,
        input_address: int,
        output_address: int,
        dtype: str,
    ) -> Program:
        """Construct a tensor-load, vector-reduce-max, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 1:
            raise ValueError("Vector reduce-max smoke tests expect exactly one input tensor.")
        if problem.output_shape != (1,):
            raise ValueError("Vector reduce-max smoke-test output shape must be (1,).")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=input_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_vector_reduce_max(dest_tensor=1, source_tensor=0, out_dtype=dtype, output_shape=problem.output_shape)
            .emit_tensor_store(source_tensor=1, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-reduce-max-smoke")
        )

    def build_staged_vector_add_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad vector-add smoke test."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Vector-add smoke tests expect exactly two input tensors.")
        if problem.input_shapes[0] != problem.input_shapes[1]:
            raise ValueError("Vector-add smoke tests expect matching input shapes.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-add smoke-test output shape must match the input tensors.")
        payload_bytes = _tensor_payload_bytes(problem.output_shape, dtype)
        lhs_scratch_address = scratchpad_base_address
        rhs_scratch_address = lhs_scratch_address + payload_bytes
        output_scratch_address = rhs_scratch_address + payload_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=lhs_address,
                dest_address=lhs_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit_dma_copy(
                source_address=rhs_address,
                dest_address=rhs_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=lhs_scratch_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_scratch_address, shape=problem.input_shapes[1], dtype=dtype)
            .emit_vector_add(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype=dtype)
            .emit_tensor_store(source_tensor=2, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-add-staged-smoke")
        )

    def build_staged_vector_mul_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad vector-multiply smoke test."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Vector-multiply smoke tests expect exactly two input tensors.")
        if problem.input_shapes[0] != problem.input_shapes[1]:
            raise ValueError("Vector-multiply smoke tests expect matching input shapes.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-multiply smoke-test output shape must match the input tensors.")
        payload_bytes = _tensor_payload_bytes(problem.output_shape, dtype)
        lhs_scratch_address = scratchpad_base_address
        rhs_scratch_address = lhs_scratch_address + payload_bytes
        output_scratch_address = rhs_scratch_address + payload_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=lhs_address,
                dest_address=lhs_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit_dma_copy(
                source_address=rhs_address,
                dest_address=rhs_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=lhs_scratch_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_scratch_address, shape=problem.input_shapes[1], dtype=dtype)
            .emit_vector_mul(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype=dtype)
            .emit_tensor_store(source_tensor=2, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-mul-staged-smoke")
        )

    def build_staged_vector_max_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad vector-max smoke test."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Vector-max smoke tests expect exactly two input tensors.")
        if problem.input_shapes[0] != problem.input_shapes[1]:
            raise ValueError("Vector-max smoke tests expect matching input shapes.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-max smoke-test output shape must match the input tensors.")
        payload_bytes = _tensor_payload_bytes(problem.output_shape, dtype)
        lhs_scratch_address = scratchpad_base_address
        rhs_scratch_address = lhs_scratch_address + payload_bytes
        output_scratch_address = rhs_scratch_address + payload_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=lhs_address,
                dest_address=lhs_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit_dma_copy(
                source_address=rhs_address,
                dest_address=rhs_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=lhs_scratch_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_scratch_address, shape=problem.input_shapes[1], dtype=dtype)
            .emit_vector_max(dest_tensor=2, lhs_tensor=0, rhs_tensor=1, out_dtype=dtype)
            .emit_tensor_store(source_tensor=2, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-max-staged-smoke")
        )

    def build_staged_vector_relu_smoke_test(
        self,
        problem: KernelProblem,
        input_address: int,
        output_address: int,
        dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad vector-ReLU smoke test."""
        if len(problem.input_shapes) != 1:
            raise ValueError("Vector-ReLU smoke tests expect exactly one input tensor.")
        if problem.output_shape != problem.input_shapes[0]:
            raise ValueError("Vector-ReLU smoke-test output shape must match the input tensor.")
        payload_bytes = _tensor_payload_bytes(problem.output_shape, dtype)
        input_scratch_address = scratchpad_base_address
        output_scratch_address = input_scratch_address + payload_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=input_address,
                dest_address=input_scratch_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=input_scratch_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_vector_relu(dest_tensor=1, source_tensor=0, out_dtype=dtype)
            .emit_tensor_store(source_tensor=1, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=payload_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-relu-staged-smoke")
        )

    def build_staged_vector_reduce_sum_smoke_test(
        self,
        problem: KernelProblem,
        input_address: int,
        output_address: int,
        dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad vector reduce-sum smoke test."""
        if len(problem.input_shapes) != 1:
            raise ValueError("Vector reduce-sum smoke tests expect exactly one input tensor.")
        if problem.output_shape != (1,):
            raise ValueError("Vector reduce-sum smoke-test output shape must be (1,).")
        input_bytes = _tensor_payload_bytes(problem.input_shapes[0], dtype)
        output_bytes = _tensor_payload_bytes(problem.output_shape, dtype)
        input_scratch_address = scratchpad_base_address
        output_scratch_address = input_scratch_address + input_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=input_address,
                dest_address=input_scratch_address,
                num_bytes=input_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=input_scratch_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_vector_reduce_sum(dest_tensor=1, source_tensor=0, out_dtype=dtype, output_shape=problem.output_shape)
            .emit_tensor_store(source_tensor=1, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=output_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-reduce-sum-staged-smoke")
        )

    def build_staged_vector_reduce_max_smoke_test(
        self,
        problem: KernelProblem,
        input_address: int,
        output_address: int,
        dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad vector reduce-max smoke test."""
        if len(problem.input_shapes) != 1:
            raise ValueError("Vector reduce-max smoke tests expect exactly one input tensor.")
        if problem.output_shape != (1,):
            raise ValueError("Vector reduce-max smoke-test output shape must be (1,).")
        input_bytes = _tensor_payload_bytes(problem.input_shapes[0], dtype)
        output_bytes = _tensor_payload_bytes(problem.output_shape, dtype)
        input_scratch_address = scratchpad_base_address
        output_scratch_address = input_scratch_address + input_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=input_address,
                dest_address=input_scratch_address,
                num_bytes=input_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=input_scratch_address, shape=problem.input_shapes[0], dtype=dtype)
            .emit_vector_reduce_max(dest_tensor=1, source_tensor=0, out_dtype=dtype, output_shape=problem.output_shape)
            .emit_tensor_store(source_tensor=1, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=output_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-vector-reduce-max-staged-smoke")
        )

    def build_matmul_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        acc_dtype: str,
        out_dtype: str,
    ) -> Program:
        """Construct a tensor-load, matmul, tensor-store microbenchmark program."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Matmul smoke tests expect exactly two input tensors.")
        lhs_shape = problem.input_shapes[0]
        rhs_shape = problem.input_shapes[1]
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            raise ValueError("Matmul smoke tests expect rank-2 input tensors.")
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("Matmul smoke tests expect compatible matrix inner dimensions.")
        if problem.output_shape != (lhs_shape[0], rhs_shape[1]):
            raise ValueError("Matmul smoke-test output shape must match matrix multiplication semantics.")
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_tensor_load(dest_tensor=0, address=lhs_address, shape=lhs_shape, dtype=out_dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_address, shape=rhs_shape, dtype=out_dtype)
            .emit_matmul(
                dest_tensor=2,
                lhs_tensor=0,
                rhs_tensor=1,
                acc_dtype=acc_dtype,
                out_dtype=out_dtype,
            )
            .emit_tensor_store(source_tensor=2, address=output_address)
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-matmul-smoke")
        )

    def build_staged_matmul_smoke_test(
        self,
        problem: KernelProblem,
        lhs_address: int,
        rhs_address: int,
        output_address: int,
        acc_dtype: str,
        out_dtype: str,
        scratchpad_base_address: int,
    ) -> Program:
        """Construct a DMA-to-scratchpad matmul smoke test."""
        if len(problem.input_shapes) != 2:
            raise ValueError("Matmul smoke tests expect exactly two input tensors.")
        lhs_shape = problem.input_shapes[0]
        rhs_shape = problem.input_shapes[1]
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            raise ValueError("Matmul smoke tests expect rank-2 input tensors.")
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("Matmul smoke tests expect compatible matrix inner dimensions.")
        if problem.output_shape != (lhs_shape[0], rhs_shape[1]):
            raise ValueError("Matmul smoke-test output shape must match matrix multiplication semantics.")
        lhs_bytes = _tensor_payload_bytes(lhs_shape, out_dtype)
        rhs_bytes = _tensor_payload_bytes(rhs_shape, out_dtype)
        output_bytes = _tensor_payload_bytes(problem.output_shape, out_dtype)
        lhs_scratch_address = scratchpad_base_address
        rhs_scratch_address = lhs_scratch_address + lhs_bytes
        output_scratch_address = rhs_scratch_address + rhs_bytes
        return (
            ProgramBuilder(base_address=self.base_address)
            .emit_dma_copy(
                source_address=lhs_address,
                dest_address=lhs_scratch_address,
                num_bytes=lhs_bytes,
            )
            .emit_dma_copy(
                source_address=rhs_address,
                dest_address=rhs_scratch_address,
                num_bytes=rhs_bytes,
            )
            .emit("fence")
            .emit_tensor_load(dest_tensor=0, address=lhs_scratch_address, shape=lhs_shape, dtype=out_dtype)
            .emit_tensor_load(dest_tensor=1, address=rhs_scratch_address, shape=rhs_shape, dtype=out_dtype)
            .emit_matmul(
                dest_tensor=2,
                lhs_tensor=0,
                rhs_tensor=1,
                acc_dtype=acc_dtype,
                out_dtype=out_dtype,
            )
            .emit_tensor_store(source_tensor=2, address=output_scratch_address)
            .emit("fence")
            .emit_dma_copy(
                source_address=output_scratch_address,
                dest_address=output_address,
                num_bytes=output_bytes,
            )
            .emit("fence")
            .emit("ebreak")
            .build(name=f"{problem.name}-matmul-staged-smoke")
        )
