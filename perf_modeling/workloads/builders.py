"""Program builders for common kernels and synthetic tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.isa.instruction import Instruction
from perf_modeling.program import Program
from perf_modeling.workloads.kernels import KernelProblem


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
        """Construct a placeholder DMA-oriented microbenchmark program."""
        _ = problem
        raise NotImplementedError("DMA smoke-test generation has not been implemented yet.")
