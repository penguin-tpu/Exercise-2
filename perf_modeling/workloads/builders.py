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

    def label(self, name: str) -> "ProgramBuilder":
        """Bind the current program counter to a textual label."""
        self.labels[name] = self.base_address + len(self.instructions) * 4
        return self

    def build(self, name: str = "generated") -> Program:
        """Freeze the accumulated instructions into a program object."""
        instruction_map = {
            self.base_address + index * 4: instruction
            for index, instruction in enumerate(self.instructions)
        }
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
