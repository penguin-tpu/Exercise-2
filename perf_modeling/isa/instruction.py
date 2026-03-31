"""Instruction and execution plan primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from perf_modeling.timing.resources import ResourceReservation

if TYPE_CHECKING:
    from perf_modeling.backend.torch_backend import TensorBackend
    from perf_modeling.config import AcceleratorConfig
    from perf_modeling.state.arch_state import ArchState
    from perf_modeling.timing.scoreboard import Scoreboard


@dataclass
class ExecutionPlan:
    """Planned execution of one decoded instruction."""

    completion_cycle: int
    resources: list[ResourceReservation] = field(default_factory=list)
    on_complete: Callable[[], None] = field(default_factory=lambda: (lambda: None))
    description: str = ""
    stats: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class Instruction:
    """Decoded instruction plus enough metadata to plan execution."""

    opcode: str
    operands: tuple[object, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def source_regs(self) -> tuple[int, ...]:
        """Return the scalar source register indexes used by this instruction."""
        return tuple(int(index) for index in self.metadata.get("source_regs", ()))

    def dest_regs(self) -> tuple[int, ...]:
        """Return the scalar destination register indexes written by this instruction."""
        return tuple(int(index) for index in self.metadata.get("dest_regs", ()))

    def source_csrs(self) -> tuple[int, ...]:
        """Return the CSR addresses read by this instruction."""
        return tuple(int(address) for address in self.metadata.get("source_csrs", ()))

    def dest_csrs(self) -> tuple[int, ...]:
        """Return the CSR addresses written by this instruction."""
        return tuple(int(address) for address in self.metadata.get("dest_csrs", ()))

    def source_tensors(self) -> tuple[int, ...]:
        """Return the tensor register indexes read by this instruction."""
        return tuple(int(index) for index in self.metadata.get("source_tensors", ()))

    def dest_tensors(self) -> tuple[int, ...]:
        """Return the tensor register indexes written by this instruction."""
        return tuple(int(index) for index in self.metadata.get("dest_tensors", ()))

    def unit_name(self) -> str:
        """Return the execution unit expected to service this instruction."""
        if self.opcode in {"lb", "lh", "lw", "lbu", "lhu", "sb", "sh", "sw"}:
            return "load_store"
        if self.opcode in {"tload", "tstore"}:
            return "load_store"
        if self.opcode in {"dma_copy"}:
            return "dma"
        if self.opcode in {"vadd"}:
            return "vector"
        if self.opcode in {"matmul"}:
            return "mxu"
        return "scalar"

    def validate(self, state: "ArchState", config: "AcceleratorConfig") -> None:
        """Validate architectural legality before issue."""
        _ = state
        _ = config

    def plan(
        self,
        cycle: int,
        state: "ArchState",
        config: "AcceleratorConfig",
        scoreboard: "Scoreboard",
        backend: "TensorBackend | None" = None,
    ) -> ExecutionPlan:
        """Create a timing and completion plan for this instruction."""
        from perf_modeling.isa.semantics import DEFAULT_SEMANTICS

        return DEFAULT_SEMANTICS.plan(self, cycle, state, config, scoreboard, backend)
