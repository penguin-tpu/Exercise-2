"""Instruction semantics registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from perf_modeling.isa.instruction import ExecutionPlan, Instruction

if TYPE_CHECKING:
    from perf_modeling.backend.torch_backend import TensorBackend
    from perf_modeling.config import AcceleratorConfig
    from perf_modeling.state.arch_state import ArchState
    from perf_modeling.timing.scoreboard import Scoreboard

InstructionPlanner = Callable[
    [Instruction, int, "ArchState", "AcceleratorConfig", "Scoreboard", "TensorBackend | None"],
    ExecutionPlan,
]


class SemanticsRegistry:
    """Registry mapping opcodes to planner callbacks."""

    def __init__(self) -> None:
        self._planners: dict[str, InstructionPlanner] = {}

    def register(self, opcode: str, planner: InstructionPlanner) -> None:
        """Register a planner callback for an opcode."""
        self._planners[opcode] = planner

    def lookup(self, opcode: str) -> InstructionPlanner:
        """Return the planner callback for an opcode."""
        return self._planners[opcode]

    def plan(
        self,
        instruction: Instruction,
        cycle: int,
        state: "ArchState",
        config: "AcceleratorConfig",
        scoreboard: "Scoreboard",
        backend: "TensorBackend | None" = None,
    ) -> ExecutionPlan:
        """Plan execution for one instruction using the registered callback."""
        planner = self.lookup(instruction.opcode)
        return planner(instruction, cycle, state, config, scoreboard, backend)
