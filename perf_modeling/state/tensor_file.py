"""Tensor architectural register or descriptor file."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.state.arch_state import TensorValue


@dataclass
class TensorRegisterFile:
    """Indexed tensor register file."""

    num_registers: int
    values: list[TensorValue | None] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate empty tensor register slots."""
        self.values = [None for _ in range(self.num_registers)]

    def reset(self) -> None:
        """Clear all tensor registers."""
        for index in range(self.num_registers):
            self.values[index] = None

    def read(self, index: int) -> TensorValue | None:
        """Read a tensor register value."""
        return self.values[index]

    def write(self, index: int, value: TensorValue) -> None:
        """Write a tensor register value."""
        self.values[index] = value
