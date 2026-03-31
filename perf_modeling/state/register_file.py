"""Scalar architectural register file."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScalarRegisterFile:
    """Simple indexed scalar register file."""

    num_registers: int
    register_width_bits: int
    values: list[int] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate storage for all architectural scalar registers."""
        self.values = [0 for _ in range(self.num_registers)]

    def reset(self) -> None:
        """Reset all scalar registers to zero."""
        for index in range(self.num_registers):
            self.values[index] = 0

    def read(self, index: int) -> int:
        """Read a scalar register value."""
        return self.values[index]

    def write(self, index: int, value: int) -> None:
        """Write a scalar register value after width truncation if needed."""
        self.values[index] = value
