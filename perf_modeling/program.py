"""Program containers used by the execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.isa.instruction import Instruction


@dataclass(frozen=True)
class ProgramSegment:
    """One loadable memory segment contained in a program image."""

    address: int
    data: bytes
    readable: bool = True
    writable: bool = False
    executable: bool = False


@dataclass
class Program:
    """Decoded program representation consumed by the simulator."""

    instructions: dict[int, Instruction] = field(default_factory=dict)
    segments: tuple[ProgramSegment, ...] = ()
    entry_point: int = 0
    initial_stack_pointer: int | None = None
    labels: dict[str, int] = field(default_factory=dict)
    name: str = "anonymous"

    def instruction_at(self, pc: int) -> Instruction:
        """Return the instruction located at the given program counter."""
        return self.instructions[pc]

    def is_done(self, pc: int) -> bool:
        """Return whether the program counter points past the program end."""
        return pc not in self.instructions

    def __len__(self) -> int:
        """Return the number of decoded instructions in the program."""
        return len(self.instructions)

    def contains_pc(self, pc: int) -> bool:
        """Return whether the fetch PC resolves to a decoded instruction."""
        return pc in self.instructions
