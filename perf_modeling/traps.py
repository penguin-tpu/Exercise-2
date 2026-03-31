"""Machine trap records and architectural cause codes."""

from __future__ import annotations

from dataclasses import dataclass

CAUSE_INSTRUCTION_ADDRESS_MISALIGNED = 0
CAUSE_ILLEGAL_INSTRUCTION = 2
CAUSE_BREAKPOINT = 3
CAUSE_LOAD_ADDRESS_MISALIGNED = 4
CAUSE_STORE_ADDRESS_MISALIGNED = 6
CAUSE_ENV_CALL_FROM_M_MODE = 11


@dataclass(frozen=True)
class MachineTrap(Exception):
    """Structured machine trap raised during planning or completion."""

    cause: int
    pc: int
    tval: int
    reason: str

    def __str__(self) -> str:
        """Return a user-readable trap description."""
        return self.reason
