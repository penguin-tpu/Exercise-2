"""Instruction and bundle format definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstructionOperand:
    """Operand reference carried by a decoded instruction."""

    kind: str
    value: object


@dataclass(frozen=True)
class Bundle:
    """Statically scheduled group of instructions issued together."""

    instructions: tuple[object, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)
