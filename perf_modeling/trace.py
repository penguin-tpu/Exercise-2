"""Execution trace recording helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.types import Cycle


@dataclass(slots=True)
class TraceRecord:
    """One trace record associated with a simulator cycle."""

    cycle: Cycle
    kind: str
    message: str


@dataclass
class TraceRecorder:
    """In-memory trace sink with a configurable record cap."""

    max_records: int
    records: list[TraceRecord] = field(default_factory=list)

    def append(self, cycle: Cycle, kind: str, message: str) -> None:
        """Append a trace record when space remains."""
        if len(self.records) >= self.max_records:
            return
        self.records.append(TraceRecord(cycle=cycle, kind=kind, message=message))

    def clear(self) -> None:
        """Drop all retained trace records."""
        self.records.clear()
