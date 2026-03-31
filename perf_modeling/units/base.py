"""Base types shared by modeled execution units."""

from __future__ import annotations

from dataclasses import dataclass

from perf_modeling.types import Cycle


@dataclass
class UnitStatus:
    """Snapshot of one execution unit's occupancy state."""

    busy_until: Cycle = 0
    queued_ops: int = 0


class BaseUnit:
    """Shared execution-unit scaffold."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = UnitStatus()

    def is_busy(self) -> bool:
        """Return whether the unit has in-flight or queued work."""
        return self.status.queued_ops > 0 or self.status.busy_until > 0

    def can_accept(self) -> bool:
        """Return whether the unit can accept a new operation this cycle."""
        raise NotImplementedError

    def issue(self, completion_cycle: Cycle) -> None:
        """Record that a new operation has been issued to the unit."""
        self.status.queued_ops += 1
        self.status.busy_until = max(self.status.busy_until, completion_cycle)

    def complete(self) -> None:
        """Record that one previously issued operation has completed."""
        if self.status.queued_ops > 0:
            self.status.queued_ops -= 1

    def tick(self, cycle: Cycle) -> None:
        """Advance any unit-local timekeeping to the specified cycle."""
        if cycle >= self.status.busy_until and self.status.queued_ops == 0:
            self.status.busy_until = 0
