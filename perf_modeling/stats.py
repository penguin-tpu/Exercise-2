"""Statistics collection helpers for simulation runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class SimulationStats:
    """Mutable statistics container updated during simulation."""

    counters: Counter[str] = field(default_factory=Counter)
    per_unit_busy_cycles: Counter[str] = field(default_factory=Counter)
    per_unit_issued_ops: Counter[str] = field(default_factory=Counter)

    def increment(self, key: str, amount: int = 1) -> None:
        """Increase a named counter by the provided amount."""
        self.counters[key] += amount

    def record_issue(self, unit_name: str) -> None:
        """Record that a unit accepted a new operation."""
        self.per_unit_issued_ops[unit_name] += 1

    def record_busy_cycle(self, unit_name: str) -> None:
        """Record one busy cycle for a modeled unit."""
        self.per_unit_busy_cycles[unit_name] += 1

    def record_queue_occupancy(self, unit_name: str, depth: int) -> None:
        """Record one sampled queue-occupancy bucket for a unit."""
        self.counters[f"{unit_name}.queue_occupancy.{depth}"] += 1

    def snapshot(self) -> dict[str, int]:
        """Return a flat dictionary representation of accumulated counters."""
        data = dict(self.counters)
        data.update({f"{name}.busy_cycles": value for name, value in self.per_unit_busy_cycles.items()})
        data.update({f"{name}.issued_ops": value for name, value in self.per_unit_issued_ops.items()})
        return data
