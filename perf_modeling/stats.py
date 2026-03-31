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
        max_key = f"{unit_name}.max_queue_occupancy"
        self.counters[max_key] = max(self.counters.get(max_key, 0), depth)

    def record_event_queue_occupancy(self, depth: int) -> None:
        """Record one sampled occupancy bucket for the completion-event queue."""
        self.counters[f"event_queue.pending.{depth}"] += 1
        max_key = "event_queue.max_pending"
        self.counters[max_key] = max(self.counters.get(max_key, 0), depth)

    def record_instruction_latency(self, opcode: str, latency_cycles: int) -> None:
        """Record one planned instruction-latency sample."""
        self.counters[f"latency.{opcode}.samples"] += 1
        self.counters[f"latency.{opcode}.total_cycles"] += latency_cycles
        max_key = f"latency.{opcode}.max_cycles"
        self.counters[max_key] = max(self.counters.get(max_key, 0), latency_cycles)

    def snapshot(self) -> dict[str, int]:
        """Return a flat dictionary representation of accumulated counters."""
        data = dict(self.counters)
        data.update({f"{name}.busy_cycles": value for name, value in self.per_unit_busy_cycles.items()})
        data.update({f"{name}.issued_ops": value for name, value in self.per_unit_issued_ops.items()})
        return data
