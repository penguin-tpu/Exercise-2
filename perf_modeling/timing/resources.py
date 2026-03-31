"""Execution resource reservation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceReservation:
    """Reservation of a named resource over a cycle interval."""

    resource_name: str
    start_cycle: int
    end_cycle: int

    def overlaps(self, cycle: int) -> bool:
        """Return whether the reservation covers the specified cycle."""
        return self.start_cycle <= cycle < self.end_cycle

    def overlaps_interval(self, start_cycle: int, end_cycle: int) -> bool:
        """Return whether this reservation overlaps another half-open interval."""
        return self.start_cycle < end_cycle and start_cycle < self.end_cycle
