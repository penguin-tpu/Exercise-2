"""Load/store path model."""

from __future__ import annotations

from perf_modeling.units.base import BaseUnit


class LoadStoreUnit(BaseUnit):
    """Shared load/store path for register and scratchpad access."""

    def can_accept(self) -> bool:
        """Return whether the path can accept a new memory access."""
        return self.status.queued_ops == 0
