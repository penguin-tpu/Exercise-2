"""Scalar execution unit model."""

from __future__ import annotations

from perf_modeling.config import ScalarUnitConfig
from perf_modeling.units.base import BaseUnit


class ScalarUnit(BaseUnit):
    """Scalar execution unit with fixed pipeline depth and queue capacity."""

    def __init__(self, name: str, config: ScalarUnitConfig) -> None:
        super().__init__(name)
        self.config = config

    def can_accept(self) -> bool:
        """Return whether a new scalar operation fits in the issue queue."""
        return self.status.queued_ops < self.config.queue_depth
