"""Tensor compute unit model."""

from __future__ import annotations

from perf_modeling.config import MXUConfig
from perf_modeling.units.base import BaseUnit


class MXUUnit(BaseUnit):
    """Tensor or matrix execution unit with configurable array dimensions."""

    def __init__(self, name: str, config: MXUConfig) -> None:
        super().__init__(name)
        self.config = config

    def can_accept(self) -> bool:
        """Return whether a new MXU operation fits in the issue queue."""
        return self.status.queued_ops < self.config.queue_depth
