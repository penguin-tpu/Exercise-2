"""Vector execution unit model."""

from __future__ import annotations

from perf_modeling.config import VectorUnitConfig
from perf_modeling.units.base import BaseUnit


class VectorUnit(BaseUnit):
    """Vector execution unit with a configurable queue and pipeline."""

    def __init__(self, name: str, config: VectorUnitConfig) -> None:
        super().__init__(name)
        self.config = config

    def can_accept(self) -> bool:
        """Return whether a new vector operation fits in the issue queue."""
        return self.status.queued_ops < self.config.queue_depth
