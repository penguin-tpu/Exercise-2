"""DMA engine model."""

from __future__ import annotations

from perf_modeling.config import DMAConfig
from perf_modeling.units.base import BaseUnit


class DMAUnit(BaseUnit):
    """DMA engine pool modeled as a single resource with queue capacity."""

    def __init__(self, name: str, config: DMAConfig) -> None:
        super().__init__(name)
        self.config = config

    def can_accept(self) -> bool:
        """Return whether the DMA subsystem can accept a new transfer."""
        return self.status.queued_ops < self.config.max_outstanding_transfers
