"""Latency helpers shared by unit planners."""

from __future__ import annotations

import math

from perf_modeling.config import DMAConfig, MXUConfig, VectorUnitConfig


def dma_latency(config: DMAConfig, num_bytes: int) -> int:
    """Estimate DMA latency for a payload size."""
    return config.setup_cycles + math.ceil(num_bytes / max(1, config.bytes_per_cycle))


def mxu_latency(config: MXUConfig, tiles: int) -> int:
    """Estimate MXU latency for a count of abstract tiles."""
    return config.pipeline_depth + max(0, tiles - 1)


def vector_latency(config: VectorUnitConfig, elements: int) -> int:
    """Estimate vector latency for an element count."""
    return config.pipeline_depth + math.ceil(elements / max(1, config.lanes))
