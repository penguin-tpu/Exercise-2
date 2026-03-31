"""Latency helpers shared by unit planners."""

from __future__ import annotations

import math

from perf_modeling.config import (
    DRAMConfig,
    DMAConfig,
    ScalarUnitConfig,
    ScratchpadConfig,
    VectorUnitConfig,
    MXUConfig,
)


def dma_latency(config: DMAConfig, num_bytes: int) -> int:
    """Estimate DMA latency for a payload size."""
    return config.setup_cycles + math.ceil(num_bytes / max(1, config.bytes_per_cycle))


def mxu_latency(config: MXUConfig, tiles: int) -> int:
    """Estimate MXU latency for a count of abstract tiles."""
    return config.pipeline_depth + max(0, tiles - 1)


def vector_latency(config: VectorUnitConfig, elements: int) -> int:
    """Estimate vector latency for an element count."""
    return config.pipeline_depth + math.ceil(elements / max(1, config.lanes))


def scalar_latency(config: ScalarUnitConfig) -> int:
    """Return the latency of one scalar pipeline operation."""
    return max(1, config.pipeline_depth)


def scratchpad_latency(config: ScratchpadConfig, num_bytes: int) -> int:
    """Estimate scratchpad access latency for a payload size."""
    bytes_per_cycle = max(1, config.num_banks * config.bank_width_bytes)
    return max(1, math.ceil(num_bytes / bytes_per_cycle))


def dram_read_latency(config: DRAMConfig, num_bytes: int) -> int:
    """Estimate DRAM read latency for a payload size."""
    return config.read_latency_cycles + math.ceil(num_bytes / max(1, config.bytes_per_cycle))


def dram_write_latency(config: DRAMConfig, num_bytes: int) -> int:
    """Estimate DRAM write latency for a payload size."""
    return config.write_latency_cycles + math.ceil(num_bytes / max(1, config.bytes_per_cycle))
