"""Named example accelerator configurations for CLI selection and experiments."""

from __future__ import annotations

from dataclasses import replace

from perf_modeling.config import (
    AcceleratorConfig,
    DMAConfig,
    DRAMConfig,
    MXUConfig,
    ScratchpadConfig,
    ScalarUnitConfig,
    VectorUnitConfig,
)


def baseline_config() -> AcceleratorConfig:
    """Return the default baseline configuration used by most tests and examples."""
    return AcceleratorConfig()


def tiny_debug_config() -> AcceleratorConfig:
    """Return a compact, slower configuration suited for debugging and simple traces."""
    return AcceleratorConfig(
        core=replace(
            AcceleratorConfig().core,
            scalar=ScalarUnitConfig(lanes=1, pipeline_depth=2, queue_depth=2),
            vector=VectorUnitConfig(lanes=4, pipeline_depth=6, queue_depth=2, max_vector_length=128),
            mxu=MXUConfig(rows=4, cols=4, macs_per_cycle=16, pipeline_depth=12, queue_depth=2),
            dma=DMAConfig(num_engines=1, bytes_per_cycle=16, setup_cycles=12, max_outstanding_transfers=4),
        ),
        scratchpad=ScratchpadConfig(capacity_bytes=1 << 16, num_banks=4, bank_width_bytes=16),
        dram=DRAMConfig(capacity_bytes=1 << 24, read_latency_cycles=120, write_latency_cycles=120, bytes_per_cycle=16),
    )


def balanced_ml_config() -> AcceleratorConfig:
    """Return a balanced accelerator-oriented preset for typical mixed workloads."""
    baseline = AcceleratorConfig()
    return replace(
        baseline,
        core=replace(
            baseline.core,
            vector=VectorUnitConfig(lanes=32, pipeline_depth=4, queue_depth=8, max_vector_length=2048),
            mxu=MXUConfig(rows=16, cols=16, macs_per_cycle=256, pipeline_depth=8, queue_depth=4),
            dma=DMAConfig(num_engines=2, bytes_per_cycle=128, setup_cycles=6, max_outstanding_transfers=32),
        ),
        scratchpad=ScratchpadConfig(
            capacity_bytes=2 << 20,
            num_banks=32,
            bank_width_bytes=32,
            read_ports=4,
            write_ports=4,
            bank_conflict_penalty_cycles=1,
        ),
        dram=DRAMConfig(capacity_bytes=2 << 30, read_latency_cycles=90, write_latency_cycles=90, bytes_per_cycle=64),
    )


def throughput_ml_config() -> AcceleratorConfig:
    """Return a wider preset that favors throughput over small-debug simplicity."""
    baseline = AcceleratorConfig()
    return replace(
        baseline,
        core=replace(
            baseline.core,
            vector=VectorUnitConfig(lanes=64, pipeline_depth=5, queue_depth=16, max_vector_length=4096),
            mxu=MXUConfig(rows=32, cols=32, macs_per_cycle=1024, pipeline_depth=10, queue_depth=8),
            dma=DMAConfig(num_engines=4, bytes_per_cycle=256, setup_cycles=6, max_outstanding_transfers=64),
        ),
        scratchpad=ScratchpadConfig(
            capacity_bytes=4 << 20,
            num_banks=64,
            bank_width_bytes=64,
            read_ports=8,
            write_ports=8,
            bank_conflict_penalty_cycles=1,
        ),
        dram=DRAMConfig(capacity_bytes=4 << 30, read_latency_cycles=80, write_latency_cycles=80, bytes_per_cycle=128),
    )


CONFIG_PRESETS: dict[str, AcceleratorConfig] = {
    "baseline": baseline_config(),
    "tiny_debug": tiny_debug_config(),
    "balanced_ml": balanced_ml_config(),
    "throughput_ml": throughput_ml_config(),
}
"""Named example hardware configurations exposed by the CLI."""

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "baseline": "Default balanced simulator baseline used by most tests and examples.",
    "tiny_debug": "Small and slower preset for easy tracing and quick debug runs.",
    "balanced_ml": "Midrange accelerator preset for mixed DMA, vector, and MXU workloads.",
    "throughput_ml": "Wide throughput-oriented preset with larger vector, DMA, and MXU capacity.",
}
"""Short descriptions for the named hardware configuration presets."""


def available_config_names() -> tuple[str, ...]:
    """Return the stable list of selectable configuration preset names."""
    return tuple(CONFIG_PRESETS)


def describe_named_config(name: str) -> str:
    """Return the short human-readable description for one named config preset."""
    return CONFIG_DESCRIPTIONS[name]


def get_named_config(name: str) -> AcceleratorConfig:
    """Return one named configuration preset or raise `KeyError` if missing."""
    return CONFIG_PRESETS[name]
