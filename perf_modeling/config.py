"""Configuration dataclasses for the accelerator model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScalarUnitConfig:
    """Configuration for the scalar execution unit."""

    lanes: int = 1
    """Number of scalar operations the unit can complete per cycle."""
    pipeline_depth: int = 1
    """Number of cycles between scalar issue and scalar result writeback."""
    queue_depth: int = 4
    """Number of scalar operations that may wait in the unit issue queue."""


@dataclass(frozen=True)
class VectorUnitConfig:
    """Configuration for the vector execution unit."""

    lanes: int = 16
    """Number of vector elements processed per cycle."""
    pipeline_depth: int = 4
    """Number of cycles between vector issue and vector result writeback."""
    queue_depth: int = 8
    """Number of vector operations that may wait in the unit issue queue."""
    max_vector_length: int = 1024
    """Maximum vector length supported by the architectural interface."""


@dataclass(frozen=True)
class MXUConfig:
    """Configuration for the tensor or matrix execution unit."""

    rows: int = 16
    """Number of processing rows in the matrix compute array."""
    cols: int = 16
    """Number of processing columns in the matrix compute array."""
    macs_per_cycle: int = 256
    """Peak multiply-accumulate operations completed each cycle."""
    pipeline_depth: int = 8
    """Number of cycles between MXU issue and first result availability."""
    queue_depth: int = 4
    """Number of MXU operations that may wait in the unit issue queue."""
    accumulator_dtype: str = "int32"
    """Accumulator datatype used for partial sums inside the MXU."""


@dataclass(frozen=True)
class DMAConfig:
    """Configuration for DMA engines and transfer timing."""

    num_engines: int = 1
    """Number of independent DMA engines available for asynchronous copies."""
    bytes_per_cycle: int = 64
    """Peak sustained transfer bandwidth expressed in bytes per cycle."""
    setup_cycles: int = 8
    """Fixed latency paid when starting a DMA transfer."""
    max_outstanding_transfers: int = 16
    """Maximum number of in-flight DMA operations tracked by the model."""
    burst_bytes: int = 64
    """Preferred DMA burst size used by the simplified memory model."""


@dataclass(frozen=True)
class RegisterFileConfig:
    """Configuration for the scalar architectural register file."""

    num_scalar_registers: int = 32
    """Number of scalar registers visible to the instruction set."""
    scalar_register_width_bits: int = 32
    """Bit width of each scalar architectural register."""


@dataclass(frozen=True)
class TensorFileConfig:
    """Configuration for the tensor architectural register or descriptor file."""

    num_tensor_registers: int = 32
    """Number of tensor registers or tensor descriptors in the architecture."""
    max_tensor_rank: int = 4
    """Maximum tensor rank supported by the architectural register file."""
    max_tensor_elements: int = 65536
    """Maximum number of logical elements tracked by one tensor register."""


@dataclass(frozen=True)
class ScratchpadConfig:
    """Configuration for on-chip scratchpad SRAM storage."""

    capacity_bytes: int = 1 << 20
    """Total scratchpad capacity in bytes."""
    num_banks: int = 16
    """Number of independently addressable scratchpad banks."""
    bank_width_bytes: int = 32
    """Number of bytes transferred by one bank in one cycle."""
    read_ports: int = 2
    """Number of simultaneous scratchpad reads allowed per cycle."""
    write_ports: int = 2
    """Number of simultaneous scratchpad writes allowed per cycle."""
    bank_conflict_penalty_cycles: int = 1
    """Extra cycles charged when multiple accesses target the same bank."""


@dataclass(frozen=True)
class DRAMConfig:
    """Configuration for the off-chip memory abstraction."""

    capacity_bytes: int = 1 << 30
    """Total external memory capacity in bytes."""
    read_latency_cycles: int = 100
    """Base latency for a DRAM read before transfer begins."""
    write_latency_cycles: int = 100
    """Base latency for a DRAM write before transfer begins."""
    bytes_per_cycle: int = 32
    """Sustained DRAM bandwidth expressed in bytes per cycle."""


@dataclass(frozen=True)
class CoreConfig:
    """Top-level core configuration for execution resources."""

    issue_width: int = 1
    """Number of instructions or bundle slots that may issue each cycle."""
    max_inflight_ops: int = 64
    """Maximum number of in-flight operations tracked by the simulator."""
    scalar: ScalarUnitConfig = field(default_factory=ScalarUnitConfig)
    """Configuration for the scalar execution unit."""
    vector: VectorUnitConfig = field(default_factory=VectorUnitConfig)
    """Configuration for the vector execution unit."""
    mxu: MXUConfig = field(default_factory=MXUConfig)
    """Configuration for the tensor compute unit."""
    dma: DMAConfig = field(default_factory=DMAConfig)
    """Configuration for DMA engines and copy behavior."""


@dataclass(frozen=True)
class TimingConfig:
    """Global timing and simulator policy configuration."""

    frequency_hz: float = 1.0e9
    """Clock frequency used to convert cycles to wall-clock time."""
    retire_events_before_issue: bool = True
    """Whether completions become visible at the start of each cycle."""
    enable_scoreboard: bool = True
    """Whether register and resource hazards are enforced by a scoreboard."""
    enable_tracing: bool = True
    """Whether the simulator records a detailed execution trace."""


@dataclass(frozen=True)
class TraceConfig:
    """Configuration controlling trace output detail."""

    keep_cycle_trace: bool = True
    """Whether to retain a cycle-by-cycle trace in memory."""
    keep_event_trace: bool = True
    """Whether to retain operation issue and completion events."""
    max_records: int = 100000
    """Maximum number of trace records stored before dropping new records."""


@dataclass(frozen=True)
class AcceleratorConfig:
    """Complete accelerator configuration used to build a simulator instance."""

    core: CoreConfig = field(default_factory=CoreConfig)
    """Core execution configuration including all modeled functional units."""
    registers: RegisterFileConfig = field(default_factory=RegisterFileConfig)
    """Configuration for the scalar architectural register file."""
    tensors: TensorFileConfig = field(default_factory=TensorFileConfig)
    """Configuration for tensor architectural registers or descriptors."""
    scratchpad: ScratchpadConfig = field(default_factory=ScratchpadConfig)
    """Configuration for on-chip SRAM storage."""
    dram: DRAMConfig = field(default_factory=DRAMConfig)
    """Configuration for the off-chip memory abstraction."""
    timing: TimingConfig = field(default_factory=TimingConfig)
    """Configuration for simulator-wide timing and policy knobs."""
    trace: TraceConfig = field(default_factory=TraceConfig)
    """Configuration for trace capture and debugging output."""

