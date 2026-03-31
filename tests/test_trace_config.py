"""Trace policy configuration tests."""

from __future__ import annotations

from perf_modeling.config import (
    AcceleratorConfig,
    CoreConfig,
    DMAConfig,
    DRAMConfig,
    ScalarUnitConfig,
    TimingConfig,
    TraceConfig,
)
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


class TestTraceConfig:
    """Validate that trace retention respects the configured policy knobs."""

    def test_enable_tracing_false_disables_all_trace_records(self) -> None:
        """Turning tracing off globally should suppress all retained records."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("addi", metadata={"rd": 10, "rs1": 0, "imm": 7, "source_regs": (0,), "dest_regs": (10,)})
            .emit("ebreak")
            .build(name="trace-disabled")
        )
        engine = SimulatorEngine(
            config=AcceleratorConfig(
                timing=TimingConfig(enable_tracing=False),
                dram=DRAMConfig(
                    capacity_bytes=1 << 20,
                    read_latency_cycles=3,
                    write_latency_cycles=3,
                    bytes_per_cycle=16,
                ),
            ),
            program=program,
        )

        engine.run(max_cycles=32)

        assert engine.state.halted
        assert engine.trace.records == []

    def test_keep_event_trace_false_retains_stalls_only(self) -> None:
        """Disabling event retention should keep stall records while dropping issue and completion events."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("addi", metadata={"rd": 1, "rs1": 0, "imm": 1, "source_regs": (0,), "dest_regs": (1,)})
            .emit("addi", metadata={"rd": 10, "rs1": 1, "imm": 2, "source_regs": (1,), "dest_regs": (10,)})
            .emit("ebreak")
            .build(name="trace-cycle-only")
        )
        engine = SimulatorEngine(
            config=AcceleratorConfig(
                core=CoreConfig(
                    scalar=ScalarUnitConfig(
                        pipeline_depth=3,
                        queue_depth=4,
                    )
                ),
                trace=TraceConfig(keep_cycle_trace=True, keep_event_trace=False, max_records=1000),
                dram=DRAMConfig(
                    capacity_bytes=1 << 20,
                    read_latency_cycles=3,
                    write_latency_cycles=3,
                    bytes_per_cycle=16,
                ),
            ),
            program=program,
        )

        engine.run(max_cycles=32)

        assert engine.state.halted
        assert any(record.kind == "stall" for record in engine.trace.records)
        assert not any(record.kind == "issue" for record in engine.trace.records)
        assert not any(record.kind == "complete" for record in engine.trace.records)

    def test_keep_cycle_trace_false_retains_events_only(self) -> None:
        """Disabling cycle-trace retention should drop stall records while keeping issue and completion events."""
        scratchpad_base = AcceleratorConfig().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=16)
            .emit("fence")
            .emit("ebreak")
            .build(name="trace-events-only")
        )
        engine = SimulatorEngine(
            config=AcceleratorConfig(
                core=CoreConfig(
                    dma=DMAConfig(
                        num_engines=1,
                        bytes_per_cycle=8,
                        setup_cycles=2,
                        max_outstanding_transfers=4,
                        burst_bytes=16,
                    )
                ),
                trace=TraceConfig(keep_cycle_trace=False, keep_event_trace=True, max_records=1000),
                dram=DRAMConfig(
                    capacity_bytes=1 << 20,
                    read_latency_cycles=3,
                    write_latency_cycles=3,
                    bytes_per_cycle=16,
                ),
            ),
            program=program,
        )
        engine.state.dram.write(0x300, b"abcdefghijklmnop")

        engine.run(max_cycles=200)

        assert engine.state.halted
        assert not any(record.kind == "stall" for record in engine.trace.records)
        assert any(record.kind == "issue" for record in engine.trace.records)
        assert any(record.kind == "complete" for record in engine.trace.records)
