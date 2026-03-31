"""Trace-focused timing visibility tests."""

from __future__ import annotations

from perf_modeling.config import AcceleratorConfig, CoreConfig, DMAConfig, DRAMConfig, ScalarUnitConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


class TestTraceVisibility:
    """Validate that the execution trace exposes timing stall reasons."""

    def test_trace_records_scalar_dependency_stall(self) -> None:
        """Dependent scalar instructions should emit stall trace records while waiting on a result."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("addi", metadata={"rd": 1, "rs1": 0, "imm": 1, "source_regs": (0,), "dest_regs": (1,)})
            .emit("addi", metadata={"rd": 10, "rs1": 1, "imm": 2, "source_regs": (1,), "dest_regs": (10,)})
            .emit("ebreak")
            .build(name="scalar-dependency-trace")
        )
        engine = SimulatorEngine(
            config=AcceleratorConfig(
                core=CoreConfig(
                    scalar=ScalarUnitConfig(
                        pipeline_depth=3,
                        queue_depth=4,
                    )
                ),
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

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 3
        assert engine.stats.counters["stall_scalar_dependency"] >= 1
        assert any("scalar dependency" in message for message in stall_messages)

    def test_trace_records_fence_wait_until_dma_completion(self) -> None:
        """Asynchronous DMA should produce fence-wait trace records before the fence issues."""
        scratchpad_base = AcceleratorConfig().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=16)
            .emit("fence")
            .emit("ebreak")
            .build(name="dma-fence-trace")
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

        trace_records = list(engine.trace.records)
        fence_stall_indexes = [
            index
            for index, record in enumerate(trace_records)
            if record.kind == "stall" and "fence wait" in record.message
        ]
        dma_complete_indexes = [
            index
            for index, record in enumerate(trace_records)
            if record.kind == "complete" and "dma_copy" in record.message
        ]
        fence_issue_indexes = [
            index
            for index, record in enumerate(trace_records)
            if record.kind == "issue" and "fence" in record.message
        ]
        assert engine.state.halted
        assert engine.stats.counters["stall_fence"] >= 1
        assert fence_stall_indexes
        assert dma_complete_indexes
        assert fence_issue_indexes
        assert dma_complete_indexes[0] < fence_issue_indexes[0]
