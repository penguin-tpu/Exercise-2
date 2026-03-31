"""Instruction-latency statistics tests."""

from __future__ import annotations

from perf_modeling.config import AcceleratorConfig, CoreConfig, DMAConfig, DRAMConfig, ScalarUnitConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


class TestLatencyStats:
    """Validate per-opcode latency accounting in the stats surface."""

    def test_scalar_latency_stats_reflect_pipeline_depth(self) -> None:
        """Scalar instructions should record planned latency based on the configured pipeline depth."""
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("addi", metadata={"rd": 10, "rs1": 0, "imm": 7, "source_regs": (0,), "dest_regs": (10,)})
            .emit("ebreak")
            .build(name="scalar-latency-stats")
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

        stats = engine.run(max_cycles=32).snapshot()

        assert engine.state.halted
        assert stats["latency.addi.samples"] == 1
        assert stats["latency.addi.total_cycles"] == 3
        assert stats["latency.addi.max_cycles"] == 3

    def test_dma_latency_stats_reflect_transfer_latency(self) -> None:
        """DMA instructions should record planned latency based on setup plus transfer time."""
        config = AcceleratorConfig(
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
        )
        scratchpad_base = config.machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=16)
            .emit("fence")
            .emit("ebreak")
            .build(name="dma-latency-stats")
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x300, b"abcdefghijklmnop")

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert stats["latency.dma_copy.samples"] == 1
        assert stats["latency.dma_copy.total_cycles"] == 4
        assert stats["latency.dma_copy.max_cycles"] == 4
