"""Asynchronous DMA placeholder slice tests."""

from __future__ import annotations

from perf_modeling.config import AcceleratorConfig, CoreConfig, DMAConfig, DRAMConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


class TestDMASlice:
    """Exercise the DMA placeholder path through the event queue."""

    def make_config(self) -> AcceleratorConfig:
        """Construct a compact configuration for DMA tests."""
        return AcceleratorConfig(
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

    def test_dma_copy_overlaps_scalar_issue_and_fence_waits_for_completion(self) -> None:
        """DMA copies should run asynchronously and `fence` should wait for their completion."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=16)
            .emit("addi", metadata={"rd": 10, "rs1": 0, "imm": 9, "source_regs": (0,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="dma-overlap")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x300, b"abcdefghijklmnop")

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert engine.state.exit_code == 9
        assert engine.state.scratchpad.read(0, 16) == b"abcdefghijklmnop"
        assert stats["dma.issued_ops"] == 1
        assert stats["dram.bytes_read"] == 16
        assert stats["scratchpad.bytes_written"] == 16
        assert stats["stall_fence"] >= 1

    def test_dma_queue_occupancy_histogram_tracks_multiple_inflight_transfers(self) -> None:
        """Back-to-back DMA copies should populate unit and event-queue occupancy histograms."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=16)
            .emit_dma_copy(source_address=0x340, dest_address=scratchpad_base + 16, num_bytes=16)
            .emit("fence")
            .emit("ebreak")
            .build(name="dma-queue-occupancy")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x300, b"abcdefghijklmnop")
        engine.state.dram.write(0x340, b"qrstuvwxyzABCDEF")

        stats = engine.run(max_cycles=200).snapshot()

        assert engine.state.halted
        assert engine.state.scratchpad.read(0, 16) == b"abcdefghijklmnop"
        assert engine.state.scratchpad.read(16, 16) == b"qrstuvwxyzABCDEF"
        assert stats["dma.issued_ops"] == 2
        assert stats["dram.bytes_read"] == 32
        assert stats["scratchpad.bytes_written"] == 32
        assert stats["dma.max_queue_occupancy"] == 2
        assert stats["dma.queue_occupancy.1"] >= 1
        assert stats["dma.queue_occupancy.2"] >= 1
        assert stats["event_queue.max_pending"] == 2
        assert stats["event_queue.pending.1"] >= 1
        assert stats["event_queue.pending.2"] >= 1
