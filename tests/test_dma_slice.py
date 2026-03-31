"""Asynchronous DMA placeholder slice tests."""

from __future__ import annotations

import unittest

from perf_modeling.config import AcceleratorConfig, CoreConfig, DMAConfig, DRAMConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


class DMASliceTestCase(unittest.TestCase):
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

        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.exit_code, 9)
        self.assertEqual(engine.state.scratchpad.read(0, 16), b"abcdefghijklmnop")
        self.assertEqual(stats["dma.issued_ops"], 1)
        self.assertGreaterEqual(stats["stall_fence"], 1)


if __name__ == "__main__":
    unittest.main()
