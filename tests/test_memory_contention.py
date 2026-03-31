"""Shared memory-layer contention tests."""

from __future__ import annotations

import struct

from perf_modeling.config import AcceleratorConfig, CoreConfig, DMAConfig, DRAMConfig, ScratchpadConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.builders import ProgramBuilder


class TestMemoryContention:
    """Verify that shared DRAM and scratchpad resources serialize across units."""

    def make_config(self) -> AcceleratorConfig:
        """Construct a compact configuration for contention tests."""
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
            scratchpad=ScratchpadConfig(
                capacity_bytes=1 << 20,
                num_banks=16,
                bank_width_bytes=32,
                read_ports=1,
                write_ports=1,
            ),
        )

    def test_dma_blocks_following_dram_load_until_memory_resource_is_free(self) -> None:
        """DMA traffic should reserve the DRAM layer and stall a following DRAM load."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("addi", metadata={"rd": 1, "rs1": 0, "imm": 0x400, "source_regs": (0,), "dest_regs": (1,)})
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=16)
            .emit("lw", metadata={"rd": 10, "rs1": 1, "imm": 0, "source_regs": (1,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="dram-contention")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x300, b"abcdefghijklmnop")
        engine.state.dram.write(0x400, struct.pack("<I", 0x11223344))

        engine.run(max_cycles=200)

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 0x11223344
        assert engine.stats.counters["stall_mem_dram_busy"] >= 1
        assert engine.stats.counters["dram.contention_stalls"] >= 1
        assert engine.stats.counters["memory.contention_stalls"] >= 1
        assert any("mem_dram busy" in message for message in stall_messages)

    def test_dma_blocks_following_scratchpad_load_until_memory_resource_is_free(self) -> None:
        """DMA traffic should reserve the touched scratchpad bank and stall a following same-bank load."""
        scratchpad_base = self.make_config().machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("lui", metadata={"rd": 1, "imm": scratchpad_base, "dest_regs": (1,)})
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=4)
            .emit("addi", metadata={"rd": 3, "rs1": 0, "imm": 0, "source_regs": (0,), "dest_regs": (3,)})
            .emit("lw", metadata={"rd": 10, "rs1": 1, "imm": 0, "source_regs": (1,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="scratchpad-contention")
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.state.dram.write(0x300, struct.pack("<I", 0x55667788))

        engine.run(max_cycles=200)

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 0x55667788
        assert engine.stats.counters["stall_sp_bank_0_busy"] >= 1
        assert engine.stats.counters["scratchpad.bank_conflict_stalls"] >= 1
        assert engine.stats.counters["scratchpad.contention_stalls"] >= 1
        assert any("sp_bank_0 busy" in message for message in stall_messages)

    def test_dma_allows_following_load_from_different_scratchpad_bank(self) -> None:
        """DMA traffic should not block a later active-phase load that targets a different scratchpad bank."""
        config = self.make_config()
        scratchpad_base = config.machine.scratchpad_base_address
        other_bank_address = scratchpad_base + config.scratchpad.bank_width_bytes
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("lui", metadata={"rd": 1, "imm": other_bank_address, "dest_regs": (1,)})
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=4)
            .emit("addi", metadata={"rd": 3, "rs1": 0, "imm": 0, "source_regs": (0,), "dest_regs": (3,)})
            .emit("lw", metadata={"rd": 10, "rs1": 1, "imm": 0, "source_regs": (1,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="scratchpad-independent-banks")
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x300, struct.pack("<I", 0x01020304))
        engine.state.write_memory(other_bank_address, struct.pack("<I", 0xAABBCCDD), config)

        engine.run(max_cycles=200)

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 0xAABBCCDD
        assert engine.stats.counters.get("stall_sp_bank_1_busy", 0) == 0
        assert engine.stats.counters.get("stall_sp_read_port_0_busy", 0) == 0
        assert engine.stats.counters.get("scratchpad.contention_stalls", 0) == 0
        assert not any("sp_bank_1 busy" in message for message in stall_messages)
        assert not any("sp_read_port_0 busy" in message for message in stall_messages)

    def test_dma_setup_allows_short_same_bank_load_before_transfer_phase(self) -> None:
        """A short scratchpad load may complete during DMA setup before same-bank transfer pressure begins."""
        config = self.make_config()
        scratchpad_base = config.machine.scratchpad_base_address
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("lui", metadata={"rd": 1, "imm": scratchpad_base, "dest_regs": (1,)})
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=4)
            .emit("lw", metadata={"rd": 10, "rs1": 1, "imm": 0, "source_regs": (1,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="scratchpad-setup-window")
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x300, struct.pack("<I", 0x01020304))
        engine.state.write_memory(scratchpad_base, struct.pack("<I", 0x55667788), config)

        engine.run(max_cycles=200)

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 0x55667788
        assert engine.state.read_memory(scratchpad_base, 4, config) == struct.pack("<I", 0x01020304)
        assert engine.stats.counters.get("stall_sp_bank_0_busy", 0) == 0
        assert engine.stats.counters.get("scratchpad.contention_stalls", 0) == 0
        assert not any("sp_bank_0 busy" in message for message in stall_messages)

    def test_dma_blocks_following_scratchpad_store_on_write_port(self) -> None:
        """DMA writes should serialize with later scratchpad stores when only one write port exists."""
        config = self.make_config()
        scratchpad_base = config.machine.scratchpad_base_address
        other_bank_address = scratchpad_base + config.scratchpad.bank_width_bytes
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("lui", metadata={"rd": 1, "imm": other_bank_address, "dest_regs": (1,)})
            .emit("addi", metadata={"rd": 2, "rs1": 0, "imm": 123, "source_regs": (0,), "dest_regs": (2,)})
            .emit_dma_copy(source_address=0x300, dest_address=scratchpad_base, num_bytes=4)
            .emit("addi", metadata={"rd": 4, "rs1": 0, "imm": 0, "source_regs": (0,), "dest_regs": (4,)})
            .emit("sw", metadata={"rs1": 1, "rs2": 2, "imm": 0, "source_regs": (1, 2)})
            .emit("lw", metadata={"rd": 10, "rs1": 1, "imm": 0, "source_regs": (1,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="scratchpad-write-port-contention")
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.dram.write(0x300, struct.pack("<I", 0x01020304))

        engine.run(max_cycles=200)

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 123
        assert engine.stats.counters["stall_sp_write_port_0_busy"] >= 1
        assert engine.stats.counters.get("stall_sp_bank_1_busy", 0) == 0
        assert engine.stats.counters["scratchpad.port_conflict_stalls"] >= 1
        assert engine.stats.counters["scratchpad.contention_stalls"] >= 1
        assert any("sp_write_port_0 busy" in message for message in stall_messages)
        assert not any("sp_bank_1 busy" in message for message in stall_messages)

    def test_dma_blocks_following_scratchpad_load_on_read_port(self) -> None:
        """DMA reads should serialize with later scratchpad loads when only one read port exists."""
        config = self.make_config()
        scratchpad_base = config.machine.scratchpad_base_address
        other_bank_address = scratchpad_base + config.scratchpad.bank_width_bytes
        program = (
            ProgramBuilder(base_address=0x1000)
            .emit("lui", metadata={"rd": 1, "imm": other_bank_address, "dest_regs": (1,)})
            .emit_dma_copy(source_address=scratchpad_base, dest_address=0x400, num_bytes=4)
            .emit("addi", metadata={"rd": 4, "rs1": 0, "imm": 0, "source_regs": (0,), "dest_regs": (4,)})
            .emit("lw", metadata={"rd": 10, "rs1": 1, "imm": 0, "source_regs": (1,), "dest_regs": (10,)})
            .emit("fence")
            .emit("ebreak")
            .build(name="scratchpad-read-port-contention")
        )
        engine = SimulatorEngine(config=config, program=program)
        engine.state.write_memory(scratchpad_base, struct.pack("<I", 0x11223344), config)
        engine.state.write_memory(other_bank_address, struct.pack("<I", 0x55667788), config)

        engine.run(max_cycles=200)

        stall_messages = [record.message for record in engine.trace.records if record.kind == "stall"]
        assert engine.state.halted
        assert engine.state.exit_code == 0x55667788
        assert engine.stats.counters["stall_sp_read_port_0_busy"] >= 1
        assert engine.stats.counters.get("stall_sp_bank_1_busy", 0) == 0
        assert engine.stats.counters["scratchpad.port_conflict_stalls"] >= 1
        assert engine.stats.counters["scratchpad.contention_stalls"] >= 1
        assert any("sp_read_port_0 busy" in message for message in stall_messages)
        assert not any("sp_bank_1 busy" in message for message in stall_messages)
