"""End-to-end RV32I execution and timing tests."""

from __future__ import annotations

import struct

from perf_modeling.config import AcceleratorConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


def encode_i(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    """Encode one RV32I I-type instruction word."""
    return (((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode) & 0xFFFF_FFFF


def encode_r(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    """Encode one RV32I R-type instruction word."""
    return (
        (funct7 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (rd << 7)
        | opcode
    ) & 0xFFFF_FFFF


def encode_s(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    """Encode one RV32I S-type instruction word."""
    immediate = imm & 0xFFF
    return (
        ((immediate >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((immediate & 0x1F) << 7)
        | opcode
    ) & 0xFFFF_FFFF


def encode_b(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    """Encode one RV32I B-type instruction word."""
    immediate = imm & 0x1FFF
    return (
        (((immediate >> 12) & 0x1) << 31)
        | (((immediate >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((immediate >> 1) & 0xF) << 8)
        | (((immediate >> 11) & 0x1) << 7)
        | opcode
    ) & 0xFFFF_FFFF


def encode_u(opcode: int, rd: int, imm: int) -> int:
    """Encode one RV32I U-type instruction word."""
    return ((imm & 0xFFFFF000) | (rd << 7) | opcode) & 0xFFFF_FFFF


def build_elf32(code: bytes, base_address: int) -> bytes:
    """Construct a minimal ELF32 image with one executable load segment."""
    elf_header_size = 52
    program_header_size = 32
    text_offset = 0x100
    ident = bytearray(16)
    ident[0:4] = b"\x7fELF"
    ident[4] = 1
    ident[5] = 1
    ident[6] = 1
    header = struct.pack(
        "<16sHHIIIIIHHHHHH",
        bytes(ident),
        2,
        243,
        1,
        base_address,
        elf_header_size,
        0,
        0,
        elf_header_size,
        program_header_size,
        1,
        40,
        0,
        0,
    )
    program_header = struct.pack(
        "<IIIIIIII",
        1,
        text_offset,
        base_address,
        base_address,
        len(code),
        len(code),
        0x5,
        0x1000,
    )
    padding = b"\x00" * (text_offset - elf_header_size - program_header_size)
    return header + program_header + padding + code


def decode_program(words: list[int], base_address: int = 0x1000, as_elf: bool = False):
    """Decode a word stream into a simulator program."""
    blob = b"".join(word.to_bytes(4, byteorder="little", signed=False) for word in words)
    if as_elf:
        blob = build_elf32(blob, base_address)
        return Decoder().decode_bytes(blob, name="elf")
    return Decoder().decode_bytes(blob, base_address=base_address, name="raw")


def run_program(words: list[int], max_cycles: int = 1000, as_elf: bool = False) -> SimulatorEngine:
    """Decode and execute one RV32I machine-code program."""
    program = decode_program(words, as_elf=as_elf)
    engine = SimulatorEngine(AcceleratorConfig(), program)
    engine.run(max_cycles=max_cycles)
    return engine


class TestRv32iEngine:
    """End-to-end verification coverage for the RV32I engine slice."""

    def test_rv32i_scalar_alu_program_runs_to_ebreak(self) -> None:
        """Execute a short scalar ALU program and verify architected state."""
        words = [
            encode_i(0x13, 1, 0b000, 0, 5),
            encode_i(0x13, 2, 0b000, 0, 7),
            encode_r(0x33, 3, 0b000, 1, 2, 0x00),
            encode_i(0x13, 4, 0b001, 3, 1),
            encode_i(0x13, 5, 0b010, 4, 30),
            encode_r(0x33, 10, 0b000, 4, 0, 0x00),
            0x0010_0073,
        ]
        engine = run_program(words, max_cycles=32)
        assert engine.state.halted
        assert engine.state.exit_code == 24
        assert engine.state.scalar_regs.read(3) == 12
        assert engine.state.scalar_regs.read(4) == 24
        assert engine.state.scalar_regs.read(5) == 1
        assert engine.stats.counters["instructions_retired"] == 7

    def test_rv32i_branch_loop_accumulates_sum(self) -> None:
        """Execute a decrementing loop and verify branch correctness."""
        words = [
            encode_i(0x13, 1, 0b000, 0, 5),
            encode_i(0x13, 2, 0b000, 0, 0),
            encode_r(0x33, 2, 0b000, 2, 1, 0x00),
            encode_i(0x13, 1, 0b000, 1, -1),
            encode_b(0x63, 0b001, 1, 0, -8),
            encode_r(0x33, 10, 0b000, 2, 0, 0x00),
            0x0010_0073,
        ]
        engine = run_program(words, max_cycles=64)
        assert engine.state.halted
        assert engine.state.exit_code == 15
        assert engine.state.scalar_regs.read(2) == 15

    def test_rv32i_dram_store_and_load_round_trip_with_latency(self) -> None:
        """Store and reload a DRAM word and confirm the modeled long latency."""
        words = [
            encode_i(0x13, 1, 0b000, 0, 128),
            encode_u(0x37, 2, 0x1234_5000),
            encode_i(0x13, 2, 0b000, 2, 0x678),
            encode_s(0x23, 0b010, 1, 2, 0),
            encode_i(0x03, 3, 0b010, 1, 0),
            encode_r(0x33, 10, 0b000, 3, 0, 0x00),
            0x0010_0073,
        ]
        engine = run_program(words, max_cycles=300)
        assert engine.state.halted
        assert engine.state.scalar_regs.read(3) == 0x1234_5678
        assert engine.state.exit_code == 0x1234_5678
        assert engine.stats.counters["bytes_written"] == 4
        assert engine.stats.counters["bytes_read"] == 4
        assert engine.stats.counters["cycles"] >= 205

    def test_rv32i_elf_loader_and_jalr_control_flow(self) -> None:
        """Decode a minimal ELF image and execute AUIPC plus JALR control flow."""
        words = [
            encode_u(0x17, 1, 0),
            encode_i(0x13, 1, 0b000, 1, 20),
            encode_i(0x67, 5, 0b000, 1, 0),
            encode_i(0x13, 10, 0b000, 0, 1),
            0x0010_0073,
            encode_r(0x33, 10, 0b000, 5, 0, 0x00),
            0x0010_0073,
        ]
        engine = run_program(words, max_cycles=64, as_elf=True)
        assert engine.state.halted
        assert engine.program.entry_point == 0x1000
        assert engine.state.exit_code == 0x100C
        assert engine.state.scalar_regs.read(5) == 0x100C
