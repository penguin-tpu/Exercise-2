"""Focused RV32I decode and execution tests."""

from __future__ import annotations

import struct
import unittest

from perf_modeling.config import (
    AcceleratorConfig,
    CoreConfig,
    DRAMConfig,
    MachineConfig,
    ScalarUnitConfig,
)
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine
from perf_modeling.program import Program, ProgramSegment


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a value to Python's integer domain."""
    sign_bit = 1 << (bits - 1)
    return (value ^ sign_bit) - sign_bit


def encode_r_type(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    """Encode one RV32I R-type instruction."""
    return (
        ((funct7 & 0x7F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_i_type(imm: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    """Encode one RV32I I-type instruction."""
    immediate = imm & 0xFFF
    return (
        (immediate << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_s_type(imm: int, rs2: int, rs1: int, funct3: int, opcode: int) -> int:
    """Encode one RV32I S-type instruction."""
    immediate = imm & 0xFFF
    lower = immediate & 0x1F
    upper = (immediate >> 5) & 0x7F
    return (
        (upper << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (lower << 7)
        | (opcode & 0x7F)
    )


def encode_b_type(imm: int, rs2: int, rs1: int, funct3: int, opcode: int) -> int:
    """Encode one RV32I B-type instruction."""
    immediate = imm & 0x1FFF
    bit12 = (immediate >> 12) & 0x1
    bit11 = (immediate >> 11) & 0x1
    bits10_5 = (immediate >> 5) & 0x3F
    bits4_1 = (immediate >> 1) & 0xF
    return (
        (bit12 << 31)
        | (bits10_5 << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (bits4_1 << 8)
        | (bit11 << 7)
        | (opcode & 0x7F)
    )


def encode_u_type(imm: int, rd: int, opcode: int) -> int:
    """Encode one RV32I U-type instruction."""
    return (imm & 0xFFFFF000) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


def encode_j_type(imm: int, rd: int, opcode: int) -> int:
    """Encode one RV32I J-type instruction."""
    immediate = imm & 0x1F_FFFF
    bit20 = (immediate >> 20) & 0x1
    bits19_12 = (immediate >> 12) & 0xFF
    bit11 = (immediate >> 11) & 0x1
    bits10_1 = (immediate >> 1) & 0x3FF
    return (
        (bit20 << 31)
        | (bits10_1 << 21)
        | (bit11 << 20)
        | (bits19_12 << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_csr_type(csr: int, rs1_or_zimm: int, funct3: int, rd: int) -> int:
    """Encode one SYSTEM/CSR instruction."""
    return (
        ((csr & 0xFFF) << 20)
        | ((rs1_or_zimm & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | 0x73
    )


def pack_words(words: list[int]) -> bytes:
    """Pack integer instruction words into a little-endian byte stream."""
    return b"".join(word.to_bytes(4, byteorder="little", signed=False) for word in words)


def build_riscv_elf(segments: list[tuple[int, bytes, int]], entry_point: int) -> bytes:
    """Build a minimal ELF32 executable with the provided loadable segments."""
    program_header_offset = 52
    program_header_size = 32
    e_ident = ELF_MAGIC + bytes((1, 1, 1, 0)) + b"\x00" * 8
    header = struct.pack(
        "<16sHHIIIIIHHHHHH",
        e_ident,
        2,
        243,
        1,
        entry_point,
        program_header_offset,
        0,
        0,
        52,
        program_header_size,
        len(segments),
        0,
        0,
        0,
    )
    offset = 0x100
    program_headers = bytearray()
    payload = bytearray()
    for address, data, flags in segments:
        aligned_offset = (offset + 0x0F) & ~0x0F
        payload.extend(b"\x00" * (aligned_offset - offset))
        program_headers.extend(
            struct.pack(
                "<IIIIIIII",
                1,
                aligned_offset,
                address,
                address,
                len(data),
                len(data),
                flags,
                0x1000,
            )
        )
        payload.extend(data)
        offset = aligned_offset + len(data)
    padding = b"\x00" * (0x100 - len(header) - len(program_headers))
    return header + bytes(program_headers) + padding + bytes(payload)


def build_minimal_riscv_elf(code: bytes, entry_point: int = 0x1000) -> bytes:
    """Build a minimal ELF32 executable with one RX loadable segment."""
    return build_riscv_elf([(entry_point, code, 0x5)], entry_point=entry_point)


ELF_MAGIC = b"\x7fELF"


class RV32ITestCase(unittest.TestCase):
    """Exercise the RV32I functional and timing model."""

    def make_config(self, scalar_latency_cycles: int = 1) -> AcceleratorConfig:
        """Construct a compact test configuration."""
        return AcceleratorConfig(
            core=CoreConfig(
                scalar=ScalarUnitConfig(
                    lanes=1,
                    pipeline_depth=scalar_latency_cycles,
                    queue_depth=4,
                )
            ),
            dram=DRAMConfig(
                capacity_bytes=1 << 20,
                read_latency_cycles=3,
                write_latency_cycles=3,
                bytes_per_cycle=4,
            ),
            machine=MachineConfig(
                reset_pc=0x1000,
                initial_stack_pointer=0x000F_FFFC,
                scratchpad_base_address=0x2000_0000,
                halt_on_ecall=True,
                halt_on_ebreak=True,
                strict_alignment=True,
                instruction_bytes=4,
            ),
        )

    def test_decoder_supports_all_rv32i_mnemonics(self) -> None:
        """The decoder should recognize all RV32I opcodes implemented by the core."""
        samples = {
            "lui": encode_u_type(0x12345000, 1, 0x37),
            "auipc": encode_u_type(0x00012000, 1, 0x17),
            "jal": encode_j_type(8, 1, 0x6F),
            "jalr": encode_i_type(4, 2, 0b000, 1, 0x67),
            "beq": encode_b_type(8, 2, 1, 0b000, 0x63),
            "bne": encode_b_type(8, 2, 1, 0b001, 0x63),
            "blt": encode_b_type(8, 2, 1, 0b100, 0x63),
            "bge": encode_b_type(8, 2, 1, 0b101, 0x63),
            "bltu": encode_b_type(8, 2, 1, 0b110, 0x63),
            "bgeu": encode_b_type(8, 2, 1, 0b111, 0x63),
            "lb": encode_i_type(4, 2, 0b000, 1, 0x03),
            "lh": encode_i_type(4, 2, 0b001, 1, 0x03),
            "lw": encode_i_type(4, 2, 0b010, 1, 0x03),
            "lbu": encode_i_type(4, 2, 0b100, 1, 0x03),
            "lhu": encode_i_type(4, 2, 0b101, 1, 0x03),
            "sb": encode_s_type(4, 3, 2, 0b000, 0x23),
            "sh": encode_s_type(4, 3, 2, 0b001, 0x23),
            "sw": encode_s_type(4, 3, 2, 0b010, 0x23),
            "addi": encode_i_type(-1, 2, 0b000, 1, 0x13),
            "slti": encode_i_type(7, 2, 0b010, 1, 0x13),
            "sltiu": encode_i_type(7, 2, 0b011, 1, 0x13),
            "xori": encode_i_type(7, 2, 0b100, 1, 0x13),
            "ori": encode_i_type(7, 2, 0b110, 1, 0x13),
            "andi": encode_i_type(7, 2, 0b111, 1, 0x13),
            "slli": encode_i_type(3, 2, 0b001, 1, 0x13),
            "srli": encode_i_type(3, 2, 0b101, 1, 0x13),
            "srai": encode_i_type((0x20 << 5) | 3, 2, 0b101, 1, 0x13),
            "add": encode_r_type(0x00, 3, 2, 0b000, 1, 0x33),
            "sub": encode_r_type(0x20, 3, 2, 0b000, 1, 0x33),
            "sll": encode_r_type(0x00, 3, 2, 0b001, 1, 0x33),
            "slt": encode_r_type(0x00, 3, 2, 0b010, 1, 0x33),
            "sltu": encode_r_type(0x00, 3, 2, 0b011, 1, 0x33),
            "xor": encode_r_type(0x00, 3, 2, 0b100, 1, 0x33),
            "srl": encode_r_type(0x00, 3, 2, 0b101, 1, 0x33),
            "sra": encode_r_type(0x20, 3, 2, 0b101, 1, 0x33),
            "or": encode_r_type(0x00, 3, 2, 0b110, 1, 0x33),
            "and": encode_r_type(0x00, 3, 2, 0b111, 1, 0x33),
            "fence": encode_i_type(0, 0, 0b000, 0, 0x0F),
            "csrrw": encode_csr_type(0x340, 2, 0b001, 1),
            "csrrs": encode_csr_type(0x340, 2, 0b010, 1),
            "csrrc": encode_csr_type(0x340, 2, 0b011, 1),
            "csrrwi": encode_csr_type(0x340, 3, 0b101, 1),
            "csrrsi": encode_csr_type(0x340, 3, 0b110, 1),
            "csrrci": encode_csr_type(0x340, 3, 0b111, 1),
            "ecall": 0x00000073,
            "ebreak": 0x00100073,
            "mret": 0x30200073,
        }
        program = Decoder().decode_bytes(pack_words(list(samples.values())), base_address=0x1000)
        decoded_opcodes = [program.instructions[0x1000 + index * 4].opcode for index in range(len(samples))]
        self.assertEqual(decoded_opcodes, list(samples.keys()))

    def test_executes_arithmetic_and_branch_program(self) -> None:
        """The engine should execute scalar ALU and branch control flow correctly."""
        words = [
            encode_u_type(0x12345000, 1, 0x37),
            encode_i_type(0x678, 1, 0b000, 1, 0x13),
            encode_i_type(5, 0, 0b000, 2, 0x13),
            encode_i_type(5, 0, 0b000, 3, 0x13),
            encode_b_type(8, 3, 2, 0b000, 0x63),
            encode_i_type(99, 0, 0b000, 4, 0x13),
            encode_r_type(0x00, 2, 1, 0b000, 5, 0x33),
            encode_i_type(6, 2, 0b010, 6, 0x13),
            encode_r_type(0x00, 1, 2, 0b011, 7, 0x33),
            encode_i_type(7, 0, 0b000, 10, 0x13),
            0x00000073,
        ]
        program = Decoder().decode_bytes(pack_words(words), base_address=0x1000, name="branch-test")
        engine = SimulatorEngine(config=self.make_config(), program=program)
        stats = engine.run(max_cycles=200)
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.exit_code, 7)
        self.assertEqual(engine.state.scalar_regs.read(1), 0x12345678)
        self.assertEqual(engine.state.scalar_regs.read(4), 0)
        self.assertEqual(engine.state.scalar_regs.read(5), 0x1234567D)
        self.assertEqual(engine.state.scalar_regs.read(6), 1)
        self.assertEqual(engine.state.scalar_regs.read(7), 1)
        self.assertIsNone(engine.state.trap_reason)
        self.assertEqual(stats.snapshot()["instructions_retired"], 10)

    def test_executes_load_store_and_sign_extension(self) -> None:
        """The engine should execute DRAM loads and stores with correct byte semantics."""
        words = [
            encode_i_type(0x80, 0, 0b000, 1, 0x13),
            encode_u_type(0x12345000, 2, 0x37),
            encode_i_type(0x678, 2, 0b000, 2, 0x13),
            encode_s_type(0, 2, 1, 0b010, 0x23),
            encode_i_type(0, 1, 0b010, 3, 0x03),
            encode_i_type(-1, 0, 0b000, 4, 0x13),
            encode_s_type(4, 4, 1, 0b000, 0x23),
            encode_i_type(4, 1, 0b000, 5, 0x03),
            encode_i_type(4, 1, 0b100, 6, 0x03),
            0x00100073,
        ]
        program = Decoder().decode_bytes(pack_words(words), base_address=0x1000, name="memory-test")
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.run(max_cycles=400)
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.scalar_regs.read(3), 0x12345678)
        self.assertEqual(engine.state.scalar_regs.read(5), 0xFFFF_FFFF)
        self.assertEqual(engine.state.scalar_regs.read(6), 0xFF)
        self.assertEqual(engine.state.dram.read(0x80, 4), b"\x78\x56\x34\x12")
        self.assertEqual(engine.state.dram.read(0x84, 1), b"\xff")

    def test_reports_dependency_stalls_for_multi_cycle_scalar_pipeline(self) -> None:
        """Dependent instructions should wait for scalar writeback visibility."""
        words = [
            encode_i_type(1, 0, 0b000, 1, 0x13),
            encode_i_type(1, 1, 0b000, 10, 0x13),
            encode_i_type(1, 10, 0b000, 11, 0x13),
            0x00100073,
        ]
        program = Decoder().decode_bytes(pack_words(words), base_address=0x1000, name="timing-test")
        engine = SimulatorEngine(config=self.make_config(scalar_latency_cycles=2), program=program)
        stats = engine.run(max_cycles=200).snapshot()
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.scalar_regs.read(11), 3)
        self.assertEqual(stats["stall_scalar_dependency"], 2)
        self.assertGreaterEqual(stats["cycles"], 8)

    def test_csr_reads_writes_and_dependency_stalls(self) -> None:
        """CSR instructions should read, write, and scoreboard correctly."""
        words = [
            encode_i_type(0x5A, 0, 0b000, 2, 0x13),
            encode_csr_type(0x340, 2, 0b001, 3),
            encode_csr_type(0x340, 0, 0b010, 4),
            encode_csr_type(0x340, 3, 0b110, 5),
            encode_csr_type(0x340, 0, 0b010, 6),
            encode_csr_type(0xF14, 0, 0b010, 7),
            encode_csr_type(0xC00, 0, 0b010, 8),
            encode_csr_type(0xC02, 0, 0b010, 9),
            0x00100073,
        ]
        program = Decoder().decode_bytes(pack_words(words), base_address=0x1000, name="csr-test")
        engine = SimulatorEngine(config=self.make_config(scalar_latency_cycles=2), program=program)
        stats = engine.run(max_cycles=200).snapshot()
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.scalar_regs.read(3), 0)
        self.assertEqual(engine.state.scalar_regs.read(4), 0x5A)
        self.assertEqual(engine.state.scalar_regs.read(5), 0x5A)
        self.assertEqual(engine.state.scalar_regs.read(6), 0x5B)
        self.assertEqual(engine.state.scalar_regs.read(7), 0)
        self.assertGreater(engine.state.scalar_regs.read(8), 0)
        self.assertEqual(engine.state.scalar_regs.read(9), 6)
        self.assertGreaterEqual(stats["stall_csr_dependency"], 2)

    def test_trap_handler_vectors_through_mtvec_and_returns_with_mret(self) -> None:
        """Machine traps should populate CSRs, jump to `mtvec`, and resume via `mret`."""
        main_code = pack_words(
            [
                encode_i_type(0x80, 0, 0b000, 2, 0x13),
                encode_csr_type(0x305, 2, 0b001, 0),
                encode_i_type(1, 0, 0b010, 3, 0x03),
                encode_i_type(0x55, 0, 0b000, 10, 0x13),
                0x00100073,
            ]
        )
        handler_code = pack_words(
            [
                encode_csr_type(0x342, 0, 0b010, 5),
                encode_csr_type(0x341, 0, 0b010, 6),
                encode_csr_type(0x343, 0, 0b010, 7),
                encode_i_type(4, 6, 0b000, 6, 0x13),
                encode_csr_type(0x341, 6, 0b001, 0),
                0x30200073,
            ]
        )
        elf_blob = build_riscv_elf(
            [
                (0x1000, main_code, 0x5),
                (0x0080, handler_code, 0x5),
            ],
            entry_point=0x1000,
        )
        config = self.make_config()
        config = AcceleratorConfig(
            core=config.core,
            registers=config.registers,
            tensors=config.tensors,
            scratchpad=config.scratchpad,
            dram=config.dram,
            machine=MachineConfig(
                reset_pc=0x1000,
                initial_stack_pointer=0x000F_FFFC,
                scratchpad_base_address=0x2000_0000,
                default_mtvec=0x80,
                hart_id=0,
                halt_on_ecall=True,
                halt_on_ebreak=True,
                enable_trap_handlers=True,
                strict_alignment=True,
                instruction_bytes=4,
            ),
            timing=config.timing,
            trace=config.trace,
        )
        program = Decoder().decode_bytes(elf_blob, name="trap-handler")
        engine = SimulatorEngine(config=config, program=program)
        engine.run(max_cycles=200)
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.exit_code, 0x55)
        self.assertEqual(engine.state.scalar_regs.read(5), 4)
        self.assertEqual(engine.state.scalar_regs.read(7), 1)
        self.assertEqual(engine.state.read_csr(0x341, engine.cycle), 0x100C)

    def test_executes_scratchpad_mapped_accesses(self) -> None:
        """Scratchpad-window accesses should round-trip through the mapped SRAM path."""
        program = Program(
            instructions=Decoder().decode_bytes(
                pack_words(
                    [
                        encode_u_type(0x20000000, 1, 0x37),
                        encode_i_type(0x34, 0, 0b000, 2, 0x13),
                        encode_s_type(0, 2, 1, 0b010, 0x23),
                        encode_i_type(0, 1, 0b010, 3, 0x03),
                        0x00100073,
                    ]
                ),
                base_address=0x1000,
                name="scratchpad-test",
            ).instructions,
            segments=(
                ProgramSegment(
                    address=0x1000,
                    data=pack_words(
                        [
                            encode_u_type(0x20000000, 1, 0x37),
                            encode_i_type(0x34, 0, 0b000, 2, 0x13),
                            encode_s_type(0, 2, 1, 0b010, 0x23),
                            encode_i_type(0, 1, 0b010, 3, 0x03),
                            0x00100073,
                        ]
                    ),
                    readable=True,
                    writable=False,
                    executable=True,
                ),
            ),
            entry_point=0x1000,
            name="scratchpad-test",
        )
        engine = SimulatorEngine(config=self.make_config(), program=program)
        stats = engine.run(max_cycles=100).snapshot()
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.scalar_regs.read(3), 0x34)
        self.assertEqual(engine.state.scratchpad.read(0, 4), b"\x34\x00\x00\x00")
        self.assertLess(stats["cycles"], 20)

    def test_loads_and_executes_minimal_elf_image(self) -> None:
        """The decoder should load a minimal ELF32 RV32I binary and run it."""
        code = pack_words(
            [
                encode_i_type(5, 0, 0b000, 10, 0x13),
                0x00000073,
            ]
        )
        elf_blob = build_minimal_riscv_elf(code, entry_point=0x1000)
        program = Decoder().decode_bytes(elf_blob, name="elf-test")
        engine = SimulatorEngine(config=self.make_config(), program=program)
        engine.run(max_cycles=50)
        self.assertTrue(engine.state.halted)
        self.assertEqual(engine.state.exit_code, 5)
        self.assertEqual(program.entry_point, 0x1000)
        self.assertEqual(program.segments[0].data[: len(code)], code)


if __name__ == "__main__":
    unittest.main()
