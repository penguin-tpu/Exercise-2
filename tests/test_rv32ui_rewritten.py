"""Handwritten RV32I ISA-style regression coverage built from assembly files."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from perf_modeling.config import AcceleratorConfig, DRAMConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


@dataclass(frozen=True)
class MemoryExpectation:
    """One expected post-run scratchpad value."""

    address: int
    """Scratchpad byte address to inspect."""
    size: int
    """Value width in bytes."""
    value: int
    """Expected unsigned little-endian value."""


@dataclass(frozen=True)
class Rv32uiCase:
    """One handwritten RV32I ISA-style regression case."""

    source_name: str
    """Assembly source filename under `tests/isa`."""
    exit_code: int
    """Expected architectural exit code from `ebreak`."""
    max_cycles: int = 128
    """Cycle budget for the simulator run."""
    covered_opcodes: tuple[str, ...] = ()
    """RV32I opcodes intentionally exercised by this source."""
    scratchpad_expectations: tuple[MemoryExpectation, ...] = ()
    """Optional scratchpad values expected after execution."""


def build_and_run(source_path: Path, max_cycles: int) -> SimulatorEngine:
    """Assemble one handwritten ISA test through the toolchain wrapper and run it."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "toolchains" / "riscv32" / "assemble_to_elf.py"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / f"{source_path.stem}.elf"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                str(source_path),
                "--output",
                str(output),
            ],
            check=True,
            text=True,
            cwd=repo_root,
            env=env,
        )
        program = Decoder().decode_bytes(output.read_bytes(), name=source_path.stem)
        engine = SimulatorEngine(
            config=AcceleratorConfig(
                dram=DRAMConfig(
                    capacity_bytes=1 << 20,
                    read_latency_cycles=3,
                    write_latency_cycles=3,
                    bytes_per_cycle=16,
                )
            ),
            program=program,
        )
        engine.run(max_cycles=max_cycles)
        return engine


def read_scratchpad(engine: SimulatorEngine, expectation: MemoryExpectation) -> int:
    """Read one value from the modeled scratchpad using the expected width."""
    if expectation.size == 1:
        return engine.state.scratchpad.read_u8(expectation.address)
    if expectation.size == 2:
        return engine.state.scratchpad.read_u16(expectation.address)
    if expectation.size == 4:
        return engine.state.scratchpad.read_u32(expectation.address)
    raise ValueError(f"Unsupported scratchpad width {expectation.size}.")


RV32I_BASE_OPCODES = {
    "lui",
    "auipc",
    "jal",
    "jalr",
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "lb",
    "lh",
    "lw",
    "lbu",
    "lhu",
    "sb",
    "sh",
    "sw",
    "addi",
    "slti",
    "sltiu",
    "xori",
    "ori",
    "andi",
    "slli",
    "srli",
    "srai",
    "add",
    "sub",
    "sll",
    "slt",
    "sltu",
    "xor",
    "srl",
    "sra",
    "or",
    "and",
    "fence",
    "ecall",
    "ebreak",
}


CASES = (
    Rv32uiCase(source_name="addi.S", exit_code=15, covered_opcodes=("addi",)),
    Rv32uiCase(source_name="add.S", exit_code=13, covered_opcodes=("add",)),
    Rv32uiCase(source_name="and.S", exit_code=40, covered_opcodes=("and",)),
    Rv32uiCase(source_name="andi.S", exit_code=12, covered_opcodes=("andi",)),
    Rv32uiCase(source_name="auipc.S", exit_code=7, covered_opcodes=("auipc",)),
    Rv32uiCase(source_name="beq.S", exit_code=11, covered_opcodes=("beq",)),
    Rv32uiCase(source_name="bge.S", exit_code=14, covered_opcodes=("bge",)),
    Rv32uiCase(source_name="bgeu.S", exit_code=16, covered_opcodes=("bgeu",)),
    Rv32uiCase(source_name="blt.S", exit_code=13, covered_opcodes=("blt",)),
    Rv32uiCase(source_name="bltu.S", exit_code=15, covered_opcodes=("bltu",)),
    Rv32uiCase(source_name="bne.S", exit_code=12, covered_opcodes=("bne",)),
    Rv32uiCase(source_name="ebreak.S", exit_code=23, covered_opcodes=("ebreak",)),
    Rv32uiCase(source_name="ecall.S", exit_code=17, covered_opcodes=("ecall",)),
    Rv32uiCase(source_name="fence.S", exit_code=33, covered_opcodes=("fence",)),
    Rv32uiCase(source_name="jal.S", exit_code=15, covered_opcodes=("jal",)),
    Rv32uiCase(source_name="jalr.S", exit_code=18, covered_opcodes=("jalr",)),
    Rv32uiCase(source_name="lb.S", exit_code=1, covered_opcodes=("lb",)),
    Rv32uiCase(source_name="lbu.S", exit_code=255, covered_opcodes=("lbu",)),
    Rv32uiCase(source_name="lh.S", exit_code=4660, covered_opcodes=("lh",)),
    Rv32uiCase(source_name="lhu.S", exit_code=4660, covered_opcodes=("lhu",)),
    Rv32uiCase(source_name="lui.S", exit_code=4096, covered_opcodes=("lui",)),
    Rv32uiCase(
        source_name="lw.S",
        exit_code=4916,
        covered_opcodes=("lw",),
        scratchpad_expectations=(
            MemoryExpectation(address=0, size=1, value=0x80),
            MemoryExpectation(address=2, size=2, value=0x1234),
            MemoryExpectation(address=4, size=4, value=4916),
        ),
    ),
    Rv32uiCase(source_name="or.S", exit_code=51, covered_opcodes=("or",)),
    Rv32uiCase(source_name="ori.S", exit_code=51, covered_opcodes=("ori",)),
    Rv32uiCase(source_name="sll.S", exit_code=48, covered_opcodes=("sll",)),
    Rv32uiCase(source_name="slli.S", exit_code=48, covered_opcodes=("slli",)),
    Rv32uiCase(source_name="slti.S", exit_code=1, covered_opcodes=("slti",)),
    Rv32uiCase(source_name="sltiu.S", exit_code=1, covered_opcodes=("sltiu",)),
    Rv32uiCase(source_name="slt.S", exit_code=1, covered_opcodes=("slt",)),
    Rv32uiCase(source_name="sltu.S", exit_code=1, covered_opcodes=("sltu",)),
    Rv32uiCase(source_name="sra.S", exit_code=8, covered_opcodes=("sra",)),
    Rv32uiCase(source_name="srai.S", exit_code=8, covered_opcodes=("srai",)),
    Rv32uiCase(source_name="srl.S", exit_code=8, covered_opcodes=("srl",)),
    Rv32uiCase(source_name="srli.S", exit_code=8, covered_opcodes=("srli",)),
    Rv32uiCase(source_name="sb.S", exit_code=52, covered_opcodes=("sb",)),
    Rv32uiCase(source_name="sh.S", exit_code=4660, covered_opcodes=("sh",)),
    Rv32uiCase(source_name="sub.S", exit_code=12, covered_opcodes=("sub",)),
    Rv32uiCase(source_name="sw.S", exit_code=852, covered_opcodes=("sw",)),
    Rv32uiCase(source_name="xor.S", exit_code=5, covered_opcodes=("xor",)),
    Rv32uiCase(source_name="xori.S", exit_code=5, covered_opcodes=("xori",)),
)


@pytest.mark.parametrize("case", CASES, ids=[case.source_name for case in CASES])
def test_rewritten_rv32ui_cases(case: Rv32uiCase) -> None:
    """Assemble and execute the handwritten RV32I ISA-style regression cases."""
    source_path = Path(__file__).resolve().parent / "isa" / case.source_name
    engine = build_and_run(source_path, max_cycles=case.max_cycles)
    assert engine.state.halted
    assert engine.state.trap_reason is None
    assert engine.state.exit_code == case.exit_code
    for expectation in case.scratchpad_expectations:
        assert read_scratchpad(engine, expectation) == expectation.value


def test_rewritten_rv32ui_cases_cover_complete_rv32i_base_instruction_set() -> None:
    """The handwritten ISA corpus should collectively cover the complete RV32I base instruction set."""
    covered = {opcode for case in CASES for opcode in case.covered_opcodes}
    assert covered == RV32I_BASE_OPCODES


def test_rewritten_rv32ui_cases_use_one_source_per_opcode() -> None:
    """Each handwritten ISA case should now map one source file to one claimed opcode."""
    for case in CASES:
        assert len(case.covered_opcodes) == 1
        assert Path(case.source_name).stem == case.covered_opcodes[0]


def test_rewritten_rv32ui_case_sources_contain_their_claimed_opcode() -> None:
    """Each handwritten ISA source should explicitly emit the opcode it claims to cover."""
    isa_dir = Path(__file__).resolve().parent / "isa"
    for case in CASES:
        source_path = isa_dir / case.source_name
        source_text = source_path.read_text()
        opcode = case.covered_opcodes[0]
        assert re.search(rf"\b{re.escape(opcode)}\b", source_text)
