"""Handwritten RV32I ISA-style regression coverage built from assembly files."""

from __future__ import annotations

import os
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
                "python3",
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


CASES = (
    Rv32uiCase(source_name="rv32ui_arithmetic.s", exit_code=23),
    Rv32uiCase(source_name="rv32ui_branch_compare.s", exit_code=15),
    Rv32uiCase(source_name="rv32ui_control_flow.s", exit_code=15),
    Rv32uiCase(
        source_name="rv32ui_memory.s",
        exit_code=4916,
        scratchpad_expectations=(
            MemoryExpectation(address=0, size=1, value=0x80),
            MemoryExpectation(address=2, size=2, value=0x1234),
            MemoryExpectation(address=4, size=4, value=4916),
        ),
    ),
    Rv32uiCase(source_name="rv32ui_shift_logic.s", exit_code=59),
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
