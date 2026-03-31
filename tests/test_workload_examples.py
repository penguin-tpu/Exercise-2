"""Executable example-workload regressions."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from perf_modeling.config import AcceleratorConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


def test_scalar_integer_matmul_example_program_runs() -> None:
    """The example scalar integer matmul assembly should assemble and run end to end."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "toolchains" / "riscv32" / "assemble_to_elf.py"
    source = repo_root / "tests" / "workload" / "scalar_int_matmul.S"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "scalar_int_matmul.elf"
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                str(source),
                "--output",
                str(output),
            ],
            check=True,
            text=True,
            cwd=repo_root,
        )
        program = Decoder().decode_bytes(output.read_bytes(), name="scalar-int-matmul")
        engine = SimulatorEngine(program=program, config=AcceleratorConfig())
        engine.run(max_cycles=2048)

    assert engine.state.halted
    assert engine.state.exit_code == 50
    assert engine.state.scratchpad.read_u32(0) == 19
    assert engine.state.scratchpad.read_u32(4) == 22
    assert engine.state.scratchpad.read_u32(8) == 43
    assert engine.state.scratchpad.read_u32(12) == 50
