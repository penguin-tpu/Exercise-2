"""Executable example-workload regressions."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from perf_modeling.config import AcceleratorConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


def assemble_and_run_workload(source_name: str) -> SimulatorEngine:
    """Assemble one workload example and run it to completion."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "toolchains" / "riscv32" / "assemble_to_elf.py"
    source = repo_root / "tests" / "workload" / source_name
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / f"{source.stem}.elf"
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
        program = Decoder().decode_bytes(output.read_bytes(), name=source.stem)
        engine = SimulatorEngine(program=program, config=AcceleratorConfig())
        engine.run(max_cycles=2048)
        return engine


def test_scalar_integer_matmul_example_program_runs() -> None:
    """The example scalar integer matmul assembly should assemble and run end to end."""
    engine = assemble_and_run_workload("scalar_int_matmul.S")
    assert engine.state.halted
    assert engine.state.exit_code == 50
    assert engine.state.scratchpad.read_u32(0) == 19
    assert engine.state.scratchpad.read_u32(4) == 22
    assert engine.state.scratchpad.read_u32(8) == 43
    assert engine.state.scratchpad.read_u32(12) == 50


def test_scalar_vector_add_example_program_runs() -> None:
    """The vector-add example should add two small vectors into scratchpad output."""
    engine = assemble_and_run_workload("scalar_vector_add.S")

    assert engine.state.halted
    assert engine.state.exit_code == 44
    assert engine.state.scratchpad.read_u32(0) == 11
    assert engine.state.scratchpad.read_u32(4) == 22
    assert engine.state.scratchpad.read_u32(8) == 33
    assert engine.state.scratchpad.read_u32(12) == 44


def test_scalar_reduce_sum_example_program_runs() -> None:
    """The reduction example should accumulate all input elements into one output sum."""
    engine = assemble_and_run_workload("scalar_reduce_sum.S")

    assert engine.state.halted
    assert engine.state.exit_code == 20
    assert engine.state.scratchpad.read_u32(0) == 20


def test_scalar_relu_example_program_runs() -> None:
    """The ReLU example should clamp negative inputs to zero in the output vector."""
    engine = assemble_and_run_workload("scalar_relu.S")

    assert engine.state.halted
    assert engine.state.exit_code == 0
    assert engine.state.scratchpad.read_u32(0) == 0
    assert engine.state.scratchpad.read_u32(4) == 0
    assert engine.state.scratchpad.read_u32(8) == 5
    assert engine.state.scratchpad.read_u32(12) == 0
