"""Executable example-workload regressions."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from perf_modeling.config import AcceleratorConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


def compile_and_run_workload(source_name: str) -> SimulatorEngine:
    """Compile one workload example source and run it to completion."""
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


def assert_scalar_integer_matmul_results(engine: SimulatorEngine) -> None:
    """Assert the expected scratchpad contents for the scalar integer matmul example."""
    assert engine.state.halted
    assert engine.state.exit_code == 50
    assert engine.state.scratchpad.read_u32(0) == 19
    assert engine.state.scratchpad.read_u32(4) == 22
    assert engine.state.scratchpad.read_u32(8) == 43
    assert engine.state.scratchpad.read_u32(12) == 50


def assert_scalar_vector_add_results(engine: SimulatorEngine) -> None:
    """Assert the expected scratchpad contents for the scalar vector-add example."""
    assert engine.state.halted
    assert engine.state.exit_code == 44
    assert engine.state.scratchpad.read_u32(0) == 11
    assert engine.state.scratchpad.read_u32(4) == 22
    assert engine.state.scratchpad.read_u32(8) == 33
    assert engine.state.scratchpad.read_u32(12) == 44


def assert_scalar_reduce_sum_results(engine: SimulatorEngine) -> None:
    """Assert the expected scratchpad contents for the scalar reduction example."""
    assert engine.state.halted
    assert engine.state.exit_code == 20
    assert engine.state.scratchpad.read_u32(0) == 20


def assert_scalar_relu_results(engine: SimulatorEngine) -> None:
    """Assert the expected scratchpad contents for the scalar ReLU example."""
    assert engine.state.halted
    assert engine.state.exit_code == 0
    assert engine.state.scratchpad.read_u32(0) == 0
    assert engine.state.scratchpad.read_u32(4) == 0
    assert engine.state.scratchpad.read_u32(8) == 5
    assert engine.state.scratchpad.read_u32(12) == 0


def test_scalar_integer_matmul_example_assembly_program_runs() -> None:
    """The scalar integer matmul assembly example should run end to end."""
    assert_scalar_integer_matmul_results(compile_and_run_workload("scalar_int_matmul.S"))


def test_scalar_integer_matmul_example_c_program_runs() -> None:
    """The scalar integer matmul C example should run end to end."""
    assert_scalar_integer_matmul_results(compile_and_run_workload("scalar_int_matmul.c"))


def test_scalar_vector_add_example_assembly_program_runs() -> None:
    """The vector-add assembly example should run end to end."""
    assert_scalar_vector_add_results(compile_and_run_workload("scalar_vector_add.S"))


def test_scalar_vector_add_example_c_program_runs() -> None:
    """The vector-add C example should run end to end."""
    assert_scalar_vector_add_results(compile_and_run_workload("scalar_vector_add.c"))


def test_scalar_reduce_sum_example_assembly_program_runs() -> None:
    """The reduction assembly example should run end to end."""
    assert_scalar_reduce_sum_results(compile_and_run_workload("scalar_reduce_sum.S"))


def test_scalar_reduce_sum_example_c_program_runs() -> None:
    """The reduction C example should run end to end."""
    assert_scalar_reduce_sum_results(compile_and_run_workload("scalar_reduce_sum.c"))


def test_scalar_relu_example_assembly_program_runs() -> None:
    """The ReLU assembly example should run end to end."""
    assert_scalar_relu_results(compile_and_run_workload("scalar_relu.S"))


def test_scalar_relu_example_c_program_runs() -> None:
    """The ReLU C example should run end to end."""
    assert_scalar_relu_results(compile_and_run_workload("scalar_relu.c"))
