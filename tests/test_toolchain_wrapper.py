"""Integration tests for the local RV32I toolchain wrapper."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from perf_modeling.config import AcceleratorConfig, DRAMConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


class ToolchainWrapperTestCase(unittest.TestCase):
    """Verify the no-linker RV32I wrapper under `toolchains/`."""

    def test_assemble_to_elf_generates_simulator_consumable_binary(self) -> None:
        """The wrapper should assemble a small RV32I program into a runnable ELF."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "toolchains" / "riscv32" / "assemble_to_elf.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "program.S"
            output = temp_path / "program.elf"
            source.write_text(
                ".globl _start\n"
                "_start:\n"
                "  addi a0, x0, 5\n"
                "  ebreak\n"
            )
            subprocess.run(
                [
                    "python3",
                    str(script),
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                text=True,
                cwd=repo_root,
            )
            program = Decoder().decode_bytes(output.read_bytes(), name="toolchain-wrapper")
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
            engine.run(max_cycles=50)
            self.assertTrue(engine.state.halted)
            self.assertEqual(engine.state.exit_code, 5)


if __name__ == "__main__":
    unittest.main()
