"""Integration tests for the GNU RV32I toolchain wrapper."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from perf_modeling.config import AcceleratorConfig, DRAMConfig
from perf_modeling.decode import Decoder
from perf_modeling.engine import SimulatorEngine


class TestToolchainWrapper:
    """Verify the GNU-binutils-backed RV32I wrapper under `toolchains/`."""

    def make_config(self) -> AcceleratorConfig:
        """Construct a compact configuration for toolchain-wrapper integration tests."""
        return AcceleratorConfig(
            dram=DRAMConfig(
                capacity_bytes=1 << 20,
                read_latency_cycles=3,
                write_latency_cycles=3,
                bytes_per_cycle=16,
            )
        )

    def build_and_run(
        self,
        assembly_source: str,
        max_cycles: int = 100,
        config: AcceleratorConfig | None = None,
    ) -> SimulatorEngine:
        """Assemble one RV32I program through the wrapper and run it in the simulator."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "toolchains" / "riscv32" / "assemble_to_elf.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "program.S"
            output = temp_path / "program.elf"
            source.write_text(assembly_source)
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
                config=config if config is not None else self.make_config(),
                program=program,
            )
            engine.run(max_cycles=max_cycles)
            return engine

    def test_assemble_to_elf_generates_simulator_consumable_binary(self) -> None:
        """The wrapper should assemble a small RV32I program into a runnable ELF."""
        engine = self.build_and_run(
            ".globl _start\n"
            "_start:\n"
            "  addi a0, x0, 5\n"
            "  ebreak\n"
        )
        assert engine.state.halted
        assert engine.state.exit_code == 5

    def test_assemble_to_elf_supports_preprocessed_sources(self) -> None:
        """The wrapper should accept `.S` sources that rely on the GNU preprocessor."""
        engine = self.build_and_run(
            "#define EXIT_CODE 9\n"
            ".globl _start\n"
            "_start:\n"
            "  addi a0, x0, EXIT_CODE\n"
            "  ebreak\n"
        )
        assert engine.state.halted
        assert engine.state.exit_code == 9

    def test_assemble_to_elf_handles_local_labels_and_branch_loops(self) -> None:
        """The wrapper should support self-contained assembly with local control flow."""
        engine = self.build_and_run(
            ".globl _start\n"
            "_start:\n"
            "  addi t0, x0, 5\n"
            "  addi t1, x0, 0\n"
            "loop:\n"
            "  add t1, t1, t0\n"
            "  addi t0, t0, -1\n"
            "  bne t0, x0, loop\n"
            "  addi a0, t1, 0\n"
            "  ebreak\n"
        )
        assert engine.state.halted
        assert engine.state.exit_code == 15

    def test_assemble_to_elf_handles_scratchpad_mapped_load_store(self) -> None:
        """The wrapper should generate ELFs that exercise the scratchpad memory window."""
        engine = self.build_and_run(
            ".globl _start\n"
            "_start:\n"
            "  lui t0, 0x20000\n"
            "  addi t1, x0, 52\n"
            "  sw t1, 0(t0)\n"
            "  lw a0, 0(t0)\n"
            "  ebreak\n"
        )
        assert engine.state.halted
        assert engine.state.exit_code == 52

    def test_assemble_to_elf_handles_data_and_bss_sections(self) -> None:
        """The wrapper should link ELF images with initialized and zero-initialized data."""
        engine = self.build_and_run(
            ".section .data\n"
            "seed:\n"
            "  .word 11\n"
            ".section .bss\n"
            ".align 2\n"
            "scratch:\n"
            "  .space 4\n"
            ".section .text\n"
            ".globl _start\n"
            "_start:\n"
            "  la t0, seed\n"
            "  lw t1, 0(t0)\n"
            "  la t2, scratch\n"
            "  lw t3, 0(t2)\n"
            "  add a0, t1, t3\n"
            "  sw a0, 0(t2)\n"
            "  lw a0, 0(t2)\n"
            "  ebreak\n",
            max_cycles=200,
        )
        assert engine.state.halted
        assert engine.state.exit_code == 11

    def test_assemble_to_elf_runs_trap_handler_and_mret_flow(self) -> None:
        """The wrapper should support ELF programs that install trap handlers and return with `mret`."""
        machine = replace(
            self.make_config().machine,
            halt_on_ecall=False,
            enable_trap_handlers=True,
        )
        engine = self.build_and_run(
            ".section .text\n"
            ".globl _start\n"
            "_start:\n"
            "  la t0, handler\n"
            "  csrw mtvec, t0\n"
            "  ecall\n"
            "  addi a0, t2, 2\n"
            "  ebreak\n"
            "handler:\n"
            "  csrr t2, mcause\n"
            "  csrr t1, mepc\n"
            "  addi t1, t1, 4\n"
            "  csrw mepc, t1\n"
            "  mret\n",
            max_cycles=200,
            config=replace(self.make_config(), machine=machine),
        )
        assert engine.state.halted
        assert engine.state.exit_code == 13
