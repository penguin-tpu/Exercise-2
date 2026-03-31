"""Compile one freestanding RV32I source file into an executable ELF32 image."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolchains.riscv32.gnu_toolchain import repo_root_from_file, resolve_toolchain


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compile a freestanding RV32I source file into an ELF32 image.")
    parser.add_argument("source", type=Path, help="Path to the input RV32I assembly or C source file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output ELF file.")
    parser.add_argument(
        "--base-address",
        type=lambda value: int(value, 0),
        default=0x1000,
        help="Virtual address used for the executable text segment.",
    )
    return parser.parse_args()


def compile_to_elf(source: Path, output: Path, compiler: Path, base_address: int) -> None:
    """Compile and link one freestanding RV32I source into a runnable ELF32 image."""
    subprocess.run(
        [
            str(compiler),
            "-march=rv32i_zicsr",
            "-mabi=ilp32",
            "-O2",
            "-ffreestanding",
            "-fno-pic",
            "-msmall-data-limit=0",
            "-nostdlib",
            "-nostartfiles",
            "-nodefaultlibs",
            "-static",
            "-Wl,--no-relax",
            f"-Wl,-Ttext,{hex(base_address)}",
            "-Wl,-e,_start",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        text=True,
    )


def main() -> None:
    """Compile and link one freestanding RV32I source file into an ELF32 executable."""
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__))
    toolchain = resolve_toolchain(repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compile_to_elf(args.source, args.output, toolchain.compiler, args.base_address)


if __name__ == "__main__":
    main()
