"""Assemble one RV32I assembly file into an executable ELF32 image with GNU binutils."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolchains.riscv32.gnu_toolchain import repo_root_from_file, resolve_toolchain


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Assemble RV32I assembly into an ELF32 image.")
    parser.add_argument("source", type=Path, help="Path to the input RV32I assembly file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output ELF file.")
    parser.add_argument(
        "--base-address",
        type=lambda value: int(value, 0),
        default=0x1000,
        help="Virtual address used for the executable text segment.",
    )
    return parser.parse_args()


def assemble_to_object(source: Path, output: Path, assembler: Path) -> None:
    """Assemble one RV32I assembly file into a relocatable object."""
    subprocess.run(
        [
            str(assembler),
            "-march=rv32i",
            "-mabi=ilp32",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        text=True,
    )


def link_to_elf(object_path: Path, output: Path, linker: Path, base_address: int) -> None:
    """Link one RV32I object into a runnable ELF32 image."""
    subprocess.run(
        [
            str(linker),
            "-m",
            "elf32lriscv",
            "--no-relax",
            "-Ttext",
            hex(base_address),
            "-e",
            "_start",
            str(object_path),
            "-o",
            str(output),
        ],
        check=True,
        text=True,
    )


def main() -> None:
    """Assemble and link one RV32I assembly file into an ELF32 executable."""
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__))
    toolchain = resolve_toolchain(repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        object_path = Path(temp_dir) / "program.o"
        assemble_to_object(args.source, object_path, toolchain.assembler)
        link_to_elf(object_path, args.output, toolchain.linker, args.base_address)


if __name__ == "__main__":
    main()
