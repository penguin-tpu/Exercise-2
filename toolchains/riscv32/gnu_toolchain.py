"""Helpers for resolving a repo-local GNU RISC-V binutils toolchain."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PACKAGE_NAME = "binutils-riscv64-unknown-elf"
GCC_PACKAGE_NAME = "gcc-riscv64-unknown-elf"
ASSEMBLER_NAME = "riscv64-unknown-elf-as"
LINKER_NAME = "riscv64-unknown-elf-ld"
OBJCOPY_NAME = "riscv64-unknown-elf-objcopy"
COMPILER_NAME = "riscv64-unknown-elf-gcc"


@dataclass(frozen=True)
class GnuToolchain:
    """Resolved GNU RISC-V binutils paths."""

    compiler: Path
    """Path to the GNU compiler driver"""
    assembler: Path
    """Path to the GNU assembler binary"""
    linker: Path
    """Path to the GNU linker binary"""
    objcopy: Path
    """Path to the GNU objcopy binary"""


def repo_root_from_file(path: Path) -> Path:
    """Return the repository root from one file under `toolchains/riscv32/`."""
    return path.resolve().parents[2]


def repo_local_toolchain_root(repo_root: Path) -> Path:
    """Return the repo-local GNU binutils extraction directory."""
    return repo_root / "toolchains" / "riscv32" / "gnu-binutils"


def repo_local_download_root(repo_root: Path) -> Path:
    """Return the repo-local cache directory for downloaded toolchain packages."""
    return repo_root / "toolchains" / "riscv32" / "downloads"


def resolve_toolchain(repo_root: Path) -> GnuToolchain:
    """Resolve GNU RISC-V binutils from the repo-local cache or PATH, bootstrapping locally when needed."""
    toolchain = resolve_from_root(repo_local_toolchain_root(repo_root))
    if toolchain is not None:
        return toolchain
    toolchain = resolve_from_path()
    if toolchain is not None:
        return toolchain
    bootstrap_repo_local_binutils(repo_root)
    toolchain = resolve_from_root(repo_local_toolchain_root(repo_root))
    if toolchain is None:
        raise FileNotFoundError(
            "GNU RISC-V binutils are unavailable after bootstrap. "
            "Expected riscv64-unknown-elf-as/ld/objcopy under toolchains/riscv32/gnu-binutils or on PATH."
        )
    return toolchain


def resolve_from_root(root: Path) -> GnuToolchain | None:
    """Resolve GNU RISC-V binutils from one extracted package root."""
    bin_dir = root / "usr" / "bin"
    compiler = bin_dir / COMPILER_NAME
    assembler = bin_dir / ASSEMBLER_NAME
    linker = bin_dir / LINKER_NAME
    objcopy = bin_dir / OBJCOPY_NAME
    if not compiler.exists() or not assembler.exists() or not linker.exists() or not objcopy.exists():
        return None
    return GnuToolchain(compiler=compiler, assembler=assembler, linker=linker, objcopy=objcopy)


def resolve_from_path() -> GnuToolchain | None:
    """Resolve GNU RISC-V binutils from the host PATH."""
    compiler = shutil.which(COMPILER_NAME)
    assembler = shutil.which(ASSEMBLER_NAME)
    linker = shutil.which(LINKER_NAME)
    objcopy = shutil.which(OBJCOPY_NAME)
    if compiler is None or assembler is None or linker is None or objcopy is None:
        return None
    return GnuToolchain(
        compiler=Path(compiler),
        assembler=Path(assembler),
        linker=Path(linker),
        objcopy=Path(objcopy),
    )


def bootstrap_repo_local_binutils(repo_root: Path) -> Path:
    """Download and unpack repo-local GNU RISC-V toolchain binaries under `toolchains/riscv32/` without root."""
    download_root = repo_local_download_root(repo_root)
    toolchain_root = repo_local_toolchain_root(repo_root)
    download_root.mkdir(parents=True, exist_ok=True)
    toolchain_root.mkdir(parents=True, exist_ok=True)
    for package_name in (PACKAGE_NAME, GCC_PACKAGE_NAME):
        subprocess.run(
            ["apt", "download", package_name],
            check=True,
            cwd=download_root,
            text=True,
            capture_output=True,
        )
        package_paths = sorted(download_root.glob(f"{package_name}_*.deb"))
        if not package_paths:
            raise FileNotFoundError(f"Unable to find downloaded {package_name} package under {download_root}.")
        subprocess.run(
            ["dpkg-deb", "-x", str(package_paths[-1]), str(toolchain_root)],
            check=True,
            text=True,
            capture_output=True,
        )
    return toolchain_root
