"""Helpers for resolving a repo-local GNU RISC-V binutils toolchain."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PACKAGE_NAME = "binutils-riscv64-unknown-elf"
ASSEMBLER_NAME = "riscv64-unknown-elf-as"
LINKER_NAME = "riscv64-unknown-elf-ld"
OBJCOPY_NAME = "riscv64-unknown-elf-objcopy"


@dataclass(frozen=True)
class GnuToolchain:
    """Resolved GNU RISC-V binutils paths."""

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
    assembler = bin_dir / ASSEMBLER_NAME
    linker = bin_dir / LINKER_NAME
    objcopy = bin_dir / OBJCOPY_NAME
    if not assembler.exists() or not linker.exists() or not objcopy.exists():
        return None
    return GnuToolchain(assembler=assembler, linker=linker, objcopy=objcopy)


def resolve_from_path() -> GnuToolchain | None:
    """Resolve GNU RISC-V binutils from the host PATH."""
    assembler = shutil.which(ASSEMBLER_NAME)
    linker = shutil.which(LINKER_NAME)
    objcopy = shutil.which(OBJCOPY_NAME)
    if assembler is None or linker is None or objcopy is None:
        return None
    return GnuToolchain(assembler=Path(assembler), linker=Path(linker), objcopy=Path(objcopy))


def bootstrap_repo_local_binutils(repo_root: Path) -> Path:
    """Download and unpack GNU RISC-V binutils under `toolchains/riscv32/` without root privileges."""
    download_root = repo_local_download_root(repo_root)
    toolchain_root = repo_local_toolchain_root(repo_root)
    download_root.mkdir(parents=True, exist_ok=True)
    toolchain_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["apt", "download", PACKAGE_NAME],
        check=True,
        cwd=download_root,
        text=True,
        capture_output=True,
    )
    package_paths = sorted(download_root.glob(f"{PACKAGE_NAME}_*.deb"))
    if not package_paths:
        raise FileNotFoundError(f"Unable to find downloaded {PACKAGE_NAME} package under {download_root}.")
    subprocess.run(
        ["dpkg-deb", "-x", str(package_paths[-1]), str(toolchain_root)],
        check=True,
        text=True,
        capture_output=True,
    )
    return toolchain_root
