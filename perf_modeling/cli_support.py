"""Helper utilities shared by the simulator CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from perf_modeling.decode import Decoder
from toolchains.riscv32.assemble_to_elf import compile_to_elf
from toolchains.riscv32.gnu_toolchain import resolve_toolchain


ASSEMBLY_SUFFIXES = frozenset({".s", ".asm"})
"""File suffixes treated as assembly sources for transient ELF assembly."""


def parse_image_load_spec(value: str) -> tuple[int, Path]:
    """Parse one CLI memory-image load specification of the form `ADDRESS:PATH`."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected memory load spec in the form ADDRESS:PATH.")
    address_text, path_text = value.split(":", 1)
    return int(address_text, 0), Path(path_text)


def parse_int_like(value: object, field_name: str) -> int:
    """Parse one manifest integer field from either numeric or string JSON input."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError(f"Expected integer-like value for {field_name}.")


def load_memory_manifest(manifest_path: Path) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    """Load DRAM and scratchpad preload specs from one JSON manifest file."""
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Memory load manifest must contain a top-level JSON object.")
    manifest_root = manifest_path.parent
    dram_loads: list[tuple[int, Path]] = []
    for entry in payload.get("dram", []):
        if not isinstance(entry, dict):
            raise ValueError("Each DRAM manifest entry must be a JSON object.")
        address = parse_int_like(entry.get("address"), "dram.address")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        dram_loads.append((address, path))
    scratchpad_loads: list[tuple[int, Path]] = []
    for entry in payload.get("scratchpad", []):
        if not isinstance(entry, dict):
            raise ValueError("Each scratchpad manifest entry must be a JSON object.")
        offset = parse_int_like(entry.get("offset"), "scratchpad.offset")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        scratchpad_loads.append((offset, path))
    return dram_loads, scratchpad_loads


def path_is_under_directory(path: Path, directory: Path) -> bool:
    """Return whether one relative path already begins under the selected output directory."""
    if len(path.parts) < len(directory.parts):
        return False
    return path.parts[: len(directory.parts)] == directory.parts


def prepare_output_path(path_text: str, output_dir: str) -> Path:
    """Create parent directories for one file output target and return the normalized path."""
    path = Path(path_text)
    output_root = Path(output_dir)
    if not path.is_absolute() and not path_is_under_directory(path, output_root):
        path = output_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def decode_program_from_path(
    program_path: Path,
    base_address: int,
    decoder: Decoder,
    repo_root: Path,
):
    """Decode one user-supplied program path, auto-assembling assembly sources first."""
    if program_path.suffix.lower() in ASSEMBLY_SUFFIXES:
        toolchain = resolve_toolchain(repo_root)
        with tempfile.TemporaryDirectory() as temp_dir:
            elf_path = Path(temp_dir) / f"{program_path.stem}.elf"
            compile_to_elf(program_path, elf_path, toolchain.compiler, base_address)
            return decoder.decode_bytes(elf_path.read_bytes(), name=program_path.name)
    return decoder.decode_bytes(
        program_path.read_bytes(),
        base_address=base_address,
        name=program_path.name,
    )
