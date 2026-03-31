"""Assemble a self-contained RV32I assembly file into a minimal executable ELF."""

from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

ELF_MAGIC = b"\x7fELF"
SHT_STRTAB = 3
SHT_RELA = 4
SHT_REL = 9


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Assemble RV32I assembly into a minimal ELF32 image.")
    parser.add_argument("source", type=Path, help="Path to the input RV32I assembly file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output ELF file.")
    parser.add_argument(
        "--base-address",
        type=lambda value: int(value, 0),
        default=0x1000,
        help="Virtual address used for the executable text segment.",
    )
    parser.add_argument(
        "--clang",
        default="clang",
        help="Assembler executable used for the RV32I assembly step.",
    )
    return parser.parse_args()


def assemble_to_object(source: Path, output: Path, clang: str) -> None:
    """Assemble one RV32I assembly file into a relocatable object."""
    subprocess.run(
        [
            clang,
            "--target=riscv32",
            "-march=rv32i",
            "-mabi=ilp32",
            "-c",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        text=True,
    )


def extract_text_section(blob: bytes) -> bytes:
    """Extract the `.text` section from a relocatable ELF32 object."""
    if len(blob) < 52 or blob[:4] != ELF_MAGIC:
        raise ValueError("Input object is not an ELF32 file.")
    if blob[4] != 1 or blob[5] != 1:
        raise ValueError("Only little-endian ELF32 objects are supported.")
    section_header_offset = struct.unpack_from("<I", blob, 32)[0]
    section_header_entry_size = struct.unpack_from("<H", blob, 46)[0]
    section_header_count = struct.unpack_from("<H", blob, 48)[0]
    section_name_index = struct.unpack_from("<H", blob, 50)[0]
    section_headers: list[tuple[int, int, int, int, int]] = []
    for index in range(section_header_count):
        offset = section_header_offset + index * section_header_entry_size
        (
            name_offset,
            section_type,
            _flags,
            _address,
            section_offset,
            size,
            _link,
            _info,
            _address_align,
            _entry_size,
        ) = struct.unpack_from("<IIIIIIIIII", blob, offset)
        section_headers.append((name_offset, section_type, section_offset, size, index))
    strtab_header = section_headers[section_name_index]
    if strtab_header[1] != SHT_STRTAB:
        raise ValueError("Object file is missing a section-name string table.")
    strtab = blob[strtab_header[2] : strtab_header[2] + strtab_header[3]]
    for name_offset, section_type, _section_offset, _size, section_index in section_headers:
        section_name_end = strtab.find(b"\x00", name_offset)
        section_name = strtab[name_offset:section_name_end].decode("utf-8")
        if section_type in {SHT_REL, SHT_RELA}:
            raise ValueError(
                f"Relocation section {section_name!r} is present; this no-linker wrapper only supports self-contained assembly."
            )
        if section_name == ".text":
            return blob[_section_offset : _section_offset + _size]
    raise ValueError("Object file does not contain a .text section.")


def build_executable_elf(text: bytes, base_address: int) -> bytes:
    """Wrap raw text bytes into a minimal executable ELF32 image."""
    program_header_offset = 52
    program_header_size = 32
    text_offset = 0x100
    e_ident = ELF_MAGIC + bytes((1, 1, 1, 0)) + b"\x00" * 8
    header = struct.pack(
        "<16sHHIIIIIHHHHHH",
        e_ident,
        2,
        243,
        1,
        base_address,
        program_header_offset,
        0,
        0,
        52,
        program_header_size,
        1,
        0,
        0,
        0,
    )
    program_header = struct.pack(
        "<IIIIIIII",
        1,
        text_offset,
        base_address,
        base_address,
        len(text),
        len(text),
        0x5,
        0x1000,
    )
    padding = b"\x00" * (text_offset - len(header) - len(program_header))
    return header + program_header + padding + text


def main() -> None:
    """Assemble an RV32I assembly file and emit a minimal ELF32 executable."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        object_path = Path(temp_dir) / "program.o"
        assemble_to_object(args.source, object_path, args.clang)
        text = extract_text_section(object_path.read_bytes())
        args.output.write_bytes(build_executable_elf(text, args.base_address))


if __name__ == "__main__":
    main()
