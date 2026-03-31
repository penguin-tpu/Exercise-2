"""Program decoding entry points."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from perf_modeling.isa.instruction import Instruction
from perf_modeling.program import Program, ProgramSegment

ELF_MAGIC = b"\x7fELF"
PT_LOAD = 1
PF_X = 0x1
PF_W = 0x2
PF_R = 0x4
SHT_SYMTAB = 2
EM_RISCV = 243


@dataclass(frozen=True)
class ElfSectionHeader:
    """ELF32 section header metadata used for symbol extraction."""

    name_offset: int
    section_type: int
    offset: int
    size: int
    link: int
    entry_size: int


class Decoder:
    """Decode RV32I program bytes into the simulator program representation."""

    def decode_lines(self, lines: list[str], base_address: int = 0) -> Program:
        """Decode textual 32-bit machine words into a program."""
        words: bytearray = bytearray()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            words.extend(int(stripped, 0).to_bytes(4, byteorder="little", signed=False))
        return self.decode_bytes(bytes(words), base_address=base_address)

    def decode_bytes(self, blob: bytes, base_address: int = 0, name: str = "anonymous") -> Program:
        """Decode either a raw RV32I binary blob or an ELF32 image."""
        if blob.startswith(ELF_MAGIC):
            return self.decode_elf(blob, name=name)
        return self._decode_raw_binary(blob, base_address=base_address, name=name)

    def decode_elf(self, blob: bytes, name: str = "anonymous") -> Program:
        """Decode an ELF32 little-endian RV32I program image."""
        entry_point, program_headers, section_headers = self._parse_elf_headers(blob)
        segments: list[ProgramSegment] = []
        instructions: dict[int, Instruction] = {}
        for program_header in program_headers:
            (
                segment_type,
                file_offset,
                virtual_address,
                _physical_address,
                file_size,
                memory_size,
                flags,
                _align,
            ) = program_header
            if segment_type != PT_LOAD:
                continue
            segment_bytes = bytearray(blob[file_offset : file_offset + file_size])
            if memory_size > file_size:
                segment_bytes.extend(b"\x00" * (memory_size - file_size))
            segment = ProgramSegment(
                address=virtual_address,
                data=bytes(segment_bytes),
                readable=bool(flags & PF_R),
                writable=bool(flags & PF_W),
                executable=bool(flags & PF_X),
            )
            segments.append(segment)
            if segment.executable:
                instructions.update(self._decode_instruction_stream(segment.address, segment.data))
        return Program(
            instructions=instructions,
            segments=tuple(segments),
            entry_point=entry_point,
            labels=self._extract_symbols(blob, section_headers),
            name=name,
        )

    def _decode_raw_binary(self, blob: bytes, base_address: int, name: str) -> Program:
        """Decode a flat binary containing only executable RV32I words."""
        instructions = self._decode_instruction_stream(base_address, blob)
        segment = ProgramSegment(
            address=base_address,
            data=blob,
            readable=True,
            writable=False,
            executable=True,
        )
        return Program(
            instructions=instructions,
            segments=(segment,),
            entry_point=base_address,
            name=name,
        )

    def _decode_instruction_stream(self, base_address: int, blob: bytes) -> dict[int, Instruction]:
        """Decode a contiguous executable instruction stream."""
        if len(blob) % 4 != 0:
            raise ValueError("RV32I executable segments must contain a whole number of 32-bit words.")
        instructions: dict[int, Instruction] = {}
        for offset in range(0, len(blob), 4):
            pc = base_address + offset
            word = int.from_bytes(blob[offset : offset + 4], byteorder="little", signed=False)
            instructions[pc] = self._decode_instruction(word, pc)
        return instructions

    def _parse_elf_headers(
        self,
        blob: bytes,
    ) -> tuple[int, list[tuple[int, int, int, int, int, int, int, int]], list[ElfSectionHeader]]:
        """Parse the ELF32 header and return loadable metadata."""
        if len(blob) < 52:
            raise ValueError("ELF image is too small to contain an ELF32 header.")
        if blob[:4] != ELF_MAGIC:
            raise ValueError("ELF magic does not match.")
        elf_class = blob[4]
        data_encoding = blob[5]
        if elf_class != 1 or data_encoding != 1:
            raise ValueError("Only little-endian ELF32 images are supported.")
        machine = struct.unpack_from("<H", blob, 18)[0]
        if machine != EM_RISCV:
            raise ValueError(f"ELF machine {machine} is not RISC-V.")
        entry_point = struct.unpack_from("<I", blob, 24)[0]
        program_header_offset = struct.unpack_from("<I", blob, 28)[0]
        section_header_offset = struct.unpack_from("<I", blob, 32)[0]
        program_header_entry_size = struct.unpack_from("<H", blob, 42)[0]
        program_header_count = struct.unpack_from("<H", blob, 44)[0]
        section_header_entry_size = struct.unpack_from("<H", blob, 46)[0]
        section_header_count = struct.unpack_from("<H", blob, 48)[0]
        program_headers: list[tuple[int, int, int, int, int, int, int, int]] = []
        for index in range(program_header_count):
            offset = program_header_offset + index * program_header_entry_size
            program_headers.append(struct.unpack_from("<IIIIIIII", blob, offset))
        section_headers: list[ElfSectionHeader] = []
        for index in range(section_header_count):
            offset = section_header_offset + index * section_header_entry_size
            (
                name_offset,
                section_type,
                _flags,
                _address,
                section_offset,
                size,
                link,
                _info,
                _address_align,
                entry_size,
            ) = struct.unpack_from("<IIIIIIIIII", blob, offset)
            section_headers.append(
                ElfSectionHeader(
                    name_offset=name_offset,
                    section_type=section_type,
                    offset=section_offset,
                    size=size,
                    link=link,
                    entry_size=entry_size,
                )
            )
        return entry_point, program_headers, section_headers

    def _extract_symbols(self, blob: bytes, section_headers: list[ElfSectionHeader]) -> dict[str, int]:
        """Extract ELF symbols when a symbol table is present."""
        symbols: dict[str, int] = {}
        for section_header in section_headers:
            if section_header.section_type != SHT_SYMTAB:
                continue
            if section_header.entry_size == 0:
                continue
            string_table_header = section_headers[section_header.link]
            string_table = blob[
                string_table_header.offset : string_table_header.offset + string_table_header.size
            ]
            entry_count = section_header.size // section_header.entry_size
            for entry_index in range(entry_count):
                offset = section_header.offset + entry_index * section_header.entry_size
                name_offset, value, _size, _info, _other, _shndx = struct.unpack_from(
                    "<IIIBBH",
                    blob,
                    offset,
                )
                if name_offset == 0:
                    continue
                end_index = string_table.find(b"\x00", name_offset)
                if end_index < 0:
                    continue
                name = string_table[name_offset:end_index].decode("utf-8")
                symbols[name] = value
        return symbols

    def _decode_instruction(self, word: int, pc: int) -> Instruction:
        """Decode one 32-bit RV32I word into a typed instruction record."""
        opcode_field = word & 0x7F
        rd = (word >> 7) & 0x1F
        funct3 = (word >> 12) & 0x7
        rs1 = (word >> 15) & 0x1F
        rs2 = (word >> 20) & 0x1F
        funct7 = (word >> 25) & 0x7F
        metadata = {
            "pc": pc,
            "word": word,
            "rd": rd,
            "rs1": rs1,
            "rs2": rs2,
            "funct3": funct3,
            "funct7": funct7,
        }
        if opcode_field == 0x37:
            imm = word & 0xFFFFF000
            return self._make_instruction("lui", metadata, rd=rd, imm=imm)
        if opcode_field == 0x17:
            imm = word & 0xFFFFF000
            return self._make_instruction("auipc", metadata, rd=rd, imm=imm)
        if opcode_field == 0x6F:
            imm = self._decode_j_type_imm(word)
            return self._make_instruction("jal", metadata, rd=rd, imm=imm, is_control=True)
        if opcode_field == 0x67:
            imm = self._sign_extend(word >> 20, 12)
            return self._make_instruction(
                "jalr",
                metadata,
                rd=rd,
                rs1=rs1,
                imm=imm,
                is_control=True,
            )
        if opcode_field == 0x63:
            branch_opcode = {
                0b000: "beq",
                0b001: "bne",
                0b100: "blt",
                0b101: "bge",
                0b110: "bltu",
                0b111: "bgeu",
            }.get(funct3)
            if branch_opcode is None:
                return self._illegal_instruction(
                    metadata,
                    f"Unsupported RV32I branch funct3=0b{funct3:03b} at 0x{pc:08x}.",
                )
            imm = self._decode_b_type_imm(word)
            return self._make_instruction(
                branch_opcode,
                metadata,
                rs1=rs1,
                rs2=rs2,
                imm=imm,
                is_control=True,
            )
        if opcode_field == 0x03:
            load_opcode = {
                0b000: "lb",
                0b001: "lh",
                0b010: "lw",
                0b100: "lbu",
                0b101: "lhu",
            }.get(funct3)
            if load_opcode is None:
                return self._illegal_instruction(
                    metadata,
                    f"Unsupported RV32I load funct3=0b{funct3:03b} at 0x{pc:08x}.",
                )
            imm = self._sign_extend(word >> 20, 12)
            return self._make_instruction(load_opcode, metadata, rd=rd, rs1=rs1, imm=imm)
        if opcode_field == 0x23:
            store_opcode = {
                0b000: "sb",
                0b001: "sh",
                0b010: "sw",
            }.get(funct3)
            if store_opcode is None:
                return self._illegal_instruction(
                    metadata,
                    f"Unsupported RV32I store funct3=0b{funct3:03b} at 0x{pc:08x}.",
                )
            imm = self._decode_s_type_imm(word)
            return self._make_instruction(store_opcode, metadata, rs1=rs1, rs2=rs2, imm=imm)
        if opcode_field == 0x13:
            imm = self._sign_extend(word >> 20, 12)
            if funct3 == 0b001:
                if funct7 != 0:
                    return self._illegal_instruction(metadata, f"Invalid SLLI encoding at 0x{pc:08x}.")
                return self._make_instruction("slli", metadata, rd=rd, rs1=rs1, imm=rs2)
            if funct3 == 0b101:
                shift_opcode = {0x00: "srli", 0x20: "srai"}.get(funct7)
                if shift_opcode is None:
                    return self._illegal_instruction(
                        metadata,
                        f"Invalid shift-immediate encoding at 0x{pc:08x}.",
                    )
                return self._make_instruction(shift_opcode, metadata, rd=rd, rs1=rs1, imm=rs2)
            alu_imm_opcode = {
                0b000: "addi",
                0b010: "slti",
                0b011: "sltiu",
                0b100: "xori",
                0b110: "ori",
                0b111: "andi",
            }.get(funct3)
            if alu_imm_opcode is None:
                return self._illegal_instruction(
                    metadata,
                    f"Unsupported RV32I ALU-immediate funct3=0b{funct3:03b} at 0x{pc:08x}."
                )
            return self._make_instruction(alu_imm_opcode, metadata, rd=rd, rs1=rs1, imm=imm)
        if opcode_field == 0x33:
            alu_opcode = {
                (0b000, 0x00): "add",
                (0b000, 0x20): "sub",
                (0b001, 0x00): "sll",
                (0b010, 0x00): "slt",
                (0b011, 0x00): "sltu",
                (0b100, 0x00): "xor",
                (0b101, 0x00): "srl",
                (0b101, 0x20): "sra",
                (0b110, 0x00): "or",
                (0b111, 0x00): "and",
            }.get((funct3, funct7))
            if alu_opcode is None:
                return self._illegal_instruction(
                    metadata,
                    f"Unsupported RV32I ALU register encoding funct3=0b{funct3:03b} funct7=0x{funct7:02x} at 0x{pc:08x}."
                )
            return self._make_instruction(alu_opcode, metadata, rd=rd, rs1=rs1, rs2=rs2)
        if opcode_field == 0x0F:
            if funct3 != 0:
                return self._illegal_instruction(metadata, f"Unsupported fence encoding at 0x{pc:08x}.")
            return self._make_instruction("fence", metadata)
        if opcode_field == 0x73:
            csr_address = word >> 20
            if funct3 == 0:
                system_opcode = {
                    0x000: "ecall",
                    0x001: "ebreak",
                    0x302: "mret",
                }.get(csr_address)
                if system_opcode is None:
                    return self._illegal_instruction(
                        metadata,
                        f"Unsupported system encoding 0x{csr_address:03x} at 0x{pc:08x}.",
                    )
                source_csrs = (0x341,) if system_opcode == "mret" else ()
                return self._make_instruction(
                    system_opcode,
                    metadata,
                    is_control=True,
                    source_csrs=source_csrs,
                )
            csr_opcode = {
                0b001: "csrrw",
                0b010: "csrrs",
                0b011: "csrrc",
                0b101: "csrrwi",
                0b110: "csrrsi",
                0b111: "csrrci",
            }.get(funct3)
            if csr_opcode is None:
                return self._illegal_instruction(
                    metadata,
                    f"Unsupported CSR funct3=0b{funct3:03b} at 0x{pc:08x}.",
                )
            source_regs: tuple[int, ...] = ()
            zimm = None
            writes_csr = True
            if funct3 >= 0b101:
                zimm = rs1
                writes_csr = csr_opcode in {"csrrwi"} or zimm != 0
            else:
                source_regs = (rs1,)
                writes_csr = csr_opcode == "csrrw" or rs1 != 0
            return self._make_instruction(
                csr_opcode,
                metadata,
                rd=rd,
                rs1=rs1,
                imm=zimm,
                source_regs=source_regs,
                source_csrs=(csr_address,),
                dest_csrs=((csr_address,) if writes_csr else ()),
            )
        return self._illegal_instruction(metadata, f"Unsupported RV32I opcode 0x{opcode_field:02x} at 0x{pc:08x}.")

    def _illegal_instruction(self, metadata: dict[str, int], reason: str) -> Instruction:
        """Return a decoded illegal instruction placeholder."""
        normalized = dict(metadata)
        normalized["illegal_reason"] = reason
        return Instruction(opcode="illegal", metadata=normalized)

    def _make_instruction(
        self,
        opcode: str,
        metadata: dict[str, int],
        rd: int | None = None,
        rs1: int | None = None,
        rs2: int | None = None,
        imm: int | None = None,
        is_control: bool = False,
        source_regs: tuple[int, ...] | None = None,
        source_csrs: tuple[int, ...] = (),
        dest_csrs: tuple[int, ...] = (),
    ) -> Instruction:
        """Construct an instruction with normalized metadata."""
        normalized = dict(metadata)
        normalized["source_regs"] = (
            tuple(int(register) for register in source_regs)
            if source_regs is not None
            else tuple(register for register in (rs1, rs2) if register is not None)
        )
        normalized["dest_regs"] = tuple(
            register for register in (rd,) if register is not None and register != 0
        )
        normalized["source_csrs"] = tuple(source_csrs)
        normalized["dest_csrs"] = tuple(dest_csrs)
        if rd is not None:
            normalized["rd"] = rd
        if rs1 is not None:
            normalized["rs1"] = rs1
        if rs2 is not None:
            normalized["rs2"] = rs2
        if imm is not None:
            normalized["imm"] = imm
        normalized["is_control"] = is_control
        return Instruction(opcode=opcode, metadata=normalized)

    def _decode_s_type_imm(self, word: int) -> int:
        """Decode an RV32I S-type immediate."""
        lower = (word >> 7) & 0x1F
        upper = (word >> 25) & 0x7F
        return self._sign_extend((upper << 5) | lower, 12)

    def _decode_b_type_imm(self, word: int) -> int:
        """Decode an RV32I B-type immediate."""
        bit12 = (word >> 31) & 0x1
        bit11 = (word >> 7) & 0x1
        bits10_5 = (word >> 25) & 0x3F
        bits4_1 = (word >> 8) & 0xF
        immediate = (bit12 << 12) | (bit11 << 11) | (bits10_5 << 5) | (bits4_1 << 1)
        return self._sign_extend(immediate, 13)

    def _decode_j_type_imm(self, word: int) -> int:
        """Decode an RV32I J-type immediate."""
        bit20 = (word >> 31) & 0x1
        bits19_12 = (word >> 12) & 0xFF
        bit11 = (word >> 20) & 0x1
        bits10_1 = (word >> 21) & 0x3FF
        immediate = (bit20 << 20) | (bits19_12 << 12) | (bit11 << 11) | (bits10_1 << 1)
        return self._sign_extend(immediate, 21)

    def _sign_extend(self, value: int, bits: int) -> int:
        """Sign-extend a field with the provided width."""
        sign_bit = 1 << (bits - 1)
        return (value ^ sign_bit) - sign_bit
