"""Machine-mode CSR storage and access helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.traps import CAUSE_ILLEGAL_INSTRUCTION, MachineTrap

CSR_MSTATUS = 0x300
CSR_MISA = 0x301
CSR_MTVEC = 0x305
CSR_MSCRATCH = 0x340
CSR_MEPC = 0x341
CSR_MCAUSE = 0x342
CSR_MTVAL = 0x343
CSR_MHARTID = 0xF14
CSR_CYCLE = 0xC00
CSR_INSTRET = 0xC02
CSR_MCYCLE = 0xB00
CSR_MINSTRET = 0xB02
CSR_CYCLEH = 0xC80
CSR_INSTRETH = 0xC82
CSR_MCYCLEH = 0xB80
CSR_MINSTRETH = 0xB82

MISA_RV32I = 0x4000_0100

READ_ONLY_CSRS = {
    CSR_MISA,
    CSR_MHARTID,
    CSR_CYCLE,
    CSR_INSTRET,
    CSR_MCYCLE,
    CSR_MINSTRET,
    CSR_CYCLEH,
    CSR_INSTRETH,
    CSR_MCYCLEH,
    CSR_MINSTRETH,
}


@dataclass
class CSRFile:
    """Architectural CSR state for a machine-mode-only RV32 core."""

    values: dict[int, int] = field(default_factory=dict)

    def reset(self, machine_config: object) -> None:
        """Reset all modeled CSRs to their configured boot values."""
        self.values = {
            CSR_MSTATUS: 0,
            CSR_MISA: MISA_RV32I,
            CSR_MTVEC: int(machine_config.default_mtvec),
            CSR_MSCRATCH: 0,
            CSR_MEPC: 0,
            CSR_MCAUSE: 0,
            CSR_MTVAL: 0,
            CSR_MHARTID: int(machine_config.hart_id),
        }

    def read(self, address: int, cycle: int, retired_instructions: int) -> int:
        """Read one architectural CSR."""
        if address in {CSR_CYCLE, CSR_MCYCLE}:
            return cycle & 0xFFFF_FFFF
        if address in {CSR_INSTRET, CSR_MINSTRET}:
            return retired_instructions & 0xFFFF_FFFF
        if address in {CSR_CYCLEH, CSR_MCYCLEH}:
            return (cycle >> 32) & 0xFFFF_FFFF
        if address in {CSR_INSTRETH, CSR_MINSTRETH}:
            return (retired_instructions >> 32) & 0xFFFF_FFFF
        if address not in self.values:
            raise MachineTrap(
                cause=CAUSE_ILLEGAL_INSTRUCTION,
                pc=0,
                tval=address,
                reason=f"Unsupported CSR read 0x{address:03x}.",
            )
        return self.values[address] & 0xFFFF_FFFF

    def write(self, address: int, value: int, pc: int) -> None:
        """Write one architectural CSR when it is modeled as writable."""
        if address in READ_ONLY_CSRS:
            raise MachineTrap(
                cause=CAUSE_ILLEGAL_INSTRUCTION,
                pc=pc,
                tval=address,
                reason=f"Write to read-only CSR 0x{address:03x}.",
            )
        if address not in self.values:
            raise MachineTrap(
                cause=CAUSE_ILLEGAL_INSTRUCTION,
                pc=pc,
                tval=address,
                reason=f"Unsupported CSR write 0x{address:03x}.",
            )
        self.values[address] = value & 0xFFFF_FFFF
