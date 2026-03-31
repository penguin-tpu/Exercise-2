"""Scratchpad memory models and bank accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.state.memory import ByteAddressableMemory


@dataclass(slots=True)
class ScratchpadAccess:
    """Description of one scratchpad access used by the banking model."""

    address: int
    size_bytes: int
    is_write: bool = False


@dataclass
class ScratchpadMemory(ByteAddressableMemory):
    """Banked scratchpad SRAM abstraction."""

    num_banks: int = 1
    bank_width_bytes: int = 1
    reserved_banks: set[int] = field(default_factory=set)

    def bank_index(self, address: int) -> int:
        """Return the bank index selected by an address."""
        return (address // max(1, self.bank_width_bytes)) % max(1, self.num_banks)

    def bank_indices_for_range(self, address: int, size_bytes: int) -> tuple[int, ...]:
        """Return the unique banks touched by one contiguous local address range."""
        if size_bytes <= 0:
            return ()
        first_bank = address // max(1, self.bank_width_bytes)
        last_bank = (address + size_bytes - 1) // max(1, self.bank_width_bytes)
        return tuple(
            sorted(
                {
                    bank_slot % max(1, self.num_banks)
                    for bank_slot in range(first_bank, last_bank + 1)
                }
            )
        )

    def reserve_bank(self, bank_index: int) -> None:
        """Mark a scratchpad bank as reserved for the current cycle."""
        self.reserved_banks.add(bank_index)

    def release_banks(self) -> None:
        """Release all bank reservations for the next cycle."""
        self.reserved_banks.clear()
