"""Scratchpad banking helpers."""

from __future__ import annotations

from dataclasses import dataclass

from perf_modeling.state.scratchpad import ScratchpadAccess


@dataclass
class BankingModel:
    """Compute bank selections and conflict penalties for scratchpad access."""

    num_banks: int
    bank_width_bytes: int
    conflict_penalty_cycles: int = 1

    def bank_index(self, address: int) -> int:
        """Return the bank index selected by the address."""
        return (address // max(1, self.bank_width_bytes)) % max(1, self.num_banks)

    def estimate_penalty(self, accesses: list[ScratchpadAccess]) -> int:
        """Estimate extra cycles caused by bank conflicts in one cycle."""
        seen: set[int] = set()
        penalty = 0
        for access in accesses:
            bank = self.bank_index(access.address)
            if bank in seen:
                penalty += self.conflict_penalty_cycles
            seen.add(bank)
        return penalty
