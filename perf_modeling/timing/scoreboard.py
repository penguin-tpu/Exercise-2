"""Scoreboard for architectural and structural hazards."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scoreboard:
    """Track busy registers and reserved resources."""

    busy_scalars: set[int] = field(default_factory=set)
    busy_csrs: set[int] = field(default_factory=set)
    busy_tensors: set[int] = field(default_factory=set)
    busy_resources: dict[str, int] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all tracked hazards."""
        self.busy_scalars.clear()
        self.busy_csrs.clear()
        self.busy_tensors.clear()
        self.busy_resources.clear()

    def mark_scalar_busy(self, index: int) -> None:
        """Mark a scalar destination register as busy."""
        if index == 0:
            return
        self.busy_scalars.add(index)

    def mark_tensor_busy(self, index: int) -> None:
        """Mark a tensor destination register as busy."""
        self.busy_tensors.add(index)

    def mark_csr_busy(self, address: int) -> None:
        """Mark one architectural CSR as busy until its writer retires."""
        self.busy_csrs.add(address)

    def release_scalar(self, index: int) -> None:
        """Release a scalar register once its producer completes."""
        self.busy_scalars.discard(index)

    def scalar_ready(self, index: int) -> bool:
        """Return whether the scalar register may be consumed this cycle."""
        return index == 0 or index not in self.busy_scalars

    def release_csr(self, address: int) -> None:
        """Release one CSR after its writer retires."""
        self.busy_csrs.discard(address)

    def csr_ready(self, address: int) -> bool:
        """Return whether the CSR may be accessed this cycle."""
        return address not in self.busy_csrs

    def release_tensor(self, index: int) -> None:
        """Release a tensor register once its producer completes."""
        self.busy_tensors.discard(index)

    def reserve_resource(self, name: str, until_cycle: int) -> None:
        """Reserve a named shared resource until the given cycle."""
        self.busy_resources[name] = until_cycle

    def resource_ready(self, name: str, cycle: int) -> bool:
        """Return whether a named resource is available at the cycle."""
        return self.busy_resources.get(name, -1) <= cycle
