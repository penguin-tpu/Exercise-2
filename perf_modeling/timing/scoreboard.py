"""Scoreboard for architectural and structural hazards."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.timing.resources import ResourceReservation


@dataclass
class Scoreboard:
    """Track busy registers and reserved resources."""

    busy_scalars: set[int] = field(default_factory=set)
    busy_csrs: set[int] = field(default_factory=set)
    busy_tensors: set[int] = field(default_factory=set)
    busy_resources: dict[str, list[ResourceReservation]] = field(default_factory=dict)

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
        """Release one tensor register after its producer retires."""
        self.busy_tensors.discard(index)

    def tensor_ready(self, index: int) -> bool:
        """Return whether the tensor register may be accessed this cycle."""
        return index not in self.busy_tensors

    def reserve_resource(self, name: str, start_cycle: int, end_cycle: int) -> None:
        """Reserve a named shared resource over one cycle interval."""
        reservations = self.busy_resources.setdefault(name, [])
        reservations.append(
            ResourceReservation(
                resource_name=name,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
            )
        )
        reservations.sort(key=lambda reservation: (reservation.start_cycle, reservation.end_cycle))

    def resource_ready(self, name: str, start_cycle: int, end_cycle: int) -> bool:
        """Return whether a named resource is available over one cycle interval."""
        reservations = self.busy_resources.get(name, [])
        retained_reservations: list[ResourceReservation] = []
        is_ready = True
        for reservation in reservations:
            if reservation.end_cycle <= start_cycle:
                continue
            retained_reservations.append(reservation)
            if reservation.overlaps_interval(start_cycle, end_cycle):
                is_ready = False
        if retained_reservations:
            self.busy_resources[name] = retained_reservations
        else:
            self.busy_resources.pop(name, None)
        return is_ready
