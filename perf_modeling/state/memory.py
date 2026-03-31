"""Byte-addressable memory models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ByteAddressableMemory:
    """Flat byte-addressable storage used for DRAM and other memories."""

    capacity_bytes: int
    name: str = "memory"
    storage: bytearray = field(init=False)

    def __post_init__(self) -> None:
        """Allocate the raw byte storage."""
        self.storage = bytearray(self.capacity_bytes)

    def reset(self) -> None:
        """Clear all memory contents."""
        self.storage[:] = b"\x00" * self.capacity_bytes

    def read(self, address: int, size: int) -> bytes:
        """Read a byte range from memory."""
        return bytes(self.storage[address : address + size])

    def write(self, address: int, data: bytes) -> None:
        """Write a byte range into memory."""
        self.storage[address : address + len(data)] = data
