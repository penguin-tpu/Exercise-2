"""Byte-addressable memory models."""

from __future__ import annotations

from dataclasses import dataclass, field


PAGE_SIZE_BYTES = 4096


@dataclass
class ByteAddressableMemory:
    """Flat byte-addressable storage used for DRAM and other memories."""

    capacity_bytes: int
    name: str = "memory"
    page_size_bytes: int = PAGE_SIZE_BYTES
    storage: dict[int, bytearray] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate the sparse backing store."""
        self.storage = {}

    def reset(self) -> None:
        """Clear all memory contents."""
        self.storage.clear()

    def _check_range(self, address: int, size: int) -> None:
        """Validate that a byte range lies within the allocated storage."""
        if address < 0 or size < 0 or address + size > self.capacity_bytes:
            raise IndexError(
                f"{self.name} access out of range: address=0x{address:08x} size={size}"
            )

    def read(self, address: int, size: int) -> bytes:
        """Read a byte range from memory."""
        self._check_range(address, size)
        result = bytearray()
        current_address = address
        remaining = size
        while remaining > 0:
            page_index, page_offset = divmod(current_address, self.page_size_bytes)
            chunk_size = min(remaining, self.page_size_bytes - page_offset)
            page = self.storage.get(page_index)
            if page is None:
                result.extend(b"\x00" * chunk_size)
            else:
                result.extend(page[page_offset : page_offset + chunk_size])
            current_address += chunk_size
            remaining -= chunk_size
        return bytes(result)

    def write(self, address: int, data: bytes) -> None:
        """Write a byte range into memory."""
        self._check_range(address, len(data))
        current_address = address
        source_offset = 0
        remaining = len(data)
        while remaining > 0:
            page_index, page_offset = divmod(current_address, self.page_size_bytes)
            chunk_size = min(remaining, self.page_size_bytes - page_offset)
            page = self.storage.setdefault(page_index, bytearray(self.page_size_bytes))
            page[page_offset : page_offset + chunk_size] = data[source_offset : source_offset + chunk_size]
            current_address += chunk_size
            source_offset += chunk_size
            remaining -= chunk_size

    def load_image(self, address: int, data: bytes) -> None:
        """Load a program or data image into memory at the target address."""
        self.write(address, data)

    def read_u8(self, address: int) -> int:
        """Read one unsigned byte from memory."""
        return self.read(address, 1)[0]

    def read_u16(self, address: int) -> int:
        """Read one unsigned 16-bit little-endian value from memory."""
        return int.from_bytes(self.read(address, 2), byteorder="little", signed=False)

    def read_u32(self, address: int) -> int:
        """Read one unsigned 32-bit little-endian value from memory."""
        return int.from_bytes(self.read(address, 4), byteorder="little", signed=False)

    def write_u8(self, address: int, value: int) -> None:
        """Write one unsigned byte into memory."""
        self.write(address, bytes((value & 0xFF,)))

    def write_u16(self, address: int, value: int) -> None:
        """Write one unsigned 16-bit little-endian value into memory."""
        self.write(address, int(value & 0xFFFF).to_bytes(2, byteorder="little", signed=False))

    def write_u32(self, address: int, value: int) -> None:
        """Write one unsigned 32-bit little-endian value into memory."""
        self.write(address, int(value & 0xFFFF_FFFF).to_bytes(4, byteorder="little", signed=False))
