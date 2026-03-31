"""Helpers for custom or packed quantized tensor representations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PackedTensorEncoding:
    """Description of a packed tensor storage format."""

    bits_per_element: int
    signed: bool = True
    little_endian: bool = True

    def pack(self, values: object) -> bytes:
        """Pack an unpacked tensor representation into bytes."""
        raise NotImplementedError("Packed quantized tensor support is not implemented yet.")

    def unpack(self, payload: bytes) -> object:
        """Unpack bytes into a backend-friendly tensor representation."""
        raise NotImplementedError("Packed quantized tensor support is not implemented yet.")
