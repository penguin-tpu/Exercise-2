"""Shared type aliases and lightweight data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

Cycle = int
Address = int
OpId = int
DTypeName = str
LayoutName = str


class StorageLocation(str, Enum):
    """Logical storage locations visible to the simulator."""

    REGISTER = "register"
    SCRATCHPAD = "scratchpad"
    DRAM = "dram"
    HOST = "host"


@dataclass(slots=True)
class QuantizationParams:
    """Quantization metadata carried alongside tensor values."""

    scale: float = 1.0
    zero_point: int = 0
    axis: int | None = None


@dataclass(slots=True)
class TensorDescriptor:
    """Metadata describing a tensor value stored by the simulator."""

    shape: tuple[int, ...]
    dtype: DTypeName
    layout: LayoutName = "contiguous"
    location: StorageLocation = StorageLocation.REGISTER
    name: str = ""
    quant: QuantizationParams = field(default_factory=QuantizationParams)
