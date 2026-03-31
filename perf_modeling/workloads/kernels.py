"""Kernel and tile descriptors used by program builders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileShape:
    """Logical shape of one tensor tile."""

    dims: tuple[int, ...]


@dataclass(frozen=True)
class KernelProblem:
    """High-level kernel description consumed by workload builders."""

    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
