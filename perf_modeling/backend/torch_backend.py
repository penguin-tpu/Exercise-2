"""Torch-backed tensor semantics."""

from __future__ import annotations

from typing import Protocol

import torch

from perf_modeling.types import DTypeName, LayoutName


class TensorBackend(Protocol):
    """Protocol implemented by tensor backends used by the simulator."""

    def zeros(self, shape: tuple[int, ...], dtype: DTypeName, layout: LayoutName = "contiguous") -> object:
        """Allocate a zero-initialized tensor."""

    def cast(self, value: object, dtype: DTypeName) -> object:
        """Convert a tensor to a new dtype."""

    def matmul(self, lhs: object, rhs: object, acc_dtype: DTypeName, out_dtype: DTypeName) -> object:
        """Execute a matrix multiply and return the result tensor."""

    def elementwise(self, op_name: str, args: tuple[object, ...], out_dtype: DTypeName) -> object:
        """Execute an elementwise tensor operation."""

    def reduce(self, op_name: str, value: object, out_dtype: DTypeName) -> object:
        """Execute a reduction and return a rank-1 tensor result."""


class TorchTensorBackend:
    """Torch-based implementation of tensor allocation and math semantics."""

    def __init__(self, device: str = "cpu") -> None:
        """Construct a Torch-backed tensor backend."""
        self.device = device

    def zeros(self, shape: tuple[int, ...], dtype: DTypeName, layout: LayoutName = "contiguous") -> object:
        """Allocate a zero-initialized tensor on the configured device."""
        _ = layout
        return torch.zeros(shape, dtype=self._resolve_dtype(dtype), device=self.device)

    def cast(self, value: object, dtype: DTypeName) -> object:
        """Convert a tensor to the requested Torch dtype."""
        return value.to(dtype=self._resolve_dtype(dtype))

    def matmul(self, lhs: object, rhs: object, acc_dtype: DTypeName, out_dtype: DTypeName) -> object:
        """Execute a matrix multiply using Torch tensor kernels."""
        lhs_cast = lhs.to(dtype=self._resolve_dtype(acc_dtype))
        rhs_cast = rhs.to(dtype=self._resolve_dtype(acc_dtype))
        result = torch.matmul(lhs_cast, rhs_cast)
        return result.to(dtype=self._resolve_dtype(out_dtype))

    def elementwise(self, op_name: str, args: tuple[object, ...], out_dtype: DTypeName) -> object:
        """Execute a supported elementwise operation using Torch."""
        if op_name == "add":
            result = args[0] + args[1]
        elif op_name == "sub":
            result = args[0] - args[1]
        elif op_name == "mul":
            result = args[0] * args[1]
        elif op_name == "max":
            result = torch.maximum(args[0], args[1])
        else:
            raise NotImplementedError(f"Unsupported Torch backend op: {op_name}")
        return result.to(dtype=self._resolve_dtype(out_dtype))

    def reduce(self, op_name: str, value: object, out_dtype: DTypeName) -> object:
        """Execute a supported reduction using Torch."""
        if op_name == "sum":
            result = value.reshape(-1).sum()
        else:
            raise NotImplementedError(f"Unsupported Torch backend reduction: {op_name}")
        return result.reshape(1).to(dtype=self._resolve_dtype(out_dtype))

    def _resolve_dtype(self, dtype: DTypeName) -> object:
        """Map a textual dtype name to a Torch dtype object."""
        mapping = {
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise KeyError(f"Unsupported Torch dtype mapping: {dtype}")
        return mapping[dtype]
