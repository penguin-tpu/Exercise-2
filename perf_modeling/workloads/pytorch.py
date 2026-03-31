"""Helpers that lower small Torch models into simulator workloads."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

from perf_modeling.program import Program
from perf_modeling.workloads.builders import ProgramBuilder
from perf_modeling.workloads.kernels import KernelProblem


@dataclass(frozen=True)
class TorchWorkloadBundle:
    """One lowered simulator workload plus its required memory image."""

    program: Program
    """Program that executes the lowered workload"""

    dram_image: dict[int, bytes]
    """Memory payloads that must be present before execution"""

    output_address: int
    """DRAM address where the workload stores its final output tensor"""

    output_shape: tuple[int, ...]
    """Logical shape of the final output tensor"""

    expected_output: tuple[float, ...]
    """Flattened float32 output expected from Torch reference execution"""


def _serialize_float32_tensor(value: torch.Tensor) -> bytes:
    """Serialize one Torch tensor as little-endian float32 bytes."""
    flattened = value.detach().to(dtype=torch.float32).contiguous().reshape(-1).tolist()
    return struct.pack("<" + "f" * len(flattened), *flattened)


def _align_address(address: int, alignment: int = 0x20) -> int:
    """Round one address up to the requested alignment."""
    mask = alignment - 1
    return (address + mask) & ~mask


def build_two_layer_mlp_from_torch_sequential(
    model: torch.nn.Sequential,
    input_tensor: torch.Tensor,
    base_address: int = 0x1000,
    data_address: int = 0x400,
    acc_dtype: str = "float32",
    out_dtype: str = "float32",
    staged: bool = False,
    scratchpad_base_address: int | None = None,
) -> TorchWorkloadBundle:
    """Lower a simple Linear-ReLU-Linear Torch model into a simulator workload."""
    if len(model) != 3:
        raise ValueError("Torch lowering currently expects nn.Sequential(Linear, ReLU, Linear).")
    if not isinstance(model[0], torch.nn.Linear) or not isinstance(model[1], torch.nn.ReLU) or not isinstance(model[2], torch.nn.Linear):
        raise ValueError("Torch lowering currently expects nn.Sequential(Linear, ReLU, Linear).")
    if model[0].bias is None or model[2].bias is None:
        raise ValueError("Torch lowering currently expects both Linear layers to include bias.")
    if input_tensor.ndim != 2:
        raise ValueError("Torch lowering currently expects a rank-2 input tensor.")
    if staged and scratchpad_base_address is None:
        raise ValueError("Torch lowering requires a scratchpad base address in staged mode.")

    input_value = input_tensor.detach().to(dtype=torch.float32).contiguous()
    weight0_value = model[0].weight.detach().to(dtype=torch.float32).transpose(0, 1).contiguous()
    bias0_value = model[0].bias.detach().to(dtype=torch.float32).reshape(1, -1).contiguous()
    weight1_value = model[2].weight.detach().to(dtype=torch.float32).transpose(0, 1).contiguous()
    bias1_value = model[2].bias.detach().to(dtype=torch.float32).reshape(1, -1).contiguous()

    problem = KernelProblem(
        name="torch-two-layer-mlp",
        input_shapes=(
            tuple(int(dimension) for dimension in input_value.shape),
            tuple(int(dimension) for dimension in weight0_value.shape),
            tuple(int(dimension) for dimension in bias0_value.shape),
            tuple(int(dimension) for dimension in weight1_value.shape),
            tuple(int(dimension) for dimension in bias1_value.shape),
        ),
        output_shape=(input_value.shape[0], weight1_value.shape[1]),
    )

    input_blob = _serialize_float32_tensor(input_value)
    weight0_blob = _serialize_float32_tensor(weight0_value)
    bias0_blob = _serialize_float32_tensor(bias0_value)
    weight1_blob = _serialize_float32_tensor(weight1_value)
    bias1_blob = _serialize_float32_tensor(bias1_value)

    input_address = _align_address(data_address)
    weight0_address = _align_address(input_address + len(input_blob))
    bias0_address = _align_address(weight0_address + len(weight0_blob))
    weight1_address = _align_address(bias0_address + len(bias0_blob))
    bias1_address = _align_address(weight1_address + len(weight1_blob))
    output_address = _align_address(bias1_address + len(bias1_blob))

    if staged:
        program = ProgramBuilder(base_address=base_address).build_staged_two_layer_mlp_smoke_test(
            problem=problem,
            input_address=input_address,
            weight0_address=weight0_address,
            bias0_address=bias0_address,
            weight1_address=weight1_address,
            bias1_address=bias1_address,
            output_address=output_address,
            acc_dtype=acc_dtype,
            out_dtype=out_dtype,
            scratchpad_base_address=int(scratchpad_base_address),
        )
    else:
        program = ProgramBuilder(base_address=base_address).build_two_layer_mlp_smoke_test(
            problem=problem,
            input_address=input_address,
            weight0_address=weight0_address,
            bias0_address=bias0_address,
            weight1_address=weight1_address,
            bias1_address=bias1_address,
            output_address=output_address,
            acc_dtype=acc_dtype,
            out_dtype=out_dtype,
        )

    expected = model(input_value).detach().to(dtype=torch.float32).reshape(-1).tolist()
    return TorchWorkloadBundle(
        program=program,
        dram_image={
            input_address: input_blob,
            weight0_address: weight0_blob,
            bias0_address: bias0_blob,
            weight1_address: weight1_blob,
            bias1_address: bias1_blob,
        },
        output_address=output_address,
        output_shape=problem.output_shape,
        expected_output=tuple(float(value) for value in expected),
    )
