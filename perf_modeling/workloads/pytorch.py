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


def build_linear_relu_sequential_from_torch(
    model: torch.nn.Sequential,
    input_tensor: torch.Tensor,
    base_address: int = 0x1000,
    data_address: int = 0x400,
    acc_dtype: str = "float32",
    out_dtype: str = "float32",
    staged: bool = False,
    scratchpad_base_address: int | None = None,
) -> TorchWorkloadBundle:
    """Lower an alternating Linear/ReLU Torch sequential model into a simulator workload."""
    if len(model) == 0:
        raise ValueError("Torch lowering currently expects at least one Linear layer.")
    if input_tensor.ndim != 2:
        raise ValueError("Torch lowering currently expects a rank-2 input tensor.")
    if staged and scratchpad_base_address is None:
        raise ValueError("Torch lowering requires a scratchpad base address in staged mode.")

    linear_layers: list[torch.nn.Linear] = []
    relu_after_layer: list[bool] = []
    for index, module in enumerate(model):
        if index % 2 == 0:
            if not isinstance(module, torch.nn.Linear):
                raise ValueError("Torch lowering currently expects alternating Linear/ReLU modules.")
            if module.bias is None:
                raise ValueError("Torch lowering currently expects every Linear layer to include bias.")
            linear_layers.append(module)
            relu_after_layer.append(False)
            continue
        if not isinstance(module, torch.nn.ReLU):
            raise ValueError("Torch lowering currently expects alternating Linear/ReLU modules.")
        relu_after_layer[-1] = True
    if len(model) % 2 == 0:
        raise ValueError("Torch lowering currently expects the sequential model to end with a Linear layer.")

    input_value = input_tensor.detach().to(dtype=torch.float32).contiguous()
    layer_values: list[tuple[torch.Tensor, torch.Tensor]] = []
    input_shapes: list[tuple[int, ...]] = [tuple(int(dimension) for dimension in input_value.shape)]
    current_shape = tuple(int(dimension) for dimension in input_value.shape)
    for layer in linear_layers:
        weight_value = layer.weight.detach().to(dtype=torch.float32).transpose(0, 1).contiguous()
        bias_value = layer.bias.detach().to(dtype=torch.float32).reshape(1, -1).expand(current_shape[0], -1).contiguous()
        layer_values.append((weight_value, bias_value))
        input_shapes.append(tuple(int(dimension) for dimension in weight_value.shape))
        input_shapes.append(tuple(int(dimension) for dimension in bias_value.shape))
        current_shape = (current_shape[0], weight_value.shape[1])

    problem = KernelProblem(
        name="torch-linear-relu-sequential",
        input_shapes=tuple(input_shapes),
        output_shape=current_shape,
    )

    blobs: list[bytes] = [_serialize_float32_tensor(input_value)]
    for weight_value, bias_value in layer_values:
        blobs.append(_serialize_float32_tensor(weight_value))
        blobs.append(_serialize_float32_tensor(bias_value))

    addresses: list[int] = []
    next_address = _align_address(data_address)
    for blob in blobs:
        addresses.append(next_address)
        next_address = _align_address(next_address + len(blob))
    output_address = next_address

    expected_tensor = model(input_value).detach().to(dtype=torch.float32).contiguous()
    expected_blob = _serialize_float32_tensor(expected_tensor)

    builder = ProgramBuilder(base_address=base_address)
    if staged:
        scratchpad_addresses = []
        next_scratchpad_address = int(scratchpad_base_address)
        for blob in blobs:
            scratchpad_addresses.append(next_scratchpad_address)
            next_scratchpad_address = _align_address(next_scratchpad_address + len(blob))
        output_scratchpad_address = next_scratchpad_address
        for dram_address, scratchpad_address, blob in zip(addresses, scratchpad_addresses, blobs, strict=True):
            builder.emit_dma_copy(source_address=dram_address, dest_address=scratchpad_address, num_bytes=len(blob))
        builder.emit("fence")
        load_addresses = scratchpad_addresses
        store_address = output_scratchpad_address
    else:
        load_addresses = addresses
        store_address = output_address

    builder.emit_tensor_load(dest_tensor=0, address=load_addresses[0], shape=problem.input_shapes[0], dtype=out_dtype)
    current_tensor = 0
    for layer_index, (weight_value, bias_value) in enumerate(layer_values):
        base_tensor = 1 + layer_index * 5
        weight_tensor = base_tensor
        bias_tensor = base_tensor + 1
        matmul_tensor = base_tensor + 2
        add_tensor = base_tensor + 3
        relu_tensor = base_tensor + 4
        builder.emit_tensor_load(
            dest_tensor=weight_tensor,
            address=load_addresses[1 + layer_index * 2],
            shape=tuple(int(dimension) for dimension in weight_value.shape),
            dtype=out_dtype,
        )
        builder.emit_tensor_load(
            dest_tensor=bias_tensor,
            address=load_addresses[2 + layer_index * 2],
            shape=tuple(int(dimension) for dimension in bias_value.shape),
            dtype=out_dtype,
        )
        builder.emit_matmul(
            dest_tensor=matmul_tensor,
            lhs_tensor=current_tensor,
            rhs_tensor=weight_tensor,
            acc_dtype=acc_dtype,
            out_dtype=out_dtype,
        )
        builder.emit_vector_add(
            dest_tensor=add_tensor,
            lhs_tensor=matmul_tensor,
            rhs_tensor=bias_tensor,
            out_dtype=out_dtype,
        )
        current_tensor = add_tensor
        if relu_after_layer[layer_index]:
            builder.emit_vector_relu(dest_tensor=relu_tensor, source_tensor=add_tensor, out_dtype=out_dtype)
            current_tensor = relu_tensor

    builder.emit_tensor_store(source_tensor=current_tensor, address=store_address)
    builder.emit("fence")
    if staged:
        builder.emit_dma_copy(source_address=store_address, dest_address=output_address, num_bytes=len(expected_blob))
        builder.emit("fence")
    builder.emit("ebreak")
    program = builder.build(name="torch-linear-relu-sequential")

    expected = expected_tensor.reshape(-1).tolist()
    return TorchWorkloadBundle(
        program=program,
        dram_image={address: blob for address, blob in zip(addresses, blobs, strict=True)},
        output_address=output_address,
        output_shape=problem.output_shape,
        expected_output=tuple(float(value) for value in expected),
    )


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
    """Lower a simple two-layer Torch MLP into a simulator workload."""
    if len(model) != 3:
        raise ValueError("Two-layer Torch lowering expects nn.Sequential(Linear, ReLU, Linear).")
    return build_linear_relu_sequential_from_torch(
        model=model,
        input_tensor=input_tensor,
        base_address=base_address,
        data_address=data_address,
        acc_dtype=acc_dtype,
        out_dtype=out_dtype,
        staged=staged,
        scratchpad_base_address=scratchpad_base_address,
    )
