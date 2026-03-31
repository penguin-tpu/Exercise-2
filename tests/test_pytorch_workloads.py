"""Torch-to-simulator workload lowering tests."""

from __future__ import annotations

import struct

import torch

from perf_modeling.config import AcceleratorConfig, CoreConfig, DRAMConfig, ScalarUnitConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.pytorch import (
    build_linear_relu_sequential_from_torch,
    build_two_layer_mlp_from_torch_sequential,
)


def unpack_float32(blob: bytes) -> tuple[float, ...]:
    """Unpack little-endian float32 values from bytes."""
    return struct.unpack("<" + "f" * (len(blob) // 4), blob)


class TestPyTorchWorkloads:
    """Exercise simple Torch-model lowering into simulator workloads."""

    def make_config(self) -> AcceleratorConfig:
        """Construct a compact configuration for Torch-lowered workload tests."""
        return AcceleratorConfig(
            core=CoreConfig(
                scalar=ScalarUnitConfig(pipeline_depth=1, queue_depth=4),
            ),
            dram=DRAMConfig(
                capacity_bytes=1 << 20,
                read_latency_cycles=3,
                write_latency_cycles=3,
                bytes_per_cycle=16,
            ),
        )

    def build_reference_model(self) -> torch.nn.Sequential:
        """Construct a deterministic two-layer Torch MLP used by the lowering tests."""
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2, bias=True),
        )
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[2.0, 1.5], [-1.0, 0.25], [0.5, -2.0]], dtype=torch.float32))
            model[0].bias.copy_(torch.tensor([0.5, 2.0, -1.0], dtype=torch.float32))
            model[2].weight.copy_(torch.tensor([[1.0, 0.5, -1.5], [-1.0, 2.0, 0.25]], dtype=torch.float32))
            model[2].bias.copy_(torch.tensor([1.0, -0.5], dtype=torch.float32))
        return model

    def build_deeper_reference_model(self) -> torch.nn.Sequential:
        """Construct a deterministic three-layer Torch MLP used by the generic lowering tests."""
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1, bias=True),
        )
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[2.0, 1.5], [-1.0, 0.25], [0.5, -2.0]], dtype=torch.float32))
            model[0].bias.copy_(torch.tensor([0.5, 2.0, -1.0], dtype=torch.float32))
            model[2].weight.copy_(torch.tensor([[1.0, 0.5, -1.5], [-1.0, 2.0, 0.25]], dtype=torch.float32))
            model[2].bias.copy_(torch.tensor([1.0, -0.5], dtype=torch.float32))
            model[4].weight.copy_(torch.tensor([[0.25, -2.0]], dtype=torch.float32))
            model[4].bias.copy_(torch.tensor([0.75], dtype=torch.float32))
        return model

    def test_two_layer_mlp_torch_lowering_executes_end_to_end(self) -> None:
        """A tiny Torch MLP should lower into a runnable simulator program plus memory image."""
        model = self.build_reference_model()

        bundle = build_two_layer_mlp_from_torch_sequential(
            model=model,
            input_tensor=torch.tensor([[1.0, -2.0]], dtype=torch.float32),
        )

        engine = SimulatorEngine(config=self.make_config(), program=bundle.program)
        for address, payload in bundle.dram_image.items():
            engine.state.dram.write(address, payload)

        stats = engine.run(max_cycles=600).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(bundle.output_address, 8)) == bundle.expected_output
        assert bundle.expected_output == (-4.0, 1.375)
        assert stats["mxu.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 3

    def test_staged_two_layer_mlp_torch_lowering_executes_end_to_end(self) -> None:
        """The Torch lowering bridge should also target the staged DMA/scratchpad execution path."""
        config = self.make_config()
        bundle = build_two_layer_mlp_from_torch_sequential(
            model=self.build_reference_model(),
            input_tensor=torch.tensor([[1.0, -2.0]], dtype=torch.float32),
            staged=True,
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )

        engine = SimulatorEngine(config=config, program=bundle.program)
        for address, payload in bundle.dram_image.items():
            engine.state.dram.write(address, payload)

        stats = engine.run(max_cycles=900).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(bundle.output_address, 8)) == bundle.expected_output
        assert bundle.expected_output == (-4.0, 1.375)
        assert stats["dma.issued_ops"] == 6
        assert stats["mxu.issued_ops"] == 2
        assert stats["vector.issued_ops"] == 3

    def test_generic_torch_lowering_supports_deeper_linear_relu_sequential_models(self) -> None:
        """The generic Torch lowering helper should support deeper alternating Linear/ReLU sequentials."""
        bundle = build_linear_relu_sequential_from_torch(
            model=self.build_deeper_reference_model(),
            input_tensor=torch.tensor([[1.0, -2.0]], dtype=torch.float32),
        )

        engine = SimulatorEngine(config=self.make_config(), program=bundle.program)
        for address, payload in bundle.dram_image.items():
            engine.state.dram.write(address, payload)

        stats = engine.run(max_cycles=900).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(bundle.output_address, 4)) == bundle.expected_output
        assert bundle.expected_output == (-2.0,)
        assert stats["mxu.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 5

    def test_generic_torch_lowering_supports_staged_deeper_linear_relu_sequential_models(self) -> None:
        """The generic Torch lowering helper should also target staged execution for deeper sequentials."""
        config = self.make_config()
        bundle = build_linear_relu_sequential_from_torch(
            model=self.build_deeper_reference_model(),
            input_tensor=torch.tensor([[1.0, -2.0]], dtype=torch.float32),
            staged=True,
            scratchpad_base_address=config.machine.scratchpad_base_address,
        )

        engine = SimulatorEngine(config=config, program=bundle.program)
        for address, payload in bundle.dram_image.items():
            engine.state.dram.write(address, payload)

        stats = engine.run(max_cycles=1200).snapshot()

        assert engine.state.halted
        assert unpack_float32(engine.state.dram.read(bundle.output_address, 4)) == bundle.expected_output
        assert bundle.expected_output == (-2.0,)
        assert stats["dma.issued_ops"] == 8
        assert stats["mxu.issued_ops"] == 3
        assert stats["vector.issued_ops"] == 5
