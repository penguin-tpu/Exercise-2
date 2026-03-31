"""Torch-to-simulator workload lowering tests."""

from __future__ import annotations

import struct

import torch

from perf_modeling.config import AcceleratorConfig, CoreConfig, DRAMConfig, ScalarUnitConfig
from perf_modeling.engine import SimulatorEngine
from perf_modeling.workloads.pytorch import build_two_layer_mlp_from_torch_sequential


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

    def test_two_layer_mlp_torch_lowering_executes_end_to_end(self) -> None:
        """A tiny Torch MLP should lower into a runnable simulator program plus memory image."""
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
