"""Top-level architectural state."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.config import AcceleratorConfig
from perf_modeling.state.memory import ByteAddressableMemory
from perf_modeling.state.register_file import ScalarRegisterFile
from perf_modeling.state.scratchpad import ScratchpadMemory
from perf_modeling.types import OpId, TensorDescriptor


@dataclass
class TensorValue:
    """Tensor value and its architectural metadata."""

    descriptor: TensorDescriptor
    payload: object


@dataclass
class ArchState:
    """Mutable architected machine state for one simulator instance."""

    scalar_regs: ScalarRegisterFile
    tensor_regs: object
    scratchpad: ScratchpadMemory
    dram: ByteAddressableMemory
    pc: int = 0
    halted: bool = False
    outstanding_ops: set[OpId] = field(default_factory=set)

    @classmethod
    def from_config(cls, config: AcceleratorConfig) -> "ArchState":
        """Construct a fresh architectural state from accelerator config."""
        from perf_modeling.state.tensor_file import TensorRegisterFile

        scalar_regs = ScalarRegisterFile(
            num_registers=config.registers.num_scalar_registers,
            register_width_bits=config.registers.scalar_register_width_bits,
        )
        tensor_regs = TensorRegisterFile(num_registers=config.tensors.num_tensor_registers)
        scratchpad = ScratchpadMemory(
            capacity_bytes=config.scratchpad.capacity_bytes,
            name="scratchpad",
            num_banks=config.scratchpad.num_banks,
            bank_width_bytes=config.scratchpad.bank_width_bytes,
        )
        dram = ByteAddressableMemory(capacity_bytes=config.dram.capacity_bytes, name="dram")
        return cls(
            scalar_regs=scalar_regs,
            tensor_regs=tensor_regs,
            scratchpad=scratchpad,
            dram=dram,
        )

    def reset(self) -> None:
        """Reset all architected state to the initial machine state."""
        self.pc = 0
        self.halted = False
        self.outstanding_ops.clear()
        self.scalar_regs.reset()
        self.tensor_regs.reset()
        self.scratchpad.reset()
        self.dram.reset()

    def next_pc(self) -> None:
        """Advance the program counter to the next sequential instruction."""
        self.pc += 1

    def mark_op_outstanding(self, op_id: OpId) -> None:
        """Track an in-flight operation until its completion event fires."""
        self.outstanding_ops.add(op_id)

    def complete_op(self, op_id: OpId) -> None:
        """Clear an operation from the outstanding set after completion."""
        self.outstanding_ops.discard(op_id)
