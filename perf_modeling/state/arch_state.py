"""Top-level architectural state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from perf_modeling.config import AcceleratorConfig
from perf_modeling.state.csr_file import (
    CSR_MEPC,
    CSR_MCAUSE,
    CSR_MTVEC,
    CSR_MTVAL,
    CSRFile,
)
from perf_modeling.state.memory import ByteAddressableMemory
from perf_modeling.state.register_file import ScalarRegisterFile
from perf_modeling.state.scratchpad import ScratchpadMemory
from perf_modeling.traps import MachineTrap
from perf_modeling.types import OpId, TensorDescriptor

if TYPE_CHECKING:
    from perf_modeling.program import Program


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
    csr_file: CSRFile
    scratchpad: ScratchpadMemory
    dram: ByteAddressableMemory
    machine_config: object
    pc: int = 0
    halted: bool = False
    fetch_stalled: bool = False
    exit_code: int | None = None
    trap_reason: str | None = None
    retired_instructions: int = 0
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
        csr_file = CSRFile()
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
            csr_file=csr_file,
            scratchpad=scratchpad,
            dram=dram,
            machine_config=config.machine,
            pc=config.machine.reset_pc,
        )

    def reset(self) -> None:
        """Reset all architected state to the initial machine state."""
        self.pc = self.machine_config.reset_pc
        self.halted = False
        self.fetch_stalled = False
        self.exit_code = None
        self.trap_reason = None
        self.retired_instructions = 0
        self.outstanding_ops.clear()
        self.scalar_regs.reset()
        self.scalar_regs.write(2, self.machine_config.initial_stack_pointer)
        self.tensor_regs.reset()
        self.csr_file.reset(self.machine_config)
        self.scratchpad.reset()
        self.dram.reset()

    def initialize_pc(self, value: int) -> None:
        """Initialize or update the current fetch PC."""
        self.pc = value & 0xFFFF_FFFF

    def next_pc(self, stride_bytes: int) -> None:
        """Advance the program counter to the next sequential instruction."""
        self.pc = (self.pc + stride_bytes) & 0xFFFF_FFFF

    def jump(self, target: int) -> None:
        """Update the program counter to a specific control-flow target."""
        self.pc = target & 0xFFFF_FFFF

    def load_program(self, program: Program, config: AcceleratorConfig) -> None:
        """Load a program image into architectural state and initialize machine registers."""
        self.reset()
        for segment in program.segments:
            if segment.address < config.machine.scratchpad_base_address:
                self.dram.load_image(segment.address, segment.data)
                continue
            scratchpad_address = segment.address - config.machine.scratchpad_base_address
            self.scratchpad.load_image(scratchpad_address, segment.data)
        self.initialize_pc(program.entry_point)
        stack_pointer = program.initial_stack_pointer
        if stack_pointer is None:
            stack_pointer = min(
                self.machine_config.initial_stack_pointer,
                config.dram.capacity_bytes - 4,
            )
        self.scalar_regs.write(2, stack_pointer)

    def halt(self, exit_code: int | None = None) -> None:
        """Stop architectural execution with an optional exit code."""
        self.halted = True
        self.fetch_stalled = False
        self.exit_code = exit_code

    def trap(self, reason: str) -> None:
        """Stop architectural execution because of a fatal machine trap."""
        self.halted = True
        self.fetch_stalled = False
        self.trap_reason = reason

    def enter_trap(self, trap: MachineTrap, cycle: int) -> int:
        """Update machine trap CSRs and return the trap handler target address."""
        self.csr_file.write(CSR_MEPC, trap.pc, trap.pc)
        self.csr_file.write(CSR_MCAUSE, trap.cause, trap.pc)
        self.csr_file.write(CSR_MTVAL, trap.tval, trap.pc)
        self.trap_reason = trap.reason
        self.halted = False
        self.fetch_stalled = False
        target = self.csr_file.read(CSR_MTVEC, cycle, self.retired_instructions)
        self.jump(target)
        return target

    def read_csr(self, address: int, cycle: int) -> int:
        """Read one machine CSR."""
        try:
            return self.csr_file.read(address, cycle, self.retired_instructions)
        except MachineTrap as trap:
            if trap.pc == 0:
                raise MachineTrap(
                    cause=trap.cause,
                    pc=self.pc,
                    tval=trap.tval,
                    reason=trap.reason,
                ) from trap
            raise

    def write_csr(self, address: int, value: int, pc: int) -> None:
        """Write one machine CSR."""
        self.csr_file.write(address, value, pc)

    def retire_instruction(self) -> None:
        """Advance the retired-instruction counter used by `instret` CSRs."""
        self.retired_instructions += 1

    def resolve_memory(self, address: int, config: AcceleratorConfig) -> tuple[ByteAddressableMemory, int]:
        """Return the addressed memory object plus its local byte address."""
        if config.machine.scratchpad_base_address <= address < (
            config.machine.scratchpad_base_address + config.scratchpad.capacity_bytes
        ):
            return self.scratchpad, address - config.machine.scratchpad_base_address
        return self.dram, address

    def read_memory(self, address: int, size: int, config: AcceleratorConfig) -> bytes:
        """Read a byte range from the mapped memory hierarchy."""
        memory, local_address = self.resolve_memory(address, config)
        return memory.read(local_address, size)

    def write_memory(self, address: int, data: bytes, config: AcceleratorConfig) -> None:
        """Write a byte range into the mapped memory hierarchy."""
        memory, local_address = self.resolve_memory(address, config)
        memory.write(local_address, data)

    def mark_op_outstanding(self, op_id: OpId) -> None:
        """Track an in-flight operation until its completion event fires."""
        self.outstanding_ops.add(op_id)

    def complete_op(self, op_id: OpId) -> None:
        """Clear an operation from the outstanding set after completion."""
        self.outstanding_ops.discard(op_id)
