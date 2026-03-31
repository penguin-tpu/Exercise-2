"""Instruction semantics registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from perf_modeling.isa.instruction import ExecutionPlan, Instruction
from perf_modeling.timing.latency import (
    dram_read_latency,
    dram_write_latency,
    scalar_latency,
    scratchpad_latency,
)
from perf_modeling.timing.resources import ResourceReservation

if TYPE_CHECKING:
    from perf_modeling.backend.torch_backend import TensorBackend
    from perf_modeling.config import AcceleratorConfig
    from perf_modeling.state.arch_state import ArchState
    from perf_modeling.timing.scoreboard import Scoreboard

InstructionPlanner = Callable[
    [Instruction, int, "ArchState", "AcceleratorConfig", "Scoreboard", "TensorBackend | None"],
    ExecutionPlan,
]

LOAD_WIDTHS = {
    "lb": (1, True),
    "lh": (2, True),
    "lw": (4, True),
    "lbu": (1, False),
    "lhu": (2, False),
}

STORE_WIDTHS = {
    "sb": 1,
    "sh": 2,
    "sw": 4,
}


class SemanticsRegistry:
    """Registry mapping opcodes to planner callbacks."""

    def __init__(self) -> None:
        self._planners: dict[str, InstructionPlanner] = {}

    def register(self, opcode: str, planner: InstructionPlanner) -> None:
        """Register a planner callback for an opcode."""
        self._planners[opcode] = planner

    def lookup(self, opcode: str) -> InstructionPlanner:
        """Return the planner callback for an opcode."""
        return self._planners[opcode]

    def plan(
        self,
        instruction: Instruction,
        cycle: int,
        state: "ArchState",
        config: "AcceleratorConfig",
        scoreboard: "Scoreboard",
        backend: "TensorBackend | None" = None,
    ) -> ExecutionPlan:
        """Plan execution for one instruction using the registered callback."""
        planner = self.lookup(instruction.opcode)
        return planner(instruction, cycle, state, config, scoreboard, backend)


def _u32(value: int) -> int:
    """Mask a value to the architectural register width."""
    return value & 0xFFFF_FFFF


def _s32(value: int) -> int:
    """Interpret an architectural value as a signed 32-bit integer."""
    value = _u32(value)
    if value & 0x8000_0000:
        return value - 0x1_0000_0000
    return value


def _load_field(instruction: Instruction, name: str, default: int = 0) -> int:
    """Read one decoded metadata field from an instruction."""
    return int(instruction.metadata.get(name, default))


def _next_pc(pc: int) -> int:
    """Return the next sequential RV32I fetch address."""
    return _u32(pc + 4)


def _check_alignment(address: int, size: int, config: "AcceleratorConfig", kind: str) -> None:
    """Validate an aligned architectural address when strict mode is enabled."""
    if not config.machine.strict_alignment:
        return
    if address % size != 0:
        raise ValueError(f"Misaligned {kind} access at 0x{address:08x} for size {size}.")


def _scalar_plan(
    cycle: int,
    config: "AcceleratorConfig",
    unit_name: str,
    callback: Callable[[], None],
    description: str,
) -> ExecutionPlan:
    """Build a scalar-unit execution plan."""
    latency = scalar_latency(config.core.scalar)
    completion_cycle = cycle + latency
    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name=unit_name,
                start_cycle=cycle,
                end_cycle=completion_cycle,
            )
        ],
        on_complete=callback,
        description=description,
    )


def _plan_immediate_alu(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    rs1 = _load_field(instruction, "rs1")
    imm = _load_field(instruction, "imm")
    lhs = state.scalar_regs.read(rs1)
    if instruction.opcode == "addi":
        result = _u32(lhs + imm)
    elif instruction.opcode == "slti":
        result = 1 if _s32(lhs) < imm else 0
    elif instruction.opcode == "sltiu":
        result = 1 if _u32(lhs) < _u32(imm) else 0
    elif instruction.opcode == "xori":
        result = _u32(lhs ^ imm)
    elif instruction.opcode == "ori":
        result = _u32(lhs | imm)
    elif instruction.opcode == "andi":
        result = _u32(lhs & imm)
    elif instruction.opcode == "slli":
        result = _u32(lhs << (imm & 0x1F))
    elif instruction.opcode == "srli":
        result = _u32(lhs >> (imm & 0x1F))
    elif instruction.opcode == "srai":
        result = _u32(_s32(lhs) >> (imm & 0x1F))
    else:
        raise KeyError(f"Unsupported ALU-immediate opcode {instruction.opcode!r}.")

    def on_complete() -> None:
        state.scalar_regs.write(rd, result)

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_register_alu(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    rs1 = _load_field(instruction, "rs1")
    rs2 = _load_field(instruction, "rs2")
    lhs = state.scalar_regs.read(rs1)
    rhs = state.scalar_regs.read(rs2)
    if instruction.opcode == "add":
        result = _u32(lhs + rhs)
    elif instruction.opcode == "sub":
        result = _u32(lhs - rhs)
    elif instruction.opcode == "sll":
        result = _u32(lhs << (rhs & 0x1F))
    elif instruction.opcode == "slt":
        result = 1 if _s32(lhs) < _s32(rhs) else 0
    elif instruction.opcode == "sltu":
        result = 1 if _u32(lhs) < _u32(rhs) else 0
    elif instruction.opcode == "xor":
        result = _u32(lhs ^ rhs)
    elif instruction.opcode == "srl":
        result = _u32(lhs >> (rhs & 0x1F))
    elif instruction.opcode == "sra":
        result = _u32(_s32(lhs) >> (rhs & 0x1F))
    elif instruction.opcode == "or":
        result = _u32(lhs | rhs)
    elif instruction.opcode == "and":
        result = _u32(lhs & rhs)
    else:
        raise KeyError(f"Unsupported ALU register opcode {instruction.opcode!r}.")

    def on_complete() -> None:
        state.scalar_regs.write(rd, result)

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_u_type(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    imm = _load_field(instruction, "imm")
    if instruction.opcode == "lui":
        result = _u32(imm)
    elif instruction.opcode == "auipc":
        result = _u32(pc + imm)
    else:
        raise KeyError(f"Unsupported U-type opcode {instruction.opcode!r}.")

    def on_complete() -> None:
        state.scalar_regs.write(rd, result)

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_jal(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    target = _u32(pc + _load_field(instruction, "imm"))
    link_value = _next_pc(pc)
    _check_alignment(target, 4, config, "jump")

    def on_complete() -> None:
        state.scalar_regs.write(rd, link_value)
        state.jump(target)
        state.fetch_stalled = False

    return _scalar_plan(cycle, config, "scalar", on_complete, f"jal @ 0x{pc:08x}")


def _plan_jalr(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    rs1 = _load_field(instruction, "rs1")
    imm = _load_field(instruction, "imm")
    base = state.scalar_regs.read(rs1)
    target = _u32(base + imm) & ~0x1
    link_value = _next_pc(pc)
    _check_alignment(target, 4, config, "jump")

    def on_complete() -> None:
        state.scalar_regs.write(rd, link_value)
        state.jump(target)
        state.fetch_stalled = False

    return _scalar_plan(cycle, config, "scalar", on_complete, f"jalr @ 0x{pc:08x}")


def _plan_branch(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rs1 = _load_field(instruction, "rs1")
    rs2 = _load_field(instruction, "rs2")
    lhs = state.scalar_regs.read(rs1)
    rhs = state.scalar_regs.read(rs2)
    if instruction.opcode == "beq":
        taken = lhs == rhs
    elif instruction.opcode == "bne":
        taken = lhs != rhs
    elif instruction.opcode == "blt":
        taken = _s32(lhs) < _s32(rhs)
    elif instruction.opcode == "bge":
        taken = _s32(lhs) >= _s32(rhs)
    elif instruction.opcode == "bltu":
        taken = _u32(lhs) < _u32(rhs)
    elif instruction.opcode == "bgeu":
        taken = _u32(lhs) >= _u32(rhs)
    else:
        raise KeyError(f"Unsupported branch opcode {instruction.opcode!r}.")
    target = _u32(pc + _load_field(instruction, "imm")) if taken else _next_pc(pc)
    _check_alignment(target, 4, config, "branch")

    def on_complete() -> None:
        state.jump(target)
        state.fetch_stalled = False

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_load(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    rs1 = _load_field(instruction, "rs1")
    imm = _load_field(instruction, "imm")
    address = _u32(state.scalar_regs.read(rs1) + imm)
    width, signed = LOAD_WIDTHS[instruction.opcode]
    _check_alignment(address, width, config, "load")
    memory, _local_address = state.resolve_memory(address, config)
    if memory is state.scratchpad:
        latency = scratchpad_latency(config.scratchpad, width)
    else:
        latency = dram_read_latency(config.dram, width)
    completion_cycle = cycle + latency

    def on_complete() -> None:
        raw = state.read_memory(address, width, config)
        value = int.from_bytes(raw, byteorder="little", signed=False)
        if signed:
            sign_bit = 1 << (width * 8 - 1)
            if value & sign_bit:
                value -= 1 << (width * 8)
        state.scalar_regs.write(rd, value)

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="load_store",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            )
        ],
        on_complete=on_complete,
        description=f"{instruction.opcode} @ 0x{pc:08x}",
        stats={"bytes_read": width},
    )


def _plan_store(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rs1 = _load_field(instruction, "rs1")
    rs2 = _load_field(instruction, "rs2")
    imm = _load_field(instruction, "imm")
    address = _u32(state.scalar_regs.read(rs1) + imm)
    value = state.scalar_regs.read(rs2)
    width = STORE_WIDTHS[instruction.opcode]
    _check_alignment(address, width, config, "store")
    memory, _local_address = state.resolve_memory(address, config)
    if memory is state.scratchpad:
        latency = scratchpad_latency(config.scratchpad, width)
    else:
        latency = dram_write_latency(config.dram, width)
    completion_cycle = cycle + latency

    def on_complete() -> None:
        masked_value = value & ((1 << (width * 8)) - 1)
        state.write_memory(
            address,
            masked_value.to_bytes(width, byteorder="little", signed=False),
            config,
        )

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="load_store",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            )
        ],
        on_complete=on_complete,
        description=f"{instruction.opcode} @ 0x{pc:08x}",
        stats={"bytes_written": width},
    )


def _plan_fence(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")

    def on_complete() -> None:
        _ = state

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_system(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")

    def on_complete() -> None:
        if instruction.opcode == "ecall" and config.machine.halt_on_ecall:
            state.halt(exit_code=state.scalar_regs.read(10))
            return
        if instruction.opcode == "ebreak" and config.machine.halt_on_ebreak:
            state.halt(exit_code=state.scalar_regs.read(10))
            return
        state.trap(f"Unhandled system opcode {instruction.opcode} at 0x{pc:08x}")

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def build_rv32i_semantics_registry() -> SemanticsRegistry:
    """Construct the default RV32I semantics registry."""
    registry = SemanticsRegistry()
    for opcode in ("addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai"):
        registry.register(opcode, _plan_immediate_alu)
    for opcode in ("add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"):
        registry.register(opcode, _plan_register_alu)
    for opcode in ("lui", "auipc"):
        registry.register(opcode, _plan_u_type)
    registry.register("jal", _plan_jal)
    registry.register("jalr", _plan_jalr)
    for opcode in ("beq", "bne", "blt", "bge", "bltu", "bgeu"):
        registry.register(opcode, _plan_branch)
    for opcode in LOAD_WIDTHS:
        registry.register(opcode, _plan_load)
    for opcode in STORE_WIDTHS:
        registry.register(opcode, _plan_store)
    registry.register("fence", _plan_fence)
    registry.register("ecall", _plan_system)
    registry.register("ebreak", _plan_system)
    return registry


DEFAULT_SEMANTICS = build_rv32i_semantics_registry()
