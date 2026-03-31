"""Instruction semantics registry."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING

from perf_modeling.isa.instruction import ExecutionPlan, Instruction
from perf_modeling.state.arch_state import TensorValue
from perf_modeling.state.csr_file import CSR_MEPC
from perf_modeling.timing.latency import (
    dma_latency,
    dram_read_latency,
    dram_write_latency,
    mxu_latency,
    scalar_latency,
    scratchpad_access_latency,
    vector_latency,
)
from perf_modeling.timing.resources import ResourceReservation
from perf_modeling.traps import (
    CAUSE_BREAKPOINT,
    CAUSE_ENV_CALL_FROM_M_MODE,
    CAUSE_ILLEGAL_INSTRUCTION,
    CAUSE_INSTRUCTION_ADDRESS_MISALIGNED,
    CAUSE_LOAD_ADDRESS_MISALIGNED,
    CAUSE_STORE_ADDRESS_MISALIGNED,
    MachineTrap,
)
from perf_modeling.types import StorageLocation, TensorDescriptor

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

DTYPE_STRUCT_FORMATS = {
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "float16": "e",
    "float32": "f",
}

DTYPE_BYTE_WIDTHS = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "float16": 2,
    "float32": 4,
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


def _check_alignment(
    address: int,
    size: int,
    pc: int,
    config: "AcceleratorConfig",
    kind: str,
) -> None:
    """Validate an aligned architectural address when strict mode is enabled."""
    if not config.machine.strict_alignment:
        return
    if address % size != 0:
        cause = {
            "jump": CAUSE_INSTRUCTION_ADDRESS_MISALIGNED,
            "branch": CAUSE_INSTRUCTION_ADDRESS_MISALIGNED,
            "load": CAUSE_LOAD_ADDRESS_MISALIGNED,
            "store": CAUSE_STORE_ADDRESS_MISALIGNED,
        }[kind]
        raise MachineTrap(
            cause=cause,
            pc=pc,
            tval=address,
            reason=f"Misaligned {kind} access at 0x{address:08x} for size {size}.",
        )


def _csr_writes(instruction: Instruction) -> bool:
    """Return whether a CSR instruction updates its target CSR."""
    opcode = instruction.opcode
    if opcode == "csrrw" or opcode == "csrrwi":
        return True
    if opcode in {"csrrs", "csrrc"}:
        return _load_field(instruction, "rs1") != 0
    if opcode in {"csrrsi", "csrrci"}:
        return _load_field(instruction, "imm") != 0
    return False


def _tensor_shape(instruction: Instruction) -> tuple[int, ...]:
    """Return the logical tensor shape carried by an instruction."""
    shape = instruction.metadata.get("shape")
    if not isinstance(shape, tuple):
        raise ValueError(f"Instruction {instruction.opcode!r} is missing a tensor shape.")
    return tuple(int(dimension) for dimension in shape)


def _tensor_dtype(instruction: Instruction, key: str = "dtype") -> str:
    """Return the tensor dtype carried by an instruction."""
    dtype = instruction.metadata.get(key)
    if not isinstance(dtype, str):
        raise ValueError(f"Instruction {instruction.opcode!r} is missing dtype metadata {key!r}.")
    return dtype


def _tensor_num_elements(shape: tuple[int, ...]) -> int:
    """Return the total element count in a tensor shape."""
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _tensor_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Return the logical storage size of one tensor payload."""
    if dtype not in DTYPE_BYTE_WIDTHS:
        raise ValueError(f"Tensor load/store dtype {dtype!r} is not supported.")
    return _tensor_num_elements(shape) * DTYPE_BYTE_WIDTHS[dtype]


def _materialize_tensor(
    backend: "TensorBackend | None",
    shape: tuple[int, ...],
    dtype: str,
    values: tuple[object, ...],
) -> object:
    """Create a backend tensor from flattened Python values."""
    if backend is None:
        raise ValueError("Tensor backend is required for tensor instructions.")
    tensor = backend.zeros(shape, dtype)
    flat = tensor.reshape(-1)
    for index, value in enumerate(values):
        flat[index] = value
    return tensor


def _load_tensor_payload(
    backend: "TensorBackend | None",
    raw: bytes,
    shape: tuple[int, ...],
    dtype: str,
) -> object:
    """Deserialize a tensor payload from memory bytes."""
    if dtype not in DTYPE_STRUCT_FORMATS:
        raise ValueError(f"Tensor load/store dtype {dtype!r} is not supported.")
    count = _tensor_num_elements(shape)
    values = struct.unpack("<" + DTYPE_STRUCT_FORMATS[dtype] * count, raw)
    return _materialize_tensor(backend, shape, dtype, values)


def _store_tensor_payload(value: object, dtype: str) -> bytes:
    """Serialize a backend tensor into raw bytes."""
    if dtype not in DTYPE_STRUCT_FORMATS:
        raise ValueError(f"Tensor load/store dtype {dtype!r} is not supported.")
    flattened = value.reshape(-1).tolist()
    if dtype.startswith("float"):
        normalized = [float(element) for element in flattened]
    else:
        normalized = [int(element) for element in flattened]
    return struct.pack("<" + DTYPE_STRUCT_FORMATS[dtype] * len(normalized), *normalized)


def _memory_resource_reservation(
    memory_name: str,
    cycle: int,
    completion_cycle: int,
) -> ResourceReservation:
    """Create one shared memory-layer reservation over an operation lifetime."""
    return ResourceReservation(
        resource_name=f"mem_{memory_name}",
        start_cycle=cycle,
        end_cycle=completion_cycle,
    )


def _scratchpad_bank_reservations(
    state: "ArchState",
    local_address: int,
    size_bytes: int,
    cycle: int,
    completion_cycle: int,
) -> list[ResourceReservation]:
    """Create shared scratchpad-bank reservations for one local access range."""
    return [
        ResourceReservation(
            resource_name=f"sp_bank_{bank_index}",
            start_cycle=cycle,
            end_cycle=completion_cycle,
        )
        for bank_index in state.scratchpad.bank_indices_for_range(local_address, size_bytes)
    ]


def _scratchpad_port_reservation(
    scoreboard: "Scoreboard",
    config: "AcceleratorConfig",
    cycle: int,
    completion_cycle: int,
    is_write: bool,
) -> ResourceReservation:
    """Reserve one scratchpad read or write port across the access lifetime."""
    if is_write:
        port_prefix = "sp_write_port"
        port_count = config.scratchpad.write_ports
    else:
        port_prefix = "sp_read_port"
        port_count = config.scratchpad.read_ports
    for port_index in range(max(1, port_count)):
        resource_name = f"{port_prefix}_{port_index}"
        if scoreboard.resource_ready(resource_name, cycle, completion_cycle):
            return ResourceReservation(
                resource_name=resource_name,
                start_cycle=cycle,
                end_cycle=completion_cycle,
            )
    return ResourceReservation(
        resource_name=f"{port_prefix}_0",
        start_cycle=cycle,
        end_cycle=completion_cycle,
    )


def _scratchpad_latency(
    state: "ArchState",
    config: "AcceleratorConfig",
    local_address: int,
    size_bytes: int,
) -> int:
    """Estimate scratchpad latency using the banks touched by one access."""
    banks_touched = len(state.scratchpad.bank_indices_for_range(local_address, size_bytes))
    return scratchpad_access_latency(config.scratchpad, banks_touched, size_bytes)


def _memory_access_reservations(
    state: "ArchState",
    config: "AcceleratorConfig",
    scoreboard: "Scoreboard",
    memory_name: str,
    local_address: int,
    size_bytes: int,
    cycle: int,
    completion_cycle: int,
    is_write: bool,
) -> list[ResourceReservation]:
    """Create shared reservations for one memory-layer access."""
    if memory_name == state.scratchpad.name:
        return [
            *_scratchpad_bank_reservations(state, local_address, size_bytes, cycle, completion_cycle),
            _scratchpad_port_reservation(scoreboard, config, cycle, completion_cycle, is_write),
        ]
    return [_memory_resource_reservation(memory_name, cycle, completion_cycle)]


def _memory_byte_stats(memory_name: str, num_bytes: int, is_write: bool) -> dict[str, int]:
    """Create per-memory byte counters for one access."""
    if is_write:
        return {f"{memory_name}.bytes_written": num_bytes}
    return {f"{memory_name}.bytes_read": num_bytes}


def _require_tensor(state: "ArchState", index: int, pc: int, name: str) -> TensorValue:
    """Read one tensor register and raise an architectural trap when it is empty."""
    tensor_value = state.tensor_regs.read(index)
    if tensor_value is None:
        raise MachineTrap(
            cause=CAUSE_ILLEGAL_INSTRUCTION,
            pc=pc,
            tval=index,
            reason=f"{name} tensor register t{index} is empty at 0x{pc:08x}.",
        )
    return tensor_value


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
    _check_alignment(target, 4, pc, config, "jump")

    def on_complete() -> None:
        state.scalar_regs.write(rd, link_value)
        state.jump(target)
        state.fetch_stalled = False
        state.fetch_stall_reason = None

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
    _check_alignment(target, 4, pc, config, "jump")

    def on_complete() -> None:
        state.scalar_regs.write(rd, link_value)
        state.jump(target)
        state.fetch_stalled = False
        state.fetch_stall_reason = None

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
    _check_alignment(target, 4, pc, config, "branch")

    def on_complete() -> None:
        state.jump(target)
        state.fetch_stalled = False
        state.fetch_stall_reason = None

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_load(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    rs1 = _load_field(instruction, "rs1")
    imm = _load_field(instruction, "imm")
    address = _u32(state.scalar_regs.read(rs1) + imm)
    width, signed = LOAD_WIDTHS[instruction.opcode]
    _check_alignment(address, width, pc, config, "load")
    memory, local_address = state.resolve_memory(address, config)
    if memory is state.scratchpad:
        latency = _scratchpad_latency(state, config, local_address, width)
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
            ),
            *_memory_access_reservations(
                state,
                config,
                scoreboard,
                memory.name,
                local_address,
                width,
                cycle,
                completion_cycle,
                False,
            ),
        ],
        on_complete=on_complete,
        description=f"{instruction.opcode} @ 0x{pc:08x}",
        stats={"bytes_read": width, **_memory_byte_stats(memory.name, width, False)},
    )


def _plan_store(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rs1 = _load_field(instruction, "rs1")
    rs2 = _load_field(instruction, "rs2")
    imm = _load_field(instruction, "imm")
    address = _u32(state.scalar_regs.read(rs1) + imm)
    value = state.scalar_regs.read(rs2)
    width = STORE_WIDTHS[instruction.opcode]
    _check_alignment(address, width, pc, config, "store")
    memory, local_address = state.resolve_memory(address, config)
    if memory is state.scratchpad:
        latency = _scratchpad_latency(state, config, local_address, width)
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
            ),
            *_memory_access_reservations(
                state,
                config,
                scoreboard,
                memory.name,
                local_address,
                width,
                cycle,
                completion_cycle,
                True,
            ),
        ],
        on_complete=on_complete,
        description=f"{instruction.opcode} @ 0x{pc:08x}",
        stats={"bytes_written": width, **_memory_byte_stats(memory.name, width, True)},
    )


def _plan_tload(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    scoreboard: "Scoreboard",
    backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    dest_tensor = instruction.dest_tensors()[0]
    address = _load_field(instruction, "address")
    shape = _tensor_shape(instruction)
    dtype = _tensor_dtype(instruction)
    num_bytes = _tensor_nbytes(shape, dtype)
    memory, local_address = state.resolve_memory(address, config)
    if memory is state.scratchpad:
        latency = _scratchpad_latency(state, config, local_address, num_bytes)
    else:
        latency = dram_read_latency(config.dram, num_bytes)
    completion_cycle = cycle + latency

    def on_complete() -> None:
        raw = state.read_memory(address, num_bytes, config)
        payload = _load_tensor_payload(backend, raw, shape, dtype)
        descriptor = TensorDescriptor(
            shape=shape,
            dtype=dtype,
            location=StorageLocation.REGISTER,
            name=f"t{dest_tensor}",
        )
        state.tensor_regs.write(dest_tensor, TensorValue(descriptor=descriptor, payload=payload))

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="load_store",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            ),
            *_memory_access_reservations(
                state,
                config,
                scoreboard,
                memory.name,
                local_address,
                num_bytes,
                cycle,
                completion_cycle,
                False,
            ),
        ],
        on_complete=on_complete,
        description=f"tload @ 0x{pc:08x}",
        stats={"bytes_read": num_bytes, **_memory_byte_stats(memory.name, num_bytes, False)},
    )


def _plan_tstore(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    source_tensor = instruction.source_tensors()[0]
    address = _load_field(instruction, "address")
    tensor_value = _require_tensor(state, source_tensor, pc, "Source")
    dtype = tensor_value.descriptor.dtype
    raw = _store_tensor_payload(tensor_value.payload, dtype)
    memory, local_address = state.resolve_memory(address, config)
    if memory is state.scratchpad:
        latency = _scratchpad_latency(state, config, local_address, len(raw))
    else:
        latency = dram_write_latency(config.dram, len(raw))
    completion_cycle = cycle + latency

    def on_complete() -> None:
        state.write_memory(address, raw, config)

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="load_store",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            ),
            *_memory_access_reservations(
                state,
                config,
                scoreboard,
                memory.name,
                local_address,
                len(raw),
                cycle,
                completion_cycle,
                True,
            ),
        ],
        on_complete=on_complete,
        description=f"tstore @ 0x{pc:08x}",
        stats={"bytes_written": len(raw), **_memory_byte_stats(memory.name, len(raw), True)},
    )


def _plan_dma_copy(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    source_address = _load_field(instruction, "source_address")
    dest_address = _load_field(instruction, "dest_address")
    num_bytes = _load_field(instruction, "num_bytes")
    source_memory, source_local_address = state.resolve_memory(source_address, config)
    dest_memory, dest_local_address = state.resolve_memory(dest_address, config)
    completion_cycle = cycle + dma_latency(config.core.dma, num_bytes)
    transfer_start_cycle = cycle + config.core.dma.setup_cycles
    shared_resources = _memory_access_reservations(
        state,
        config,
        scoreboard,
        source_memory.name,
        source_local_address,
        num_bytes,
        transfer_start_cycle,
        completion_cycle,
        False,
    )
    for reservation in _memory_access_reservations(
        state,
        config,
        scoreboard,
        dest_memory.name,
        dest_local_address,
        num_bytes,
        transfer_start_cycle,
        completion_cycle,
        True,
    ):
        if reservation not in shared_resources:
            shared_resources.append(reservation)

    def on_complete() -> None:
        payload = state.read_memory(source_address, num_bytes, config)
        state.write_memory(dest_address, payload, config)

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="dma",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            ),
            *shared_resources,
        ],
        on_complete=on_complete,
        description=f"dma_copy @ 0x{pc:08x}",
        stats={
            "bytes_read": num_bytes,
            "bytes_written": num_bytes,
            **_memory_byte_stats(source_memory.name, num_bytes, False),
            **_memory_byte_stats(dest_memory.name, num_bytes, True),
        },
    )


def _plan_vadd(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    lhs_index, rhs_index = instruction.source_tensors()
    dest_index = instruction.dest_tensors()[0]
    lhs = _require_tensor(state, lhs_index, pc, "Left")
    rhs = _require_tensor(state, rhs_index, pc, "Right")
    if lhs.descriptor.shape != rhs.descriptor.shape:
        raise MachineTrap(
            cause=CAUSE_ILLEGAL_INSTRUCTION,
            pc=pc,
            tval=0,
            reason=f"Vector add shape mismatch at 0x{pc:08x}: {lhs.descriptor.shape} vs {rhs.descriptor.shape}.",
        )
    out_dtype = _tensor_dtype(instruction, "out_dtype")
    elements = _tensor_num_elements(lhs.descriptor.shape)
    completion_cycle = cycle + vector_latency(config.core.vector, elements)

    def on_complete() -> None:
        if backend is None:
            raise ValueError("Tensor backend is required for vector instructions.")
        payload = backend.elementwise("add", (lhs.payload, rhs.payload), out_dtype)
        descriptor = TensorDescriptor(
            shape=lhs.descriptor.shape,
            dtype=out_dtype,
            location=StorageLocation.REGISTER,
            name=f"t{dest_index}",
        )
        state.tensor_regs.write(dest_index, TensorValue(descriptor=descriptor, payload=payload))

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="vector",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            )
        ],
        on_complete=on_complete,
        description=f"vadd @ 0x{pc:08x}",
    )


def _plan_matmul(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    lhs_index, rhs_index = instruction.source_tensors()
    dest_index = instruction.dest_tensors()[0]
    lhs = _require_tensor(state, lhs_index, pc, "Left")
    rhs = _require_tensor(state, rhs_index, pc, "Right")
    if len(lhs.descriptor.shape) != 2 or len(rhs.descriptor.shape) != 2:
        raise MachineTrap(
            cause=CAUSE_ILLEGAL_INSTRUCTION,
            pc=pc,
            tval=0,
            reason=f"MXU matmul expects rank-2 tensors at 0x{pc:08x}.",
        )
    lhs_rows, lhs_cols = lhs.descriptor.shape
    rhs_rows, rhs_cols = rhs.descriptor.shape
    if lhs_cols != rhs_rows:
        raise MachineTrap(
            cause=CAUSE_ILLEGAL_INSTRUCTION,
            pc=pc,
            tval=0,
            reason=f"MXU matmul shape mismatch at 0x{pc:08x}: {lhs.descriptor.shape} x {rhs.descriptor.shape}.",
        )
    out_dtype = _tensor_dtype(instruction, "out_dtype")
    acc_dtype = _tensor_dtype(instruction, "acc_dtype")
    result_elements = lhs_rows * rhs_cols
    tile_capacity = max(1, config.core.mxu.rows * config.core.mxu.cols)
    completion_cycle = cycle + mxu_latency(config.core.mxu, math.ceil(result_elements / tile_capacity))

    def on_complete() -> None:
        if backend is None:
            raise ValueError("Tensor backend is required for MXU instructions.")
        payload = backend.matmul(lhs.payload, rhs.payload, acc_dtype, out_dtype)
        descriptor = TensorDescriptor(
            shape=(lhs_rows, rhs_cols),
            dtype=out_dtype,
            location=StorageLocation.REGISTER,
            name=f"t{dest_index}",
        )
        state.tensor_regs.write(dest_index, TensorValue(descriptor=descriptor, payload=payload))

    return ExecutionPlan(
        completion_cycle=completion_cycle,
        resources=[
            ResourceReservation(
                resource_name="mxu",
                start_cycle=cycle,
                end_cycle=completion_cycle,
            )
        ],
        on_complete=on_complete,
        description=f"matmul @ 0x{pc:08x}",
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
        if instruction.opcode == "ecall":
            raise MachineTrap(
                cause=CAUSE_ENV_CALL_FROM_M_MODE,
                pc=pc,
                tval=0,
                reason=f"Environment call from M-mode at 0x{pc:08x}.",
            )
        if instruction.opcode == "ebreak":
            raise MachineTrap(
                cause=CAUSE_BREAKPOINT,
                pc=pc,
                tval=0,
                reason=f"Breakpoint at 0x{pc:08x}.",
            )
        raise MachineTrap(
            cause=CAUSE_ILLEGAL_INSTRUCTION,
            pc=pc,
            tval=_load_field(instruction, "word"),
            reason=f"Unhandled system opcode {instruction.opcode} at 0x{pc:08x}.",
        )

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_csr(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    rd = _load_field(instruction, "rd")
    csr_address = instruction.source_csrs()[0]
    old_value = state.read_csr(csr_address, cycle)
    if instruction.opcode in {"csrrw", "csrrs", "csrrc"}:
        rs1 = _load_field(instruction, "rs1")
        operand_value = state.scalar_regs.read(rs1)
    else:
        operand_value = _load_field(instruction, "imm")
    write_value = old_value
    if instruction.opcode in {"csrrw", "csrrwi"}:
        write_value = operand_value
    elif instruction.opcode in {"csrrs", "csrrsi"}:
        write_value = old_value | operand_value
    elif instruction.opcode in {"csrrc", "csrrci"}:
        write_value = old_value & ~operand_value

    def on_complete() -> None:
        if _csr_writes(instruction):
            state.write_csr(csr_address, write_value, pc)
        state.scalar_regs.write(rd, old_value)

    return _scalar_plan(cycle, config, "scalar", on_complete, f"{instruction.opcode} @ 0x{pc:08x}")


def _plan_mret(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    target = state.read_csr(CSR_MEPC, cycle)
    _check_alignment(target, 4, pc, config, "jump")

    def on_complete() -> None:
        state.jump(target)
        state.fetch_stalled = False
        state.fetch_stall_reason = None

    return _scalar_plan(cycle, config, "scalar", on_complete, f"mret @ 0x{pc:08x}")


def _plan_illegal(
    instruction: Instruction,
    cycle: int,
    state: "ArchState",
    config: "AcceleratorConfig",
    _scoreboard: "Scoreboard",
    _backend: "TensorBackend | None",
) -> ExecutionPlan:
    pc = _load_field(instruction, "pc")
    word = _load_field(instruction, "word")
    reason = str(instruction.metadata.get("illegal_reason", f"Illegal instruction at 0x{pc:08x}."))

    def on_complete() -> None:
        raise MachineTrap(
            cause=CAUSE_ILLEGAL_INSTRUCTION,
            pc=pc,
            tval=word,
            reason=reason,
        )

    return _scalar_plan(cycle, config, "scalar", on_complete, f"illegal @ 0x{pc:08x}")


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
    registry.register("dma_copy", _plan_dma_copy)
    registry.register("tload", _plan_tload)
    registry.register("tstore", _plan_tstore)
    registry.register("vadd", _plan_vadd)
    registry.register("matmul", _plan_matmul)
    registry.register("fence", _plan_fence)
    for opcode in ("csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"):
        registry.register(opcode, _plan_csr)
    registry.register("ecall", _plan_system)
    registry.register("ebreak", _plan_system)
    registry.register("mret", _plan_mret)
    registry.register("illegal", _plan_illegal)
    return registry


DEFAULT_SEMANTICS = build_rv32i_semantics_registry()
