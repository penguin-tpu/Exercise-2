"""Top-level simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.backend.torch_backend import TorchTensorBackend
from perf_modeling.config import AcceleratorConfig
from perf_modeling.events import CompletionEvent, EventQueue
from perf_modeling.program import Program
from perf_modeling.state.arch_state import ArchState
from perf_modeling.stats import SimulationStats
from perf_modeling.timing.scoreboard import Scoreboard
from perf_modeling.trace import TraceRecorder
from perf_modeling.traps import MachineTrap
from perf_modeling.types import Cycle
from perf_modeling.units.dma import DMAUnit
from perf_modeling.units.load_store import LoadStoreUnit
from perf_modeling.units.mxu import MXUUnit
from perf_modeling.units.scalar import ScalarUnit
from perf_modeling.units.vector import VectorUnit


@dataclass
class SimulatorEngine:
    """Execution-driven, tick-based accelerator simulator scaffold."""

    config: AcceleratorConfig
    program: Program
    state: ArchState = field(init=False)
    cycle: Cycle = 0
    next_op_id: int = 1
    event_queue: EventQueue = field(default_factory=EventQueue)
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    stats: SimulationStats = field(default_factory=SimulationStats)
    trace: TraceRecorder = field(init=False)
    backend: TorchTensorBackend = field(init=False)
    scalar_unit: ScalarUnit = field(init=False)
    vector_unit: VectorUnit = field(init=False)
    mxu_unit: MXUUnit = field(init=False)
    dma_unit: DMAUnit = field(init=False)
    load_store_unit: LoadStoreUnit = field(init=False)

    def __post_init__(self) -> None:
        """Initialize architectural state, units, and trace sinks."""
        self.state = ArchState.from_config(self.config)
        self.trace = TraceRecorder(max_records=self.config.trace.max_records)
        self.backend = TorchTensorBackend()
        self.scalar_unit = ScalarUnit("scalar", self.config.core.scalar)
        self.vector_unit = VectorUnit("vector", self.config.core.vector)
        self.mxu_unit = MXUUnit("mxu", self.config.core.mxu)
        self.dma_unit = DMAUnit("dma", self.config.core.dma)
        self.load_store_unit = LoadStoreUnit("load_store")
        self.state.load_program(self.program, self.config)

    def reset(self) -> None:
        """Reset the simulator to cycle zero and clear all transient state."""
        self.cycle = 0
        self.next_op_id = 1
        self.event_queue.clear()
        self.scoreboard.clear()
        self.stats = SimulationStats()
        self.trace.clear()
        self.state.load_program(self.program, self.config)

    def is_done(self) -> bool:
        """Return whether the program and all in-flight work have completed."""
        if self.state.halted:
            return not self.state.outstanding_ops and self.event_queue.pending_count() == 0
        return (
            self.program.is_done(self.state.pc)
            and not self.state.outstanding_ops
            and self.event_queue.pending_count() == 0
        )

    def step(self) -> None:
        """Advance the simulator by one architectural cycle."""
        self.retire_ready_events()
        self.update_resources()
        self.try_issue_head_instruction()
        self.sample_stats()
        self.cycle += 1

    def run(self, max_cycles: int | None = None) -> SimulationStats:
        """Run until completion or until the optional cycle limit is hit."""
        while not self.is_done():
            if max_cycles is not None and self.cycle >= max_cycles:
                break
            self.step()
        return self.stats

    def retire_ready_events(self) -> None:
        """Commit all operations whose completion cycle has arrived."""
        for event in self.event_queue.pop_ready(self.cycle):
            event.fire()
            self.state.complete_op(event.op_id)
            self._append_trace("complete", event.description or f"op {event.op_id}")

    def update_resources(self) -> None:
        """Advance unit-local timekeeping and refresh scoreboard state."""
        for unit in self.iter_units():
            unit.tick(self.cycle)
        self.state.scratchpad.release_banks()

    def try_issue_head_instruction(self) -> None:
        """Attempt to issue the next instruction if all checks pass."""
        if self.state.halted or self.state.fetch_stalled:
            return
        if self.program.is_done(self.state.pc):
            return
        instruction = self.program.instruction_at(self.state.pc)
        if any(not self.scoreboard.scalar_ready(index) for index in instruction.source_regs()):
            self._record_stall("stall_scalar_dependency", f"scalar dependency @ 0x{self.state.pc:08x}")
            return
        if any(not self.scoreboard.scalar_ready(index) for index in instruction.dest_regs()):
            self._record_stall("stall_scalar_waw", f"scalar WAW @ 0x{self.state.pc:08x}")
            return
        if any(not self.scoreboard.tensor_ready(index) for index in instruction.source_tensors()):
            self._record_stall("stall_tensor_dependency", f"tensor dependency @ 0x{self.state.pc:08x}")
            return
        if any(not self.scoreboard.tensor_ready(index) for index in instruction.dest_tensors()):
            self._record_stall("stall_tensor_waw", f"tensor WAW @ 0x{self.state.pc:08x}")
            return
        if any(not self.scoreboard.csr_ready(address) for address in instruction.source_csrs()):
            self._record_stall("stall_csr_dependency", f"csr dependency @ 0x{self.state.pc:08x}")
            return
        if any(not self.scoreboard.csr_ready(address) for address in instruction.dest_csrs()):
            self._record_stall("stall_csr_waw", f"csr WAW @ 0x{self.state.pc:08x}")
            return
        if instruction.opcode == "fence" and self.state.outstanding_ops:
            self._record_stall("stall_fence", f"fence wait @ 0x{self.state.pc:08x}")
            return
        unit = self._unit_for_instruction(instruction)
        if not unit.can_accept():
            self._record_stall(f"stall_{unit.name}_busy", f"{unit.name} busy @ 0x{self.state.pc:08x}")
            return
        try:
            plan = instruction.plan(
                cycle=self.cycle,
                state=self.state,
                config=self.config,
                scoreboard=self.scoreboard,
                backend=self.backend,
            )
        except MachineTrap as trap:
            self._deliver_trap(trap)
            return
        except Exception as exc:
            self.state.trap(str(exc))
            self._append_trace("trap", str(exc))
            return
        shared_reservations = self._shared_resource_reservations(plan.resources)
        for reservation in shared_reservations:
            if not self.scoreboard.resource_ready(
                reservation.resource_name,
                reservation.start_cycle,
                reservation.end_cycle,
            ):
                self._record_stall(
                    f"stall_{reservation.resource_name}_busy",
                    f"{reservation.resource_name} busy @ 0x{self.state.pc:08x}",
                )
                return
        op_id = self.next_op_id
        self.next_op_id += 1
        for register in instruction.dest_regs():
            self.scoreboard.mark_scalar_busy(register)
        for tensor in instruction.dest_tensors():
            self.scoreboard.mark_tensor_busy(tensor)
        for address in instruction.dest_csrs():
            self.scoreboard.mark_csr_busy(address)
        for reservation in shared_reservations:
            self.scoreboard.reserve_resource(
                reservation.resource_name,
                reservation.start_cycle,
                reservation.end_cycle,
            )
        completion_callback = self._wrap_completion_callback(
            instruction=instruction,
            unit=unit,
            plan=plan,
        )
        unit.issue(plan.completion_cycle)
        self.state.mark_op_outstanding(op_id)
        self.event_queue.schedule(
            event=CompletionEvent(
                ready_cycle=plan.completion_cycle,
                op_id=op_id,
                callback=completion_callback,
                description=plan.description or instruction.opcode,
            )
        )
        for key, value in plan.stats.items():
            if isinstance(value, int):
                self.stats.increment(key, value)
        self.stats.record_instruction_latency(
            instruction.opcode,
            max(0, plan.completion_cycle - self.cycle),
        )
        if instruction.metadata.get("is_control", False):
            self.state.fetch_stalled = True
        else:
            self.state.next_pc(self.config.machine.instruction_bytes)
        self.stats.increment("instructions_issued", 1)
        self.stats.record_issue(unit.name)
        self._append_trace("issue", plan.description or f"{instruction.opcode} @ 0x{self.state.pc:08x}")

    def sample_stats(self) -> None:
        """Update cycle-level statistics after issue and completion work."""
        self.stats.increment("cycles", 1)
        self.stats.record_event_queue_occupancy(self.event_queue.pending_count())
        for unit in self.iter_units():
            self.stats.record_queue_occupancy(unit.name, unit.status.queued_ops)
            if unit.is_busy():
                self.stats.record_busy_cycle(unit.name)

    def _record_stall(self, counter_name: str, message: str) -> None:
        """Increment one stall counter and append a matching trace record."""
        self.stats.increment(counter_name, 1)
        self._record_contention_stats(counter_name)
        self._append_trace("stall", message)

    def _record_contention_stats(self, counter_name: str) -> None:
        """Update aggregate contention statistics derived from stall categories."""
        if counter_name.startswith("stall_mem_dram_"):
            self.stats.increment("dram.contention_stalls", 1)
            self.stats.increment("memory.contention_stalls", 1)
            self.stats.increment("memory.contention.resource.mem_dram", 1)
            return
        if counter_name.startswith("stall_mem_"):
            self.stats.increment("memory.contention_stalls", 1)
            resource_name = counter_name.removeprefix("stall_").removesuffix("_busy")
            self.stats.increment(f"memory.contention.resource.{resource_name}", 1)
            return
        if counter_name.startswith("stall_sp_bank_"):
            self.stats.increment("scratchpad.bank_conflict_stalls", 1)
            self.stats.increment("scratchpad.contention_stalls", 1)
            bank_name = counter_name.removeprefix("stall_").removesuffix("_busy")
            self.stats.increment(f"scratchpad.bank_conflict.{bank_name}", 1)
            return
        if counter_name.startswith("stall_sp_read_port_") or counter_name.startswith("stall_sp_write_port_"):
            self.stats.increment("scratchpad.port_conflict_stalls", 1)
            self.stats.increment("scratchpad.contention_stalls", 1)
            port_name = counter_name.removeprefix("stall_").removesuffix("_busy")
            self.stats.increment(f"scratchpad.port_conflict.{port_name}", 1)

    def _append_trace(self, kind: str, message: str) -> None:
        """Append one trace record when the current trace policy allows it."""
        if not self.config.timing.enable_tracing:
            return
        if kind == "stall" and not self.config.trace.keep_cycle_trace:
            return
        if kind in {"issue", "complete", "trap"} and not self.config.trace.keep_event_trace:
            return
        self.trace.append(self.cycle, kind, message)

    def _shared_resource_reservations(self, reservations: list[object]) -> list[object]:
        """Return the reservations that model cross-unit shared resources."""
        return [
            reservation
            for reservation in reservations
            if reservation.resource_name.startswith("mem_")
            or reservation.resource_name.startswith("sp_bank_")
            or reservation.resource_name.startswith("sp_read_port_")
            or reservation.resource_name.startswith("sp_write_port_")
        ]

    def iter_units(self) -> tuple[object, ...]:
        """Return all modeled units in a stable iteration order."""
        return (
            self.scalar_unit,
            self.vector_unit,
            self.mxu_unit,
            self.dma_unit,
            self.load_store_unit,
        )

    def _unit_for_instruction(self, instruction: object) -> object:
        """Resolve the execution-unit object for one decoded instruction."""
        unit_name = instruction.unit_name()
        if unit_name == "scalar":
            return self.scalar_unit
        if unit_name == "load_store":
            return self.load_store_unit
        if unit_name == "dma":
            return self.dma_unit
        if unit_name == "vector":
            return self.vector_unit
        if unit_name == "mxu":
            return self.mxu_unit
        raise KeyError(f"Unsupported unit selection {unit_name!r}.")

    def _wrap_completion_callback(self, instruction: object, unit: object, plan: object) -> object:
        """Wrap the instruction completion to release hazards and unit occupancy."""

        def callback() -> None:
            try:
                plan.on_complete()
            except MachineTrap as trap:
                self._deliver_trap(trap)
            except Exception as exc:
                self.state.trap(str(exc))
                self._append_trace("trap", str(exc))
            unit.complete()
            for register in instruction.dest_regs():
                self.scoreboard.release_scalar(register)
            for tensor in instruction.dest_tensors():
                self.scoreboard.release_tensor(tensor)
            for address in instruction.dest_csrs():
                self.scoreboard.release_csr(address)
            self.state.retire_instruction()
            self.stats.increment("instructions_retired", 1)

        return callback

    def _deliver_trap(self, trap: MachineTrap) -> None:
        """Deliver one machine trap either into `mtvec` or into a terminal halt."""
        if self.config.machine.enable_trap_handlers:
            target = self.state.enter_trap(trap, self.cycle)
            if self.program.contains_pc(target):
                self._append_trace("trap", trap.reason)
                return
        self.state.trap(trap.reason)
        self._append_trace("trap", trap.reason)
