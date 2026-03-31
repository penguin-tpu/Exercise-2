"""Top-level simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_modeling.config import AcceleratorConfig
from perf_modeling.events import EventQueue
from perf_modeling.program import Program
from perf_modeling.state.arch_state import ArchState
from perf_modeling.stats import SimulationStats
from perf_modeling.timing.scoreboard import Scoreboard
from perf_modeling.trace import TraceRecorder
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
    event_queue: EventQueue = field(default_factory=EventQueue)
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    stats: SimulationStats = field(default_factory=SimulationStats)
    trace: TraceRecorder = field(init=False)
    scalar_unit: ScalarUnit = field(init=False)
    vector_unit: VectorUnit = field(init=False)
    mxu_unit: MXUUnit = field(init=False)
    dma_unit: DMAUnit = field(init=False)
    load_store_unit: LoadStoreUnit = field(init=False)

    def __post_init__(self) -> None:
        """Initialize architectural state, units, and trace sinks."""
        self.state = ArchState.from_config(self.config)
        self.trace = TraceRecorder(max_records=self.config.trace.max_records)
        self.scalar_unit = ScalarUnit("scalar", self.config.core.scalar)
        self.vector_unit = VectorUnit("vector", self.config.core.vector)
        self.mxu_unit = MXUUnit("mxu", self.config.core.mxu)
        self.dma_unit = DMAUnit("dma", self.config.core.dma)
        self.load_store_unit = LoadStoreUnit("load_store")

    def reset(self) -> None:
        """Reset the simulator to cycle zero and clear all transient state."""
        self.cycle = 0
        self.event_queue.clear()
        self.scoreboard.clear()
        self.stats = SimulationStats()
        self.trace.clear()
        self.state.reset()

    def is_done(self) -> bool:
        """Return whether the program and all in-flight work have completed."""
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
            self.trace.append(self.cycle, "complete", event.description or f"op {event.op_id}")

    def update_resources(self) -> None:
        """Advance unit-local timekeeping and refresh scoreboard state."""
        for unit in self.iter_units():
            unit.tick(self.cycle)

    def try_issue_head_instruction(self) -> None:
        """Attempt to issue the next instruction if all checks pass."""
        if self.program.is_done(self.state.pc):
            return
        instruction = self.program.instruction_at(self.state.pc)
        raise NotImplementedError(
            f"Issue path is not implemented yet; next opcode is {instruction.opcode!r}."
        )

    def sample_stats(self) -> None:
        """Update cycle-level statistics after issue and completion work."""
        self.stats.increment("cycles", 1)
        for unit in self.iter_units():
            if unit.is_busy():
                self.stats.record_busy_cycle(unit.name)

    def iter_units(self) -> tuple[object, ...]:
        """Return all modeled units in a stable iteration order."""
        return (
            self.scalar_unit,
            self.vector_unit,
            self.mxu_unit,
            self.dma_unit,
            self.load_store_unit,
        )
