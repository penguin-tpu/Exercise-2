# Concrete Plan for a Combined Functional + Performance Model

## Recommendation

Build the simulator as a **single execution-driven, tick-based model** with a strict split between:

- **architectural state**: registers, memories, queues, PC, predicates, fences
- **resource/timing state**: unit occupancy, DMA progress, bank conflicts, bandwidth limits
- **tensor compute backend**: Torch for data movement and tensor math, not for control flow

The right abstraction for your accelerator is **ISA-driven functional execution with event-scheduled completion**. That fits an in-order, data-parallel machine much better than a mapping-only analytical model.

The model should answer three questions at once:

1. Is the program architecturally correct?
2. How many cycles does it take under the current hardware parameters?
3. Why did it take those cycles?

---

## Design Principles

### 1. One simulator, two kinds of state

Do not build a completely separate functional simulator and performance simulator at the beginning. That usually creates duplicated logic and eventual drift.

Instead, build one simulator where:

- instruction semantics define **what result must happen**
- timing/resources define **when the result becomes visible**

This keeps correctness and cycle counts tied to the same execution trace.

### 2. Tick-based outer loop, event-driven inner mechanics

Use a **cycle-by-cycle main loop** so stalls, fences, and overlaps are easy to reason about. Inside that loop, use **completion events** for long-latency operations instead of re-executing tensor math every cycle.

That gives you:

- cycle visibility for debugging
- efficient handling of long operations
- clear modeling of in-order issue with asynchronous completion

### 3. Use Torch only for bulk data semantics

Use `torch` to implement:

- tensor ALU ops
- matrix multiply / convolution kernels
- datatype conversion and saturation
- reductions, elementwise ops, and layout transforms

Do not use Torch to represent the simulator itself. The simulator state machine, queues, scoreboards, and timing model should stay in normal Python classes.

### 4. Config-first architecture

Every hardware parameter that affects behavior or timing should live in a dataclass config tree. The simulator instance should be fully constructible from a config object plus a program/workload.

That is the right foundation for design-space exploration.

---

## Target Architecture Style

Model the accelerator using these conceptual layers:

1. **Program layer**
   - instruction stream or decoded micro-ops
   - static scheduling metadata
   - tensor shapes / tile descriptors

2. **Architectural state**
   - scalar register file
   - tensor register file or tensor descriptor table
   - scratchpads / local SRAMs
   - main memory model
   - control state: PC, barriers, fence counters, predicates

3. **Execution resources**
   - instruction issue slot
   - DMA engine(s)
   - MXU / tensor compute unit
   - vector unit / scalar unit
   - load-store path
   - memory banks / ports / crossbar constraints

4. **Timing engine**
   - resource reservation
   - latency computation
   - event queue
   - stall and dependency checks

5. **Statistics/reporting**
   - total cycles
   - busy/idle cycles per unit
   - bytes moved
   - bank conflict counts
   - queue occupancy histograms
   - instruction latency breakdown

---

## Recommended Python Package Layout

Use a real Python package name like `perf_modeling/`, not `perf-modeling/`.

Suggested structure:

```text
perf_modeling/
  __init__.py
  config.py
  types.py
  engine.py
  events.py
  program.py
  decode.py
  stats.py
  trace.py
  backend/
    __init__.py
    torch_backend.py
    quant.py
  state/
    __init__.py
    arch_state.py
    register_file.py
    tensor_file.py
    memory.py
    scratchpad.py
  units/
    __init__.py
    base.py
    dma.py
    mxu.py
    vector.py
    scalar.py
    load_store.py
  isa/
    __init__.py
    instruction.py
    formats.py
    semantics.py
  timing/
    __init__.py
    latency.py
    scoreboard.py
    resources.py
    banking.py
  workloads/
    kernels.py
    builders.py
  tests/
    ...
```

This separation matters:

- `state/` owns architected data
- `units/` owns unit-specific behavior
- `timing/` owns contention and latency rules
- `backend/` hides Torch-specific details

---

## Core Execution Model

### High-level flow

Each cycle should do roughly this:

1. Retire all events completing at the current cycle.
2. Update resource availability and wake blocked instructions.
3. Try to issue the next in-order instruction or bundle.
4. Start any newly issued operations by reserving resources and scheduling completion.
5. Advance the global cycle counter.

### Important rule

For long-latency instructions, **architectural results should become visible on completion**, not on issue.

Examples:

- DMA load updates the destination scratchpad when the DMA completion event fires.
- MXU op writes its destination tensor register when compute completes.
- Fence completes only when all tracked prior operations are done.

This is the cleanest way to keep correctness aligned with timing.

### Hybrid tick/event structure

Use a min-heap event queue keyed by `ready_cycle`, but still iterate cycle by cycle.

That means:

- the simulator remains easy to debug
- background operations are modeled naturally
- you avoid storing per-cycle intermediate tensor work

Pseudo-flow:

```python
while not sim.is_done():
    sim.retire_ready_events()
    sim.update_scoreboards()
    sim.try_issue_head_instruction()
    sim.sample_stats()
    sim.cycle += 1
```

---

## Instruction Modeling Strategy

Represent each decoded instruction as an object with three responsibilities:

1. **validate**  
   Check operands, shapes, register availability, and architectural legality.

2. **plan**  
   Compute resource needs, latency, memory traffic, and completion conditions.

3. **commit**  
   Apply the architected result when the operation completes.

Recommended interface:

```python
class Instruction:
    def validate(self, state, config) -> None: ...
    def plan(self, state, timing, backend) -> ExecutionPlan: ...

class ExecutionPlan:
    resources: list[ResourceReservation]
    completion_cycle: int
    on_complete: Callable[[ArchState], None]
    stats: dict[str, int | float]
```

This is better than mixing timing and state mutation directly into `tick()` methods for each unit.

---

## Functional State Model

Keep the architectural state explicit and boring.

Suggested contents of `ArchState`:

- `pc`
- scalar registers
- tensor registers or tensor descriptors
- predicate/status registers
- local SRAM / scratchpad memories
- main memory backing store
- outstanding operation IDs for fences/barriers
- program completion state

### Scalar vs tensor values

Do not store every tensor inline in instruction objects. Use handles or named storage locations.

Recommended pattern:

- scalar registers store Python integers / enums / small metadata
- tensor registers store references to `TensorValue`
- `TensorValue` wraps a Torch tensor plus metadata:
  - shape
  - dtype
  - layout
  - scale / zero-point if quantized
  - storage location

That keeps tensor lifetime and memory accounting manageable.

---

## Memory and Data Movement Model

This part will dominate realism.

Model memory as multiple layers:

1. **Host / DRAM memory**
   - large address space
   - bandwidth-limited
   - optional fixed latency + burst model

2. **On-chip SRAM / scratchpads**
   - banked
   - port-limited
   - lower latency

3. **Register or tensor file**
   - smallest / fastest
   - capacity-limited

### What to model explicitly

- bank conflicts
- read/write port limits
- burst size
- DMA setup latency
- transfer size dependent latency
- alignment penalties if they exist in the real design

### What not to over-model initially

- detailed DRAM protocol
- sub-cycle arbitration
- individual wire timing

Start with simple deterministic formulas and refine later.

---

## Timing Model

The timing model should answer: when can an instruction issue, and when does it complete?

### Resource checks before issue

At minimum, check:

- source operands ready
- destination hazards
- execution unit free or queue has space
- memory path available
- scratchpad banks available if relevant
- fence or barrier conditions satisfied

### Latency modeling

Use closed-form latency functions parameterized by config and operation shape.

Examples:

- DMA latency  
  `setup_cycles + ceil(bytes / bytes_per_cycle) + contention_penalty`

- MXU matmul latency  
  `pipeline_fill + steady_state_tiles + drain`

- vector op latency  
  `startup + ceil(elements / lanes)`

Keep these formulas in `timing/latency.py`, not distributed across unit implementations.

### Resource reservation

When an instruction issues:

- reserve the relevant resource(s)
- assign an operation ID
- schedule its completion event
- mark destination registers busy if needed

This gives you a natural scoreboard.

---

## In-Order Issue Semantics

Since the machine is in-order, the frontend should be simple:

- only the head instruction or head bundle is eligible to issue
- if it cannot issue because of operands, resources, or fences, the machine stalls
- younger instructions never bypass it

That keeps the model faithful and greatly simplifies debugging.

You can still allow **multiple units to overlap** because older issued operations remain in flight while the frontend waits or continues issuing subsequent instructions when legal.

### Bundle support

If your ISA has statically scheduled bundles or chimes, model them explicitly:

- bundle-level legality check
- per-slot resource reservation
- completion events per operation
- bundle retirement rules only if architecturally required

---

## Torch Backend Plan

Torch is the right choice for tensor semantics, but wrap it behind a backend interface so the simulator is not hard-coded to Torch APIs everywhere.

Suggested backend responsibilities:

- create tensors
- cast between dtypes
- perform matmul / conv / reduce / elementwise ops
- pack/unpack custom formats if needed
- apply saturation / rounding policies

Example:

```python
class TensorBackend(Protocol):
    def zeros(self, shape, dtype, layout): ...
    def load(self, storage, desc): ...
    def store(self, storage, value, desc): ...
    def matmul(self, a, b, *, acc_dtype, out_dtype): ...
    def elementwise(self, op, *args, out_dtype): ...
```

### Datatype guidance

Use native Torch dtypes where possible:

- `torch.int8`
- `torch.float16`
- `torch.bfloat16`
- `torch.float32`

For formats Torch does not represent natively, such as packed int4-style storage, use:

- packed storage in memory models
- unpack to a compute dtype on operand read
- repack on writeback if the architecture stores packed values

That avoids polluting the whole simulator with custom numeric code.

---

## Config-Class Design

Use nested dataclasses. Keep them immutable where practical.

Example shape:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MXUConfig:
    rows: int
    cols: int
    macs_per_cycle: int
    pipeline_depth: int
    accumulator_dtype: str

@dataclass(frozen=True)
class DMAConfig:
    num_engines: int
    bytes_per_cycle: int
    setup_cycles: int

@dataclass(frozen=True)
class ScratchpadConfig:
    capacity_bytes: int
    num_banks: int
    bank_width_bytes: int
    read_ports: int
    write_ports: int

@dataclass(frozen=True)
class AcceleratorConfig:
    frequency_hz: float
    issue_width: int
    mxu: MXUConfig
    dma: DMAConfig
    scratchpad: ScratchpadConfig
```

### Why this matters for DSE

It should be easy to run:

```python
from dataclasses import replace

cfg2 = replace(cfg, mxu=replace(cfg.mxu, rows=32, cols=32))
```

That lets you sweep design points without changing simulator code.

### Derived values

Do not hand-compute derived parameters everywhere. Put them on config helpers:

- bytes per tile
- peak ops/cycle
- scratchpad bank capacity
- tensor register capacity

---

## Recommended Development Phases

### Phase 1: ISA and config foundation

Deliverables:

- config dataclasses
- instruction definitions
- decoded program representation
- architectural state containers

Goal:

- enough structure to express programs and hardware variants cleanly

### Phase 2: Functional-only execution

Deliverables:

- sequential interpreter
- Torch-backed tensor ops
- memory load/store semantics
- correctness tests for each instruction

Goal:

- prove semantic correctness before adding timing complexity

Important constraint:

- keep the same instruction and state interfaces you will use in the timed model

### Phase 3: Tick-based timing engine

Deliverables:

- cycle counter
- event queue
- resource model
- scoreboard / hazard tracking
- in-order issue loop

Goal:

- convert the functional model into an execution-driven performance simulator without rewriting instruction semantics

### Phase 4: Memory realism

Deliverables:

- DMA engines
- banked scratchpad model
- bandwidth and contention
- fence/barrier support

Goal:

- make overlap and stalls realistic enough to guide architecture decisions

### Phase 5: Calibration and reporting

Deliverables:

- counters and traces
- timeline dumps
- kernel-level latency reports
- config sweep scripts

Goal:

- make the simulator useful for design reviews, not just correctness testing

### Phase 6: Optional extensions

- energy model driven by action counts
- trace export to pandas/parquet
- visualization of unit occupancy
- import from compiler-generated traces
- calibration against RTL or silicon data

---

## Validation Strategy

You will need three levels of validation.

### 1. Instruction-level correctness

For every instruction:

- construct minimal operand cases
- compare against a reference Torch result
- check saturation, clipping, layout, and corner-case behavior

### 2. Microarchitectural timing checks

Build tiny programs that isolate one effect:

- pure DMA transfer
- MXU only
- back-to-back dependent tensor ops
- bank conflict stress
- fence behavior
- compute/memory overlap

Each test should have an expected cycle count from hand calculation.

### 3. End-to-end kernel validation

Run representative ML kernels:

- GEMM
- convolution tiles
- elementwise fusion chains
- reduction-heavy kernels

Check:

- final outputs
- total cycles
- bandwidth usage
- dominant stall reasons

---

## What to Avoid

- Do not model every PE or multiply-accumulate as a separate event.
- Do not scatter timing formulas across instruction implementations.
- Do not let Torch tensors become the only representation of architectural state.
- Do not start with detailed DRAM timing.
- Do not build a giant monolithic `Simulator` class with all logic mixed together.

The right level is **operation-level timing with bulk tensor semantics**.

---

## First Implementation Milestone

The first milestone should support a minimal but representative path:

- scalar config objects
- a small instruction set
  - load tensor tile
  - store tensor tile
  - MXU matmul
  - vector elementwise op
  - fence / wait
- one DMA engine
- one MXU
- one scratchpad
- in-order issue
- cycle counting
- Torch-backed functional results

If that works, you already have the core architecture. Everything else is refinement.

---

## Concrete Next Steps

1. Rename `perf-modeling/` to `perf_modeling/` and create the package layout above.
2. Define the config dataclasses and derived-parameter helpers.
3. Define `ArchState`, `TensorValue`, and the memory abstractions.
4. Define the instruction interface as `validate -> plan -> commit`.
5. Implement a functional interpreter with Torch-backed kernels.
6. Add the tick loop, event queue, and resource reservations.
7. Add counters and a simple text trace for cycle-by-cycle debugging.
8. Build three microbenchmarks: DMA only, MXU only, DMA+MXU overlap.
9. Use those tests as the baseline before adding more units or more realistic memory behavior.

---

## Bottom Line

The best architecture for your project is:

- **Python dataclass config tree** for all hardware parameters
- **explicit architectural state objects**
- **Torch backend for tensor math and datatype behavior**
- **tick-based execution loop**
- **event-scheduled completion for long operations**
- **resource/timing layer separated from functional semantics**

That structure will scale from bring-up to design-space exploration without forcing a rewrite when the model becomes more realistic.
