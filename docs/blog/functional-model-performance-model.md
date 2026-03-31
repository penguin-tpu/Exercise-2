# Functional Models, Performance Models, and Why We Are Building a Combined Execution-Driven Simulator

In most accelerator projects, the first serious modeling question is not “what is the peak TOPS number?” It is whether the architecture has an executable contract, whether software can target it without ambiguity, and whether the inevitable performance gaps can be explained in terms of compute, movement, and synchronization rather than wishful averages. In our case, that question is even sharper because the design spec already treats the model as part of the architectural definition: the ISA section explicitly says readers should refer to the functional and performance model for instruction definitions, and the current plan recommends a single execution-driven, tick-based simulator with a strict separation between architectural state, resource/timing state, and the tensor compute backend.

The right mental model is simple but important: a **functional model** answers **what must happen**, while a **performance model** answers **when it happens** and **why it took that long**. Mature public stacks make this separation very explicit. Gemmini exposes a fast functional ISA path through Spike and a slower cycle-accurate path through Verilator/VCS; AMD’s AI Engine flow similarly distinguishes the `x86simulator`, which is useful for debugging, from `aiesimulator`, which is the one you must run for performance information; and NVDLA offers both RTL and a much faster TLM/SystemC virtual platform for software bring-up and integration.

A functional model, then, is best understood as an **executable architectural specification**. It should implement the architecturally visible state transitions of the machine: PC movement, register writes, memory effects, predicates, fences, and instruction legality. Gemmini’s documentation is very clear that Spike is faster but “can only verify functional correctness” and cannot provide accurate performance metrics, while AMD’s x86 simulator is explicitly framed as a debug vehicle whose behavior may not match the AI Engine simulator cycle-for-cycle. That is exactly the point: a functional model is the ground truth for semantics, not for latency.

A performance model is different in kind, not just in speed. It needs to represent queues, arbitration, bandwidth ceilings, overlap opportunities, bank conflicts, and completion timing. NVDLA’s virtual platform is register-accurate and intended for software development and debug in a full-system environment; AMD’s `aiesimulator` models global memory and the NoC in addition to the array; and Arm is unusually explicit that Ethos-U Vela’s performance estimates are **not cycle-accurate** and are meant to guide compiler optimizations rather than stand in for measured execution. Those examples are useful because they separate three classes of tools that are too often conflated: functional simulators, performance simulators, and estimators. [^](https://nvdla.org/vp.html)

That distinction matters because our accelerator is not merely “a GEMM engine with a rough bandwidth model.” The spec describes a statically scheduled instruction architecture, deterministic arithmetic-unit latencies, long-chime operations, asynchronous DMA, fence/wait semantics, a banked matrix register file, and two different MXU organizations—a systolic array and a parallel inner-product tree. It also states that memory traffic beyond the local block is inherently variable and must be synchronized in software, and it calls out bank-conflict implications explicitly. This is exactly the profile of a machine whose behavior is defined by instruction semantics and resource interactions, not just by loop mappings.

## Why a mapping-only model is not enough

Analytical mapping frameworks are still valuable, but they solve a different problem. Timeloop’s own documentation says its core value is a fast analytical model plus a mapper over tensor-algebra mappings, and its mapping reports revolve around loop tiling, loop permutation, and spatial execution. MAESTRO and ZigZag occupy the same neighborhood: they evaluate hardware, workload, and mapping together, and they are extremely useful when the main question is how dataflow and tiling shape reuse, traffic, and utilization. [^](https://timeloop.csail.mit.edu/v4/output-formats/mapping), [^](https://maestro.ece.gatech.edu/docs/build/html/tutorials/micro2020.html), [^](https://github.com/KULeuven-MICAS/zigzag), [^](https://github.com/maestro-project/maestro)

That is not the dominant uncertainty here. Our design has explicit DMA issue/wait behavior, architecturally visible barriers, register banking constraints, and long-latency instructions that overlap in time. Those are not naturally captured by a mapper, because they are properties of **instruction issue, completion, and shared resources** rather than of loop nests alone. Mapping-first tools can still be excellent submodels—for example, to study MXU tiling or to sanity-check register working sets—but they are not the right top-level abstraction for a statically scheduled ISA machine with asynchronous off-chip traffic.

Recent NPU simulators make exactly this point from the opposite direction. ONNXim argues that tile compute inside a scratchpad-centric NPU core is often deterministic enough to abstract, while DMA dependencies, DRAM, and NoC contention still need cycle-level treatment. PyTorchSim pushes further by compiling through a custom RISC-V-based ISA and using both instruction-level and tile-level simulation modes to keep accuracy while controlling runtime. Those are strong public confirmations that once the architecture grows explicit orchestration machinery, the simulator must expose it instead of burying everything under a matrix kernel oracle. [^](https://arxiv.org/abs/2406.08051)

The same lesson appears in hardware papers closer to our domain. The Adelia accelerator emphasizes matched external-memory/compute bandwidth, direct streaming into compute engines, and dynamic switching between context mode and batch mode depending on runtime service objectives. In other words, performance is not just “how fast is matmul”; it is “how well does the whole machine coordinate movement, reuse, and scheduling.” That is precisely why the model must be execution-driven.

## What we are building

The plan we should follow is to build one execution-driven, tick-based model with a strict split between three concerns: architectural state, resource/timing state, and a tensor compute backend. Put differently, instruction semantics decide what result must happen, while timing and resources decide when the result becomes visible. That is the core design choice in the current plan, and it is the right one.

The **architectural state** should be explicit and boring in the best possible way: PC, scalar registers, predicate/status state, matrix registers or tensor descriptors, visible memories, outstanding fence tags, and program termination state. If an ISA-visible behavior exists, it belongs here. The model should never hide architected state inside incidental simulator objects or temporary tensor handles. That is the discipline that turns a simulator into an executable specification instead of a pile of heuristics.

The **resource/timing state** is separate. It should track unit occupancy, DMA progress, queue state, bank reservations, bandwidth consumption, scoreboard bits, and completion events. This is where stalls live. This is also where the explanation for latency lives: if an instruction was delayed, the model should be able to say whether it was waiting on source readiness, DMA completion, bank conflicts, execution-unit occupancy, or front-end ordering. ONNXim’s split between abstracted compute and cycle-level DRAM/NoC, along with PyTorchSim’s combination of ISA-level fidelity and faster tile-level timing, are strong public examples of this layered approach.

The **tensor compute backend** should be used for bulk semantics, not for simulator control flow. The current plan’s recommendation—use Torch for tensor math and data movement, but keep the simulator as ordinary Python state machines and data structures—is exactly right. Modern framework-facing simulators such as OpenReg and PyTorchSim reinforce the value of a clean backend boundary: OpenReg is intentionally a standalone, self-contained simulation backend for PyTorch integration, while PyTorchSim uses PyTorch 2 as the front-end context but keeps the NPU model itself explicit and architectural.

## Why these design choices are the right ones

### Keep one semantic source of truth

The easiest way to ruin a modeling effort is to build two independent simulators too early: one for correctness and one for cycles. They begin identical in intent and diverge the first time someone fixes a fence corner case in one path and forgets the other. The better rule is: one semantic source of truth, multiple fidelity modes if needed. Gemmini, AMD AI Engine, and NVDLA all expose different simulation modes to users, but that does not mean their architectural semantics should fork internally. Our model should keep a single definition of instruction behavior even if we later expose a faster functional mode and a richer timed mode on top of it.

### Tick-based outside, event-driven inside

The outer loop should advance cycle by cycle because stalls, fences, and overlap are easier to reason about at cycle granularity. The inner mechanics should still be event-driven, because re-executing long tensor operations every cycle is both slow and conceptually wrong. ONNXim’s design is the public argument for this compromise: compute can often be summarized at tile granularity, but dependencies and contention still need event-level timing. PyTorchSim’s Tile-Level Simulation makes the same point with a more elaborate software stack.

### Use Torch for tensor semantics, not simulator control

Torch is excellent at implementing dense tensor semantics, dtype conversions, and vectorized data movement. It is not the right place to encode queues, fences, dispatch eligibility, or bank arbitration. If those concerns get folded into backend kernels, the simulator becomes opaque precisely where architects need visibility. OpenReg is a useful supporting example here: its value is not as a full performance backend, but as a minimal reference that cleanly separates accelerator integration mechanics from the framework around it.

### Make configuration first-class

All hardware parameters that affect legality or timing should live in a config tree rather than being scattered across unit implementations. Timeloop, MAESTRO, ZigZag, VTA, Gemmini, and NVDLA all derive much of their power from parameterization. The exact shape of the parameterization differs, but the principle is consistent: architecture exploration only works when the hardware description is a first-class input to the model rather than a pile of hidden constants. [^](https://tvm.apache.org/2018/07/12/vta-release-announcement)

### How the simulator should work

At the top level, the simulator should behave like an in-order issuing front-end with asynchronous completion. Each cycle, it retires any events that complete on that cycle, updates scoreboards and busy flags, checks whether the head instruction is legal to issue, reserves resources for any newly issued work, schedules completion events, samples statistics, and advances time. That matches both the spirit of the plan and the structure of the spec: instructions are statically scheduled, memory stalls block further dispatch when synchronization has not completed, and already issued work continues draining in flight.

A practical outer loop looks like this:

```python
while not sim.done():
    sim.retire_ready_events()
    sim.update_scoreboards()
    sim.try_issue_head_instruction()
    sim.sample_stats()
    sim.cycle += 1
```

Long-latency instructions should not mutate architected state at issue time unless the ISA explicitly requires it. Instead, they should schedule a completion that performs the visible writeback when the modeled latency expires. A DMA load updates the destination register file or scratchpad when the transfer completes; an MXU operation writes its result when compute completes; and a wait or fence instruction retires only when the tracked operations it depends on have completed. This is the cleanest way to keep semantics and timing aligned.

Instruction modeling should therefore follow a three-part discipline: **validate**, **plan**, and **commit**. Validation checks legality and operand availability. Planning computes resource reservations, latency, traffic, and any completion dependencies. Commit applies the architecturally visible result. Keeping those phases explicit is much healthier than allowing every unit to mutate state ad hoc inside a `tick()` method. It also makes it easier to produce meaningful traces and later attach calibration hooks.

The memory model deserves unusual care because it will dominate realism. At minimum it should distinguish host/DRAM memory, on-chip SRAM or scratchpads, and register/tensor files. It should model bank conflicts, read/write port limits, burst or beat granularity, DMA setup cost, and transfer-size-dependent latency. The spec already tells us why: the matrix register file is banked in a way that creates software-visible exclusion patterns, and off-block DMA latency is explicitly nondeterministic and must be synchronized in software. Those are first-order architectural effects, not second-order cleanup.

The timing model should stay formula-based and centralized. DMA latency belongs in a latency model, not hard-coded across every instruction that invokes DMA. MXU latency should come from array parameters, tile shapes, and pipeline depths; vector latency should come from startup plus lane throughput; bank conflicts and unit occupancy should be resource checks, not magic penalties appended later. This is the difference between a model that can survive design-space exploration and one that collapses under the second configuration sweep.

## Where other modeling styles still help

None of this means other tools become irrelevant. It means they get used at the right layer.

Timeloop, MAESTRO, and ZigZag remain excellent for mapping-space reasoning. They formalize tiling, permutation, spatial partitioning, reuse, and hardware-workload coupling better than most bespoke simulators do. If the question is “what tiling and placement should this MXU favor?” those tools are often better than a full ISA simulator.

SCALE-Sim is valuable as a kernel-level submodel when the question is specifically about regular systolic-array behavior. Its documentation explicitly describes it as a simulator for systolic-array accelerators for convolution, feed-forward, and GEMM-like layers, and its Python implementation makes it approachable for rapid experimentation. That makes it a good pattern for local array timing or trace generation, even if it is not the right top-level model for an ISA-driven machine.

VTA, Gemmini, NVDLA, ONNXim, and PyTorchSim are the more directly relevant family for our project. VTA exposes a RISC-like tensor interface with DMA and explicit compute/memory arbitration. Gemmini couples a programmable ISA, banked local memories, DMA, and both functional and timed flows. NVDLA shows how a register-accurate virtual platform fits into software bring-up and system integration. ONNXim and PyTorchSim show how to keep timing realism while refusing to simulate every multiply as a first-class event. That is the lineage our model belongs to.

Accelergy is the right complement on the energy side. Its methodology explicitly takes an architecture description plus runtime action counts, and those action counts are intended to come from performance models such as cycle-level simulators or Timeloop. That is exactly why our performance model should emit action counts rather than embedding a fragile energy model into every unit.

## A concrete implementation sketch

A clean package structure should make the state/resource/backend split obvious:

```bash
perf_modeling/
  config.py
  engine.py
  events.py
  program.py
  decode.py
  stats.py
  trace.py
  isa/
    instruction.py
    semantics.py
  state/
    arch_state.py
    register_file.py
    tensor_file.py
    memory.py
  units/
    dma.py
    mxu.py
    vector.py
    scalar.py
  timing/
    latency.py
    scoreboard.py
    resources.py
    banking.py
  backend/
    torch_backend.py
    quant.py
```

The important architectural rule is that `state/` owns architected data, `timing/` owns resource and latency policy, `units/` own unit-local behavior, and `backend/` owns bulk numeric semantics. That package boundary is not cosmetic; it is what keeps the model debuggable when the first real workload arrives.

A representative instruction interface should look something like this:

```python
class Instruction:
    def validate(self, state, config): ...
    def plan(self, state, timing, backend): ...
    # returns an ExecutionPlan

class ExecutionPlan:
    resources: list
    completion_cycle: int
    on_complete: callable
    stats: dict
```

This makes architectural intent explicit. The instruction knows how to prove it is legal, how to request resources, and how to commit the visible result. The engine knows how to order time. The latency model knows how long things take. That is the right separation of concerns for a statically scheduled accelerator simulator.

## What success looks like

A good model for this project should answer three questions simultaneously:

1. Is the program architecturally correct?
2. How many cycles did it take under this hardware configuration?
3. Why did it take those cycles?

The third question is the one weak models fail. A total cycle count with no explanation is not architecture guidance; it is a number. The simulator should therefore emit instruction traces, unit occupancy, bytes moved, bank-conflict counts, queue occupancy histories, and stall-reason breakdowns, plus action counts for later energy estimation. That expectation is already embedded in the current plan, and it aligns with how analytical and hybrid frameworks interface with energy estimation tools such as Accelergy. [^](https://accelergy.mit.edu/paper.pdf)

## The practical roadmap

The development order should follow dependency, not ambition. First define the config tree, decoded program representation, and architectural state. Then implement a functional interpreter with the same instruction/state interfaces that the timed model will reuse. After that, add the cycle loop, event queue, scoreboards, and resource reservations. Only then should memory realism, bank conflicts, and richer overlap behavior be layered in. Calibration and reporting come after the semantics and timing machinery are stable, not before. That staged plan is already captured in the current implementation notes, and it is the safest way to grow from “correct” to “useful.”

## Closing

The central idea is straightforward: for this accelerator, the functional model is not just a software convenience, and the performance model is not just a spreadsheet with better branding. The functional model is the executable contract of the ISA and visible state. The performance model is the explanation engine for overlap, contention, and latency. Because our machine is statically scheduled, banked, DMA-driven, and synchronization-sensitive, the right top-level abstraction is a combined execution-driven simulator with one semantic source of truth and a separate timing/resource layer. That is the model architecture most consistent with the spec, with the current plan, and with the best public examples from research and industry.
