# AGENTS.md

## Project Overview

This repository builds a combined functional and performance model for an ML-oriented hardware accelerator.

The intended simulator architecture is:

- execution-driven
- tick-based
- in-order issue
- event-scheduled completion for long-latency operations
- Torch-backed tensor semantics
- dataclass-based configuration for design-space exploration

Keep one simulator with a strict split between:

- architectural state
- resource and timing state
- tensor backend

Functional semantics define what happens. Timing and resource logic define when results become visible.

## Setup Commands

- Create or sync the environment: `uv sync`
- Run the default entrypoint: `uv run python main.py`
- Verify the package imports and syntax: `python3 -m compileall perf_modeling tests`

## Operating Cadence

- Write a progress log under `logs/` every 30 minutes while actively working.
- Also write a progress log whenever handing off substantial partial work to another thread.
- Each progress log should include:
  - current progress
  - current implementation status
  - current roadblocks
  - current questions
  - pending design decisions
  - next-step plan
- Use timestamped log filenames so progress is easy to follow chronologically.
- Make a git commit at least every 2 hours while working, or sooner when reaching a clear milestone.
- Milestone commits should summarize the functional change, not just “progress” or “WIP”.

## Dev Environment Tips

- Use the `perf_modeling/` package. Do not create or reintroduce `perf-modeling/`.
- Use the `logs/` directory for periodic progress logs.
- Keep module ownership stable:
  - `perf_modeling/config.py` for config dataclasses
  - `perf_modeling/state/` for architected machine state
  - `perf_modeling/units/` for unit-local resource models
  - `perf_modeling/timing/` for latency, scoreboards, banking, and reservations
  - `perf_modeling/isa/` for decoded instructions and planning semantics
  - `perf_modeling/backend/` for Torch-backed tensor operations
  - `perf_modeling/workloads/` for builders and kernel descriptors
  - `perf_modeling/engine.py` for the top-level simulator loop
- Preserve these boundaries when adding code. Do not collapse timing, state, and backend logic into one large class.
- Treat `notes/modeling.md` as the high-level design plan for the simulator.

## Code Style

- Treat Torch as a required dependency. Import it directly.
- Do not use `try/except ImportError` or optional-dependency fallback paths unless the user explicitly asks for them.
- Avoid wildcard argument patterns in public APIs:
  - no keyword-only `*` markers
  - no `*args`
  - no `**kwargs`
- Prefer explicit parameter lists, explicit types, and small typed containers.
- Use dataclasses for configuration objects and simple records.
- Keep config classes easy to copy with `dataclasses.replace`.
- For config dataclasses, follow each field with a one-line docstring.

Example:

```python
@dataclass(frozen=True)
class CoreConfig:
    num_scalar_registers: int = 32
    """Number of scalar registers"""
```

- Use `NotImplementedError` for intentional extension points in the scaffold.

## Architecture Conventions

- The top-level simulator is cycle-stepped.
- Long-latency operations complete through scheduled events.
- Architectural results become visible on completion, not on issue.
- The frontend is in-order:
  - only the head instruction or head bundle may issue
  - younger instructions do not bypass stalled older instructions
- Resource overlap is allowed through in-flight operations.

Preferred cycle flow:

1. Retire completion events for the current cycle.
2. Update unit and resource state.
3. Attempt to issue the next instruction or bundle.
4. Sample counters and tracing.
5. Advance the cycle.

Modeling rules:

- Use Torch for tensor allocation, dtype conversion, matrix multiply, elementwise ops, reductions, and layout-sensitive tensor transforms.
- Do not use Torch to represent simulator control flow, scoreboarding, event queues, or issue logic.
- Keep timing formulas centralized in `perf_modeling/timing/latency.py` or closely related timing modules.
- Do not scatter latency math across unrelated instruction or engine code.
- Model memory with explicit layers:
  - DRAM or host memory
  - scratchpad or SRAM
  - register or tensor file
- Start memory timing simple and deterministic before adding protocol-level detail.
- Do not model each PE or MAC as an individual event. Use operation-level timing with bulk tensor semantics.

## Testing Instructions

- After structural edits, run: `python3 -m compileall perf_modeling tests`
- As real behavior is added, validate at three levels:
  - instruction correctness
  - microarchitectural timing behavior
  - end-to-end kernel execution
- Add targeted tests when functionality lands. Do not leave behavior-only changes untested if a focused test is practical.

## Current Implementation Status

The repository currently contains:

- the simulator design plan in `notes/modeling.md`
- a scaffolded `perf_modeling/` package
- config classes, state containers, unit stubs, timing helpers, and a Torch backend
- intentional `NotImplementedError` placeholders for unfinished execution paths

Future threads should extend the scaffold rather than replace it.

## Recommended Next Milestones

- implement the first vertical slice:
  - load tensor tile
  - store tensor tile
  - MXU matmul
  - vector elementwise op
  - fence or wait
- wire issue-path planning into `perf_modeling/engine.py`
- add event scheduling and scoreboard updates
- add a few microbenchmarks and focused tests

## Change Discipline

- Preserve existing user decisions unless the user changes direction.
- Prefer small, composable additions over broad rewrites.
- Keep the scaffold importable after each step.
- If a new project convention is established in-thread, update this file so future threads inherit it.
