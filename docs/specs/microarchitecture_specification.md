# Microarchitecture Specification

## Purpose

This document describes the current internal execution and timing model used by the simulator. It focuses on how the machine advances cycle by cycle, how resources are reserved, and how long-latency work completes.

## Top-Level Model

The simulator is:

- execution-driven
- cycle-stepped at the outer loop
- in-order at the frontend
- event-scheduled for long-latency completion

It keeps a strict split between:

- architectural state
- resource and timing state
- tensor backend behavior

## Cycle Flow

Each architectural cycle executes this high-level sequence:

1. retire all completion events ready for the current cycle
2. update unit-local resource state
3. attempt to issue the next head instruction
4. sample stats and trace data
5. advance the cycle counter

This behavior is implemented in `perf_modeling/engine.py`.

## Frontend and Issue Rules

The issue model is strictly in-order:

- there is one head instruction at a time
- if the head instruction stalls, all younger instructions wait
- control-flow instructions stall fetch until their completion redirects the PC

The issue path checks:

- scalar RAW and WAW hazards
- tensor RAW and WAW hazards
- CSR RAW and WAW hazards
- fence completion conditions
- unit queue availability
- shared-resource reservations

## Completion Model

Long-latency operations are represented as completion events in a min-heap event queue keyed by `ready_cycle`.

Each scheduled event carries:

- completion cycle
- operation ID
- completion callback
- description string

Architectural mutation occurs in completion callbacks, not at issue time.

## Functional Units

The current simulator instantiates these execution resources:

- scalar unit
- vector unit
- MXU or matrix unit
- DMA unit
- load-store unit

Each unit tracks occupancy and queue depth through its local status plus the shared simulator stats surface.

## Scoreboards and Reservations

The simulator uses scoreboards for:

- scalar registers
- tensor registers
- CSRs
- shared resources

Shared-resource reservations currently cover:

- DRAM transfer or access intervals
- scratchpad bank reservations
- scratchpad read-port reservations
- scratchpad write-port reservations

Reservations are interval-based rather than only “busy until cycle N,” which allows overlapping resource use when intervals do not conflict.

## Memory Hierarchy Model

The memory model exposes:

- off-chip DRAM or host memory
- on-chip scratchpad SRAM
- scalar and tensor architectural registers

Scratchpad is explicitly banked and ported. The timing model accounts for:

- bank footprint
- bank conflicts
- read-port conflicts
- write-port conflicts

DMA pressure is modeled in phases so setup latency and transfer occupancy are distinct.

## Latency Formulas

Central latency formulas live in `perf_modeling/timing/latency.py`.

Current formulas:

- scalar latency: `max(1, pipeline_depth)`
- vector latency: `pipeline_depth + ceil(elements / lanes)`
- MXU latency: `pipeline_depth + max(0, tiles - 1)`
- DMA latency: `setup_cycles + ceil(bytes / bytes_per_cycle)`
- scratchpad access latency: scaled by banks touched and bank width
- DRAM read or write latency: base latency plus transfer time from bandwidth

## Control-Path Timing

The frontend now tracks fetch stalls caused by unresolved control instructions.

Current counters include:

- `fetch_stall_cycles`
- `fetch_stall.branch_cycles`
- `fetch_stall.jump_cycles`
- `fetch_stall.system_cycles`

These counters are sampled cycle by cycle while fetch is blocked.

## Contention and Stall Accounting

The simulator records explicit stall counters such as:

- scalar dependency stalls
- tensor dependency stalls
- CSR dependency stalls
- fence stalls
- unit busy stalls
- DRAM contention stalls
- scratchpad bank and port contention stalls

The stats surface also includes:

- aggregate contention families
- per-resource contention keys
- queue occupancy histograms
- event-queue occupancy histograms
- per-opcode latency totals and maxima
- bytes moved by memory layer

## Trace Model

Tracing is controlled by policy flags under `TraceConfig` and `TimingConfig`.

The current trace surface can retain:

- issue records
- completion records
- trap records
- cycle-level stall records

Trace retention is separately configurable for cycle-level and event-level visibility.

## CLI Reporting Surface

The microarchitectural stats are exposed through curated report families:

- `summary`
- `latency`
- `occupancy`
- `events`
- `fetch`
- `memory`
- `contention`
- `stalls`
- `pipeline`
- `units`
- `isa`

These are implemented in `perf_modeling/reporting.py` and consumed by `scripts/run_sim.py`.

## Preset Configurations

The simulator now includes named hardware presets in `perf_modeling/config/presets.py`.

These presets vary:

- scalar pipeline depth
- vector width and queue depth
- MXU shape and throughput
- DMA width and outstanding transfer depth
- scratchpad capacity, banks, and ports
- DRAM latency and bandwidth

They are intended for repeatable experiments and CLI selection rather than as a final benchmark suite.

## Known Microarchitectural Limits

The current timing model intentionally does not yet include:

- out-of-order issue
- speculative execution
- cache hierarchies
- coherence protocols
- interrupt timing
- detailed PE-by-PE or MAC-by-MAC tensor-array modeling

The model stays at operation granularity for tensor and DMA work, which matches the project’s current design goal of combined correctness plus parameterized performance estimation.

## Source of Truth

The implementation described here is primarily realized in:

- `perf_modeling/engine.py`
- `perf_modeling/events.py`
- `perf_modeling/timing/`
- `perf_modeling/stats.py`
- `perf_modeling/reporting.py`
- `perf_modeling/units/`

For the programmer-visible machine model, see `docs/specs/architecture_specification.md`.
