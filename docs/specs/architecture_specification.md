# Architecture Specification

## Purpose

This document describes the current architected behavior of the combined functional and performance model in this repository. It focuses on the programmer-visible machine model rather than the internal timing implementation.

## Scope

The current simulator is a bare-metal, execution-driven machine model with:

- complete RV32I base-instruction execution coverage
- machine-mode CSR and trap support sufficient for current tests and workloads
- a memory-mapped scratchpad
- placeholder accelerator-side tensor/vector or DMA operations used by synthetic programs

The simulator is not yet a full privileged RISC-V platform and does not currently model an operating system environment.

## Architectural Model

The simulator exposes one architectural machine with these major state elements:

- 32 scalar architectural registers, each 32 bits wide
- a tensor register file used by accelerator-side placeholder operations
- a machine CSR file
- one byte-addressable DRAM backing store
- one byte-addressable scratchpad SRAM window
- a scalar program counter
- architectural halt, trap, exit-code, and retired-instruction state

Architectural results become visible on operation completion, not at issue time.

## Program Formats

The simulator accepts three program forms:

- raw RV32I binaries
- ELF32 little-endian RISC-V images
- assembly sources, assembled through the local toolchain wrapper before execution

The default CLI entrypoint is:

```bash
uv run python scripts/run_sim.py
```

## Memory Map

The machine uses a flat DRAM space plus a fixed memory-mapped scratchpad window.

Default machine parameters:

- reset PC: `0x00001000`
- default trap vector: `0x00000080`
- scratchpad base address: `0x20000000`

Addresses below the scratchpad base map to DRAM. Addresses in the scratchpad aperture map to the local scratchpad.

## Supported RV32I Instruction Surface

The simulator currently executes the complete RV32I base instruction set:

- U-type: `lui`, `auipc`
- jumps: `jal`, `jalr`
- branches: `beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`
- loads: `lb`, `lh`, `lw`, `lbu`, `lhu`
- stores: `sb`, `sh`, `sw`
- immediate ALU: `addi`, `slti`, `sltiu`, `xori`, `ori`, `andi`, `slli`, `srli`, `srai`
- register ALU: `add`, `sub`, `sll`, `slt`, `sltu`, `xor`, `srl`, `sra`, `or`, `and`
- ordering/system: `fence`, `ecall`, `ebreak`

The handwritten ISA corpus under `tests/isa/` and the consolidated coverage in `tests/test_rv32i.py` validate that complete RV32I base set.

## Machine-Level Extensions

The current machine-mode slice includes:

- CSR reads and writes
- machine trap delivery
- machine trap return through `mret`
- machine counters such as `cycle` and `instret`
- configurable trap-handler vectoring through `mtvec`

Supported CSR instructions:

- `csrrw`, `csrrs`, `csrrc`
- `csrrwi`, `csrrsi`, `csrrci`

Supported machine-state behavior includes population of `mepc`, `mcause`, and `mtval` on delivered traps.

## Accelerator-Side Placeholder Operations

The simulator also supports synthetic accelerator-oriented instructions used by builders and tests:

- `tload`
- `tstore`
- `vadd`
- `matmul`
- `dma_copy`

These are not currently decoded from a public binary ISA encoding. They are generated through the simulator’s builder paths and execute through the same event-scheduled engine as RV32I instructions.

## Synchronization and Visibility Rules

The frontend is in-order:

- only the head instruction may issue
- younger instructions never bypass older stalled instructions

Architectural visibility rules:

- register results become visible on completion
- DMA results become visible when the DMA completion event fires
- tensor/vector results become visible on completion
- `fence` waits for all tracked prior outstanding operations

Control-flow instructions stall fetch until the control operation resolves.

## Configuration Surface

The architectural machine is parameterized by `AcceleratorConfig` and supporting dataclasses under `perf_modeling.config`.

Packaged named presets currently include:

- `baseline`
- `tiny_debug`
- `balanced_ml`
- `throughput_ml`

The CLI selects these with:

```bash
uv run python scripts/run_sim.py --config baseline
```

## CLI Artifacts

The simulator can emit:

- flat stats JSON and CSV
- trace JSON and CSV
- a run manifest
- scratchpad dumps
- DRAM dumps

The CLI can also preload DRAM and scratchpad contents before execution.

## Representative Workloads

Representative runnable example workloads currently include:

- `tests/workload/scalar_int_matmul.S`
- `tests/workload/scalar_vector_add.S`
- `tests/workload/scalar_reduce_sum.S`
- `tests/workload/scalar_relu.S`

These exercise scalar compute, loops, memory access, and elementwise control flow in a fully executable bare-metal form.

## Known Architectural Limits

The current architecture model intentionally does not yet provide:

- a full privileged-RISC-V environment
- virtual memory
- interrupts
- a public encoded accelerator ISA frontend
- operating-system services or syscall ABI modeling beyond bare-metal halt or trap behavior

## Source of Truth

The implementation described here is primarily realized in:

- `perf_modeling/config/`
- `perf_modeling/state/`
- `perf_modeling/decode.py`
- `perf_modeling/isa/`
- `scripts/run_sim.py`

For timing and internal resource behavior, see `docs/specs/microarchitecture_specification.md`.
