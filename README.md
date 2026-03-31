# Architect

Combined functional and performance model for an ML-oriented accelerator.

Reference docs:

- `docs/README.md` for the documentation index
- `docs/specs/architecture_specification.md` for the programmer-visible machine model
- `docs/specs/microarchitecture_specification.md` for the timing and execution model
- `logs/README.md` for interval summaries and recent progress logs

The simulator is:

- execution-driven
- cycle-stepped
- in-order at the frontend
- event-scheduled for long-latency completion
- split across architectural state, timing/resource state, and backend tensor semantics

## Setup

Create or sync the environment:

```bash
uv sync
```

Verify imports and syntax:

```bash
uv run python -m compileall perf_modeling tests scripts
```

Run the built-in smoke program:

```bash
uv run python scripts/run_sim.py
```

## Running Programs

`scripts/run_sim.py` accepts:

- raw binaries
- ELF32 images
- assembly sources such as `.S` or `.s`

Assembly inputs are assembled to a transient bare-metal ELF automatically.

Run an existing assembly workload example:

```bash
uv run python scripts/run_sim.py tests/workload/scalar_int_matmul.S
```

Run an ELF directly:

```bash
uv run python scripts/run_sim.py path/to/program.elf
```

Useful common options:

- `--max-cycles 100000`
- `--config baseline`
- `--list-configs`
- `--sweep-config baseline --sweep-config tiny_debug`
- `--sweep-json sweep.json`
- `--sweep-csv sweep.csv`
- `--sweep-sort cycles --sweep-desc`
- `--sweep-limit 3`
- `--sweep-manifest-json sweep.json`
- `--sweep-report summary --sweep-report delta`
- `--experiment-json experiment.json`
- `--output-dir out`
- `--stats-json stats.json`
- `--trace-json trace.json`
- `--stats-csv stats.csv`
- `--trace-csv trace.csv`
- `--manifest-json manifest.json`
- `--scratchpad-dump results.bin`
- `--dram-dump dram.bin`

Example with artifacts:

```bash
uv run python scripts/run_sim.py tests/workload/scalar_int_matmul.S \
  --output-dir out/matmul \
  --stats-json stats.json \
  --trace-json trace.json \
  --manifest-json manifest.json \
  --scratchpad-dump results.bin \
  --scratchpad-dump-size 16
```

Relative artifact paths are rooted under `out/` by default.

List the packaged hardware presets:

```bash
uv run python scripts/run_sim.py --list-configs
```

Sweep one workload across multiple named presets:

```bash
uv run python scripts/run_sim.py tests/workload/scalar_int_matmul.S \
  --sweep-config baseline \
  --sweep-config tiny_debug \
  --sweep-sort cycles \
  --sweep-desc \
  --sweep-limit 2 \
  --sweep-report summary \
  --sweep-json sweep.json \
  --sweep-csv sweep.csv
```

You can also package a grouped sweep experiment into one manifest:

```json
{
  "program": "tests/workload/scalar_int_matmul.S",
  "sweep_configs": ["baseline", "tiny_debug"],
  "sweep_sort": "cycles",
  "sweep_desc": true,
  "sweep_limit": 2,
  "artifacts": {
    "output_dir": "artifacts",
    "sweep_json": "sweep-results.json",
    "sweep_csv": "sweep-results.csv"
  }
}
```

```bash
uv run python scripts/run_sim.py --sweep-manifest-json sweep.json
```

The same grouped manifest style can drive a single run too:

```json
{
  "program": "tests/workload/scalar_int_matmul.S",
  "config": "tiny_debug",
  "max_cycles": 100000,
  "artifacts": {
    "output_dir": "artifacts",
    "stats_json": "stats.json",
    "trace_json": "trace.json",
    "manifest_json": "run-manifest.json"
  }
}
```

```bash
uv run python scripts/run_sim.py --experiment-json experiment.json
```

## Reports

The CLI can print curated report views from the flat stats surface:

- `summary`
- `config`
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

Example:

```bash
uv run python scripts/run_sim.py tests/workload/scalar_int_matmul.S \
  --report config \
  --report summary \
  --report pipeline \
  --report fetch \
  --report units
```

Structured JSON outputs include the selected config name and a recursive config snapshot so runs can be reproduced without reopening the source preset definitions.

You can also filter and limit multi-row reports:

```bash
uv run python scripts/run_sim.py \
  --report latency \
  --report-limit 5 \
  --report-match add
```

## Memory Preloads

The CLI supports preloading DRAM or scratchpad before execution:

```bash
uv run python scripts/run_sim.py program.S \
  --dram-load 0x100:data/input.bin \
  --scratchpad-load 0x0:data/tile.bin
```

Or load both from one manifest:

```json
{
  "dram": [
    { "address": "0x100", "path": "dram_input.bin" }
  ],
  "scratchpad": [
    { "offset": "0x0", "path": "tile.bin" }
  ]
}
```

```bash
uv run python scripts/run_sim.py program.S --memory-loads-json data/loads.json
```

## Tests

Run the full test suite:

```bash
uv run pytest tests -q
```

Focused areas covered today include:

- complete RV32I base-instruction execution coverage
- machine CSR and trap handling
- DMA/vector/tensor vertical slices
- CLI artifact generation and reporting
- microarchitectural timing behavior such as dependency, memory, and fetch stalls

## Repo Structure

- `perf_modeling/`: simulator package
- `scripts/run_sim.py`: CLI entrypoint
- `tests/`: RV32I, timing, workload, and CLI regressions
- `tests/workload/`: runnable assembly examples
- `toolchains/riscv32/`: local GNU RISC-V assembly wrapper
- `docs/`: architecture, microarchitecture, and design documents
- `logs/`: timestamped progress logs
- `notes/modeling.md`: high-level design plan
