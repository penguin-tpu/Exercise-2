# RV32I Local Toolchain

This directory contains a minimal local RV32I build path that works in the current environment without a dedicated RISC-V linker.

## What It Does

- assembles self-contained RV32I assembly with `clang`
- extracts the assembled `.text` bytes from the relocatable ELF object
- wraps those bytes into a minimal executable ELF32 image

## Current Limits

- this path is intended for self-contained `.S` files
- it does not perform full ELF linking
- it rejects relocation sections, so cross-section and unresolved-symbol workflows are out of scope for now

## Example

```bash
python3 toolchains/riscv32/assemble_to_elf.py program.S --output build/program.elf
uv run python scripts/run_sim.py build/program.elf
```
