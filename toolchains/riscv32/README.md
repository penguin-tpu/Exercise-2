# RV32I GNU Toolchain

This directory contains the repo-local RV32I build path used by the simulator tests and scripts.

## What It Does

- resolves GNU RISC-V binutils from `toolchains/riscv32/gnu-binutils/` or the host `PATH`
- bootstraps `binutils-riscv64-unknown-elf` and `gcc-riscv64-unknown-elf` under `toolchains/riscv32/` without root when the tools are missing
- compiles and links freestanding RV32I plus machine-CSR assembly/C sources with `riscv64-unknown-elf-gcc`
  using `-ffreestanding -nostdlib -nostartfiles -nodefaultlibs -static`
- keeps the lower-level GNU `as`, `ld`, and `objcopy` tools available under the same repo-local prefix

## Current Limits

- this path is intended for freestanding RV32I assembly and C programs
- it assumes an `_start` symbol as the program entry point
- it intentionally does not link a hosted libc, startup files, or printf-style runtime support

## Example

```bash
uv run python toolchains/riscv32/bootstrap_gnu_binutils.py
uv run python toolchains/riscv32/assemble_to_elf.py program.S --output build/program.elf
uv run python toolchains/riscv32/assemble_to_elf.py program.c --output build/program.elf
uv run python scripts/run_sim.py build/program.elf
```
