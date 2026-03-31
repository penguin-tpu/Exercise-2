# Open Questions

- `INSTRUCTIONS.md` asks to set up the toolchain using `riscv-gnu-toolchain`. I currently implemented a repo-local GNU RISC-V toolchain by bootstrapping packaged `binutils-riscv64-unknown-elf` and `gcc-riscv64-unknown-elf` under `toolchains/riscv32/`. Is that acceptable for this repository, or do you specifically want a source-built checkout of `riscv-collab/riscv-gnu-toolchain` under `toolchains/`?
