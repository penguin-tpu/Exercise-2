# Instructions

-[x] for RISC-V ISA tests, ensure that it's testing the complete RV32I base instruction set. We are still missing out a lot of instructions.
-[x] replace all `python` commands to use `uv`. e.g. `python XXX.py` -> `uv run python XXX.py` or `uv run XXX.py`
-[] in tests/workload, provide an assembly file performing scalar integer matmul as example program.
-[] in scripts/run_sim.py, add support such that we can pass in result dump path with folders. Currently it errors out: FileNotFoundError: [Errno 2] No such file or directory: 'out/trace.json'