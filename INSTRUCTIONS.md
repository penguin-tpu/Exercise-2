# Instructions

-[x] for RISC-V ISA tests, ensure that it's testing the complete RV32I base instruction set. We are still missing out a lot of instructions.
-[x] replace all `python` commands to use `uv`. e.g. `python XXX.py` -> `uv run python XXX.py` or `uv run XXX.py`
-[x] in tests/workload, provide an assembly file performing scalar integer matmul as example program.
-[x] in scripts/run_sim.py, add support such that we can pass in result dump path with folders. Currently it errors out: FileNotFoundError: [Errno 2] No such file or directory: 'out/trace.json'
-[x] For run_sim.py, modify the behavior of `parser.add_argument("program", type=Path, nargs="?", help="Path to a raw binary or ELF32 image.")` such that it can take in either elf or assembly, detected by the file type extension. If an assembly file is passed in, invoke the compiler pipeline to convert it to ELF and then run it.
-[x] also, by default, generate all program output under out/ folder.
-[x] what is test_rv32ui_rewritten.py? Could you consolidate all the ISA test into a single test script?
