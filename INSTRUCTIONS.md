# Instructions

-[x] for RISC-V ISA tests, ensure that it's testing the complete RV32I base instruction set. We are still missing out a lot of instructions.
-[x] replace all `python` commands to use `uv`. e.g. `python XXX.py` -> `uv run python XXX.py` or `uv run XXX.py`
-[] in tests/workload, add a few other vrey representative and typical example workload.
-[x] update README to document example usage of this repo
-[x] try to compile the logs in logs/, make it one file per ~2 hour interval.
-[x] try to simplify scripts/run_sim.py. Try to move the helper functions into proper locations inside the perf_modeling package.
-[] add a config/ subpackage in perf_modeling. It should contain a bunch of example configurations for the hardware, so scripts/run_sim.py can select one of the config to run
-[] add a specification document in docs/specs/architecture_specification.md documenting the current architecture design.
-[] add a specification document in docs/specs/microarchitecture_specification.md documenting the current microarchitecture design.
