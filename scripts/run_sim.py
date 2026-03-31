"""CLI entry point for the RV32I functional and performance model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf_modeling import AcceleratorConfig, SimulatorEngine
from perf_modeling.decode import Decoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Execute a bare-metal RV32I binary in the simulator.")
    parser.add_argument("program", type=Path, nargs="?", help="Path to a raw binary or ELF32 image.")
    parser.add_argument(
        "--base-address",
        type=lambda value: int(value, 0),
        default=0x1000,
        help="Base address for raw binaries when ELF metadata is absent.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=100000,
        help="Maximum number of cycles to simulate before stopping.",
    )
    return parser.parse_args()


def build_default_program() -> bytes:
    """Return a tiny built-in RV32I smoke-test binary."""
    words = [
        0x02A00513,
        0x00000073,
    ]
    return b"".join(word.to_bytes(4, byteorder="little", signed=False) for word in words)


def main() -> None:
    """Load and run one RV32I program image."""
    args = parse_args()
    decoder = Decoder()
    if args.program is None:
        blob = build_default_program()
        program = decoder.decode_bytes(blob, base_address=args.base_address, name="builtin-smoke")
    else:
        blob = args.program.read_bytes()
        program = decoder.decode_bytes(blob, base_address=args.base_address, name=args.program.name)
    engine = SimulatorEngine(config=AcceleratorConfig(), program=program)
    stats = engine.run(max_cycles=args.max_cycles).snapshot()
    print(f"program={program.name}")
    print(f"halted={engine.state.halted} exit_code={engine.state.exit_code} trap={engine.state.trap_reason}")
    print(
        f"cycles={stats.get('cycles', 0)} issued={stats.get('instructions_issued', 0)} retired={stats.get('instructions_retired', 0)}"
    )


if __name__ == "__main__":
    main()
