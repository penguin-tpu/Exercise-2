"""CLI entry point for the RV32I functional and performance model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf_modeling import AcceleratorConfig, SimulatorEngine
from perf_modeling.decode import Decoder


def _format_average(total_cycles: int, samples: int) -> str:
    """Format one average-latency value for human-readable reports."""
    if samples <= 0:
        return "0.00"
    return f"{total_cycles / samples:.2f}"


def emit_report(report_name: str, stats: dict[str, int]) -> None:
    """Print one curated report from the flattened stats snapshot."""
    if report_name == "latency":
        sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
        for key in sample_keys:
            opcode = key.removeprefix("latency.").removesuffix(".samples")
            samples = stats[key]
            total_cycles = stats.get(f"latency.{opcode}.total_cycles", 0)
            max_cycles = stats.get(f"latency.{opcode}.max_cycles", 0)
            print(
                f"report latency opcode={opcode} samples={samples} total_cycles={total_cycles} max_cycles={max_cycles} avg_cycles={_format_average(total_cycles, samples)}"
            )
        return
    if report_name == "occupancy":
        occupancy_keys = sorted(key for key in stats if ".queue_occupancy." in key)
        occupancy_by_unit: dict[str, list[tuple[int, int]]] = {}
        for key in occupancy_keys:
            unit_name, _, depth = key.partition(".queue_occupancy.")
            occupancy_by_unit.setdefault(unit_name, []).append((int(depth), stats[key]))
        for unit_name in sorted(occupancy_by_unit):
            samples = sum(count for _, count in occupancy_by_unit[unit_name])
            weighted_depth = sum(depth * count for depth, count in occupancy_by_unit[unit_name])
            max_depth = stats.get(
                f"{unit_name}.max_queue_occupancy",
                max(depth for depth, _ in occupancy_by_unit[unit_name]),
            )
            print(
                f"report occupancy_summary unit={unit_name} samples={samples} avg_depth={_format_average(weighted_depth, samples)} max_depth={max_depth}"
            )
        for key in occupancy_keys:
            unit_name, _, depth = key.partition(".queue_occupancy.")
            print(f"report occupancy unit={unit_name} depth={depth} samples={stats[key]}")
        return
    if report_name == "memory":
        memory_keys = sorted(
            key
            for key in stats
            if key.endswith(".bytes_read") or key.endswith(".bytes_written")
        )
        for key in memory_keys:
            print(f"report memory key={key} value={stats[key]}")
        return
    if report_name == "contention":
        contention_keys = sorted(
            key
            for key in stats
            if "contention" in key or "bank_conflict" in key or "port_conflict" in key
        )
        for key in contention_keys:
            print(f"report contention key={key} value={stats[key]}")
        return
    if report_name == "stalls":
        stall_keys = sorted(key for key in stats if key.startswith("stall_"))
        total_stalls = sum(stats[key] for key in stall_keys)
        print(f"report stalls_summary total={total_stalls} categories={len(stall_keys)}")
        for key in stall_keys:
            print(f"report stalls key={key} value={stats[key]}")
        return
    if report_name == "units":
        unit_names = sorted(
            {
                key.removesuffix(".issued_ops")
                for key in stats
                if key.endswith(".issued_ops")
            }
            | {
                key.removesuffix(".busy_cycles")
                for key in stats
                if key.endswith(".busy_cycles")
            }
        )
        for unit_name in unit_names:
            print(
                f"report units unit={unit_name} issued_ops={stats.get(f'{unit_name}.issued_ops', 0)} busy_cycles={stats.get(f'{unit_name}.busy_cycles', 0)} max_queue_occupancy={stats.get(f'{unit_name}.max_queue_occupancy', 0)}"
            )
        return
    if report_name == "isa":
        sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
        for key in sample_keys:
            opcode = key.removeprefix("latency.").removesuffix(".samples")
            samples = stats[key]
            total_cycles = stats.get(f"latency.{opcode}.total_cycles", 0)
            print(f"report isa opcode={opcode} issued={samples} total_cycles={total_cycles}")
        return
    raise ValueError(f"Unsupported report {report_name!r}.")


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
    parser.add_argument(
        "--stats-json",
        type=str,
        default=None,
        help="Optional path for a JSON stats export, or '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--trace-json",
        type=str,
        default=None,
        help="Optional path for a JSON trace export, or '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--print-stats-prefix",
        action="append",
        default=[],
        help="Print sorted stats whose keys start with the provided prefix. May be repeated.",
    )
    parser.add_argument(
        "--print-trace-limit",
        type=int,
        default=0,
        help="Print the last N retained trace records after the summary.",
    )
    parser.add_argument(
        "--report",
        action="append",
        choices=("latency", "occupancy", "memory", "contention", "stalls", "units", "isa"),
        default=[],
        help="Print a curated report for one stats family. May be repeated.",
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
    if args.print_stats_prefix:
        for key in sorted(stats):
            if any(key.startswith(prefix) for prefix in args.print_stats_prefix):
                print(f"stat[{key}]={stats[key]}")
    if args.print_trace_limit > 0:
        for record in engine.trace.records[-args.print_trace_limit :]:
            print(f"trace cycle={record.cycle} kind={record.kind} message={record.message}")
    for report_name in args.report:
        emit_report(report_name, stats)
    if args.stats_json is not None:
        stats_payload = {
            "program": program.name,
            "halted": engine.state.halted,
            "exit_code": engine.state.exit_code,
            "trap": engine.state.trap_reason,
            "stats": stats,
        }
        serialized = json.dumps(stats_payload, indent=2, sort_keys=True)
        if args.stats_json == "-":
            print(serialized)
        else:
            Path(args.stats_json).write_text(serialized + "\n")
    if args.trace_json is not None:
        trace_payload = [
            {
                "cycle": record.cycle,
                "kind": record.kind,
                "message": record.message,
            }
            for record in engine.trace.records
        ]
        serialized = json.dumps(trace_payload, indent=2, sort_keys=True)
        if args.trace_json == "-":
            print(serialized)
        else:
            Path(args.trace_json).write_text(serialized + "\n")


if __name__ == "__main__":
    main()
