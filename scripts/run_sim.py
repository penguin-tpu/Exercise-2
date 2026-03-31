"""CLI entry point for the RV32I functional and performance model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf_modeling import AcceleratorConfig, SimulatorEngine
from perf_modeling.cli_support import (
    decode_program_from_path,
    load_memory_manifest,
    parse_image_load_spec,
    prepare_output_path,
)
from perf_modeling.config import available_config_names, get_named_config
from perf_modeling.decode import Decoder
from perf_modeling.reporting import build_run_summary, emit_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Execute a bare-metal RV32I binary in the simulator.")
    parser.add_argument(
        "program",
        type=Path,
        nargs="?",
        help="Path to a raw binary, ELF32 image, or assembly source.",
    )
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
        "--config",
        choices=available_config_names(),
        default="baseline",
        help="Named accelerator configuration preset to use for the run.",
    )
    parser.add_argument(
        "--dram-load",
        action="append",
        type=parse_image_load_spec,
        default=[],
        help="Repeatable DRAM preload in the form ADDRESS:PATH.",
    )
    parser.add_argument(
        "--scratchpad-load",
        action="append",
        type=parse_image_load_spec,
        default=[],
        help="Repeatable scratchpad preload in the form OFFSET:PATH.",
    )
    parser.add_argument(
        "--memory-loads-json",
        type=Path,
        default=None,
        help="Optional JSON manifest describing DRAM and scratchpad preloads.",
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        default=None,
        help="Optional path for a JSON stats export, or '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--stats-csv",
        type=str,
        default=None,
        help="Optional path for a CSV stats export, or '-' to write CSV to stdout.",
    )
    parser.add_argument(
        "--trace-json",
        type=str,
        default=None,
        help="Optional path for a JSON trace export, or '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--trace-csv",
        type=str,
        default=None,
        help="Optional path for a CSV trace export, or '-' to write CSV to stdout.",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default=None,
        help="Optional path for a JSON run manifest, or '-' to write the manifest to stdout.",
    )
    parser.add_argument(
        "--scratchpad-dump",
        type=str,
        default=None,
        help="Optional path for a raw scratchpad dump written after simulation completes.",
    )
    parser.add_argument(
        "--scratchpad-dump-offset",
        type=lambda value: int(value, 0),
        default=0,
        help="Byte offset within scratchpad for the dump start.",
    )
    parser.add_argument(
        "--scratchpad-dump-size",
        type=int,
        default=0,
        help="Number of scratchpad bytes to dump. Zero means from offset to scratchpad end.",
    )
    parser.add_argument(
        "--dram-dump",
        type=str,
        default=None,
        help="Optional path for a raw DRAM dump written after simulation completes.",
    )
    parser.add_argument(
        "--dram-dump-offset",
        type=lambda value: int(value, 0),
        default=0,
        help="Byte offset within DRAM for the dump start.",
    )
    parser.add_argument(
        "--dram-dump-size",
        type=int,
        default=0,
        help="Number of DRAM bytes to dump. Zero means from offset to DRAM end.",
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
        choices=("summary", "latency", "occupancy", "events", "fetch", "memory", "contention", "stalls", "pipeline", "units", "isa"),
        default=[],
        help="Print a curated report for one stats family. May be repeated.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=0,
        help="Optional maximum number of detail rows to print for each multi-row report.",
    )
    parser.add_argument(
        "--report-match",
        type=str,
        default=None,
        help="Optional case-insensitive substring filter for multi-row report entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Base directory used for relative artifact outputs.",
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
        program = decode_program_from_path(args.program, args.base_address, decoder, REPO_ROOT)
    engine = SimulatorEngine(config=get_named_config(args.config), program=program)
    manifest_dram_loads: list[tuple[int, Path]] = []
    manifest_scratchpad_loads: list[tuple[int, Path]] = []
    if args.memory_loads_json is not None:
        manifest_dram_loads, manifest_scratchpad_loads = load_memory_manifest(args.memory_loads_json)
    for address, path in manifest_dram_loads:
        engine.state.dram.load_image(address, path.read_bytes())
    for offset, path in manifest_scratchpad_loads:
        engine.state.scratchpad.load_image(offset, path.read_bytes())
    for address, path in args.dram_load:
        engine.state.dram.load_image(address, path.read_bytes())
    for offset, path in args.scratchpad_load:
        engine.state.scratchpad.load_image(offset, path.read_bytes())
    stats = engine.run(max_cycles=args.max_cycles).snapshot()
    artifact_outputs: dict[str, str] = {}
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
        emit_report(
            report_name,
            stats,
            report_limit=args.report_limit if args.report_limit > 0 else None,
            report_match=args.report_match,
        )
    if args.stats_json is not None:
        stats_payload = {
            "program": program.name,
            "config": args.config,
            "halted": engine.state.halted,
            "exit_code": engine.state.exit_code,
            "trap": engine.state.trap_reason,
            "stats": stats,
        }
        serialized = json.dumps(stats_payload, indent=2, sort_keys=True)
        if args.stats_json == "-":
            print(serialized)
            artifact_outputs["stats_json"] = "stdout"
        else:
            stats_json_path = prepare_output_path(args.stats_json, args.output_dir)
            stats_json_path.write_text(serialized + "\n")
            artifact_outputs["stats_json"] = str(stats_json_path.resolve())
    if args.stats_csv is not None:
        rows = [("key", "value")]
        rows.extend((key, str(value)) for key, value in sorted(stats.items()))
        if args.stats_csv == "-":
            writer = csv.writer(sys.stdout)
            writer.writerows(rows)
            artifact_outputs["stats_csv"] = "stdout"
        else:
            stats_csv_path = prepare_output_path(args.stats_csv, args.output_dir)
            with stats_csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)
            artifact_outputs["stats_csv"] = str(stats_csv_path.resolve())
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
            artifact_outputs["trace_json"] = "stdout"
        else:
            trace_json_path = prepare_output_path(args.trace_json, args.output_dir)
            trace_json_path.write_text(serialized + "\n")
            artifact_outputs["trace_json"] = str(trace_json_path.resolve())
    if args.trace_csv is not None:
        rows = [("cycle", "kind", "message")]
        rows.extend((str(record.cycle), record.kind, record.message) for record in engine.trace.records)
        if args.trace_csv == "-":
            writer = csv.writer(sys.stdout)
            writer.writerows(rows)
            artifact_outputs["trace_csv"] = "stdout"
        else:
            trace_csv_path = prepare_output_path(args.trace_csv, args.output_dir)
            with trace_csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)
            artifact_outputs["trace_csv"] = str(trace_csv_path.resolve())
    if args.scratchpad_dump is not None:
        offset = args.scratchpad_dump_offset
        if args.scratchpad_dump_size > 0:
            size = args.scratchpad_dump_size
        else:
            size = engine.config.scratchpad.capacity_bytes - offset
        payload = engine.state.scratchpad.read(offset, size)
        scratchpad_dump_path = prepare_output_path(args.scratchpad_dump, args.output_dir)
        scratchpad_dump_path.write_bytes(payload)
        artifact_outputs["scratchpad_dump"] = str(scratchpad_dump_path.resolve())
    if args.dram_dump is not None:
        offset = args.dram_dump_offset
        if args.dram_dump_size > 0:
            size = args.dram_dump_size
        else:
            size = engine.config.dram.capacity_bytes - offset
        payload = engine.state.dram.read(offset, size)
        dram_dump_path = prepare_output_path(args.dram_dump, args.output_dir)
        dram_dump_path.write_bytes(payload)
        artifact_outputs["dram_dump"] = str(dram_dump_path.resolve())
    if args.manifest_json is not None:
        manifest_destination = "stdout"
        if args.manifest_json != "-":
            manifest_destination = str(prepare_output_path(args.manifest_json, args.output_dir).resolve())
        manifest_payload = {
            "program": program.name,
            "config": args.config,
            "halted": engine.state.halted,
            "exit_code": engine.state.exit_code,
            "trap": engine.state.trap_reason,
            "cycles": stats.get("cycles", 0),
            "summary": build_run_summary(stats),
            "artifacts": artifact_outputs,
            "manifest": manifest_destination,
        }
        serialized = json.dumps(manifest_payload, indent=2, sort_keys=True)
        if args.manifest_json == "-":
            print(serialized)
        else:
            Path(manifest_destination).write_text(serialized + "\n")


if __name__ == "__main__":
    main()
