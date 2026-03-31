"""CLI entry point for the RV32I functional and performance model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf_modeling import AcceleratorConfig, SimulatorEngine
from perf_modeling.decode import Decoder
from toolchains.riscv32.assemble_to_elf import compile_to_elf
from toolchains.riscv32.gnu_toolchain import resolve_toolchain


ASSEMBLY_SUFFIXES = frozenset({".s", ".asm"})
"""File suffixes treated as assembly sources for transient ELF assembly."""


def _format_average(total_cycles: int, samples: int) -> str:
    """Format one average-latency value for human-readable reports."""
    if samples <= 0:
        return "0.00"
    return f"{total_cycles / samples:.2f}"


def _format_percentage(numerator: int, denominator: int) -> str:
    """Format one percentage for human-readable reports."""
    if denominator <= 0:
        return "0.00"
    return f"{(numerator * 100) / denominator:.2f}"


def _format_per_cycle(count: int, cycles: int) -> str:
    """Format one per-cycle rate for human-readable reports."""
    if cycles <= 0:
        return "0.00"
    return f"{count / cycles:.2f}"


def _matches_report_filter(value: str, report_match: str | None) -> bool:
    """Return whether one report field matches the optional substring filter."""
    if report_match is None or report_match == "":
        return True
    return report_match.lower() in value.lower()


def _parse_image_load_spec(value: str) -> tuple[int, Path]:
    """Parse one CLI memory-image load specification of the form `ADDRESS:PATH`."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected memory load spec in the form ADDRESS:PATH.")
    address_text, path_text = value.split(":", 1)
    return int(address_text, 0), Path(path_text)


def _parse_int_like(value: object, field_name: str) -> int:
    """Parse one manifest integer field from either numeric or string JSON input."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError(f"Expected integer-like value for {field_name}.")


def _load_memory_manifest(manifest_path: Path) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    """Load DRAM and scratchpad preload specs from one JSON manifest file."""
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Memory load manifest must contain a top-level JSON object.")
    manifest_root = manifest_path.parent
    dram_loads: list[tuple[int, Path]] = []
    for entry in payload.get("dram", []):
        if not isinstance(entry, dict):
            raise ValueError("Each DRAM manifest entry must be a JSON object.")
        address = _parse_int_like(entry.get("address"), "dram.address")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        dram_loads.append((address, path))
    scratchpad_loads: list[tuple[int, Path]] = []
    for entry in payload.get("scratchpad", []):
        if not isinstance(entry, dict):
            raise ValueError("Each scratchpad manifest entry must be a JSON object.")
        offset = _parse_int_like(entry.get("offset"), "scratchpad.offset")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        scratchpad_loads.append((offset, path))
    return dram_loads, scratchpad_loads


def _path_is_under_directory(path: Path, directory: Path) -> bool:
    """Return whether one relative path already begins under the selected output directory."""
    if len(path.parts) < len(directory.parts):
        return False
    return path.parts[: len(directory.parts)] == directory.parts


def _prepare_output_path(path_text: str, output_dir: str) -> Path:
    """Create parent directories for one file output target and return the normalized path."""
    path = Path(path_text)
    output_root = Path(output_dir)
    if not path.is_absolute() and not _path_is_under_directory(path, output_root):
        path = output_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_run_summary(stats: dict[str, int]) -> dict[str, object]:
    """Build one compact summary view from the flattened stats snapshot."""
    cycles = stats.get("cycles", 0)
    issued = stats.get("instructions_issued", 0)
    retired = stats.get("instructions_retired", 0)
    total_stalls = sum(value for key, value in stats.items() if key.startswith("stall_"))

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
    busiest_unit = "none"
    busiest_busy_cycles = -1
    for unit_name in unit_names:
        busy_cycles = stats.get(f"{unit_name}.busy_cycles", 0)
        if busy_cycles > busiest_busy_cycles:
            busiest_unit = unit_name
            busiest_busy_cycles = busy_cycles

    sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
    top_opcode = "none"
    top_total_cycles = -1
    for key in sample_keys:
        opcode = key.removeprefix("latency.").removesuffix(".samples")
        total_cycles = stats.get(f"latency.{opcode}.total_cycles", 0)
        if total_cycles > top_total_cycles:
            top_opcode = opcode
            top_total_cycles = total_cycles

    memory_keys = sorted(
        key
        for key in stats
        if key.endswith(".bytes_read") or key.endswith(".bytes_written")
    )
    top_memory_key = "none"
    top_memory_bytes = -1
    for key in memory_keys:
        if stats[key] > top_memory_bytes:
            top_memory_key = key
            top_memory_bytes = stats[key]

    contention_keys = sorted(
        key
        for key in stats
        if key.startswith("memory.contention.resource.")
        or key.startswith("scratchpad.bank_conflict.")
        or key.startswith("scratchpad.port_conflict.")
    )
    top_contention_key = "none"
    top_contention_value = -1
    for key in contention_keys:
        if stats[key] > top_contention_value:
            top_contention_key = key
            top_contention_value = stats[key]

    latency_samples = stats.get(f"latency.{top_opcode}.samples", 0) if top_opcode != "none" else 0
    latency_total_cycles = stats.get(f"latency.{top_opcode}.total_cycles", 0) if top_opcode != "none" else 0
    latency_max_cycles = stats.get(f"latency.{top_opcode}.max_cycles", 0) if top_opcode != "none" else 0

    return {
        "pipeline": {
            "cycles": cycles,
            "issued": issued,
            "retired": retired,
            "total_stalls": total_stalls,
        },
        "busiest_unit": {
            "name": busiest_unit,
            "busy_cycles": max(busiest_busy_cycles, 0),
            "busy_pct": _format_percentage(max(busiest_busy_cycles, 0), cycles),
            "issued_ops": stats.get(f"{busiest_unit}.issued_ops", 0) if busiest_unit != "none" else 0,
        },
        "latency_hotspot": {
            "opcode": top_opcode,
            "avg_cycles": _format_average(latency_total_cycles, latency_samples),
            "max_cycles": latency_max_cycles,
        },
        "memory_hotspot": {
            "key": top_memory_key,
            "total_bytes": max(top_memory_bytes, 0),
        },
        "contention_hotspot": {
            "key": top_contention_key,
            "value": max(top_contention_value, 0),
        },
    }


def _decode_program_from_path(program_path: Path, base_address: int, decoder: Decoder):
    """Decode one user-supplied program path, auto-assembling assembly sources first."""
    if program_path.suffix.lower() in ASSEMBLY_SUFFIXES:
        toolchain = resolve_toolchain(REPO_ROOT)
        with tempfile.TemporaryDirectory() as temp_dir:
            elf_path = Path(temp_dir) / f"{program_path.stem}.elf"
            compile_to_elf(program_path, elf_path, toolchain.compiler, base_address)
            return decoder.decode_bytes(elf_path.read_bytes(), name=program_path.name)
    return decoder.decode_bytes(
        program_path.read_bytes(),
        base_address=base_address,
        name=program_path.name,
    )


def emit_report(
    report_name: str,
    stats: dict[str, int],
    report_limit: int | None = None,
    report_match: str | None = None,
) -> None:
    """Print one curated report from the flattened stats snapshot."""
    if report_name == "summary":
        summary = build_run_summary(stats)
        pipeline = summary["pipeline"]
        busiest_unit = summary["busiest_unit"]
        latency_hotspot = summary["latency_hotspot"]
        memory_hotspot = summary["memory_hotspot"]
        contention_hotspot = summary["contention_hotspot"]
        print(
            f"report summary pipeline cycles={pipeline['cycles']} issued={pipeline['issued']} retired={pipeline['retired']} total_stalls={pipeline['total_stalls']}"
        )
        print(
            f"report summary unit={busiest_unit['name']} busy_cycles={busiest_unit['busy_cycles']} busy_pct={busiest_unit['busy_pct']} issued_ops={busiest_unit['issued_ops']}"
        )
        print(
            f"report summary latency opcode={latency_hotspot['opcode']} avg_cycles={latency_hotspot['avg_cycles']} max_cycles={latency_hotspot['max_cycles']}"
        )
        print(
            f"report summary memory key={memory_hotspot['key']} total_bytes={memory_hotspot['total_bytes']}"
        )
        print(
            f"report summary contention key={contention_hotspot['key']} value={contention_hotspot['value']}"
        )
        return
    if report_name == "latency":
        rows: list[tuple[int, str, int, int, int]] = []
        sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
        for key in sample_keys:
            opcode = key.removeprefix("latency.").removesuffix(".samples")
            if not _matches_report_filter(opcode, report_match):
                continue
            samples = stats[key]
            total_cycles = stats.get(f"latency.{opcode}.total_cycles", 0)
            max_cycles = stats.get(f"latency.{opcode}.max_cycles", 0)
            rows.append((total_cycles, opcode, samples, max_cycles, total_cycles))
        rows.sort(key=lambda row: (-row[0], row[1]))
        if report_limit is not None:
            rows = rows[:report_limit]
        for _, opcode, samples, max_cycles, total_cycles in rows:
            print(
                f"report latency opcode={opcode} samples={samples} total_cycles={total_cycles} max_cycles={max_cycles} avg_cycles={_format_average(total_cycles, samples)}"
            )
        return
    if report_name == "occupancy":
        occupancy_keys = sorted(key for key in stats if ".queue_occupancy." in key)
        occupancy_by_unit: dict[str, list[tuple[int, int]]] = {}
        for key in occupancy_keys:
            unit_name, _, depth = key.partition(".queue_occupancy.")
            if not _matches_report_filter(unit_name, report_match):
                continue
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
        read_keys = sorted(key for key in stats if key.endswith(".bytes_read"))
        write_keys = sorted(key for key in stats if key.endswith(".bytes_written"))
        read_keys = [key for key in read_keys if _matches_report_filter(key, report_match)]
        write_keys = [key for key in write_keys if _matches_report_filter(key, report_match)]
        total_read = sum(stats[key] for key in read_keys)
        total_write = sum(stats[key] for key in write_keys)
        print(f"report memory_summary direction=read total_bytes={total_read}")
        if report_limit is not None:
            read_keys = sorted(read_keys, key=lambda key: (-stats[key], key))[:report_limit]
            write_keys = sorted(write_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in read_keys:
            print(
                f"report memory key={key} value={stats[key]} pct={_format_percentage(stats[key], total_read)}"
            )
        print(f"report memory_summary direction=write total_bytes={total_write}")
        for key in write_keys:
            print(
                f"report memory key={key} value={stats[key]} pct={_format_percentage(stats[key], total_write)}"
            )
        return
    if report_name == "contention":
        contention_keys = sorted(
            key
            for key in stats
            if "contention" in key or "bank_conflict" in key or "port_conflict" in key
        )
        stall_total = sum(value for key, value in stats.items() if key.startswith("stall_"))
        resource_keys = {
            key
            for key in contention_keys
            if key.startswith("memory.contention.resource.")
            or key.startswith("scratchpad.bank_conflict.")
            or key.startswith("scratchpad.port_conflict.")
        }
        contention_keys = [key for key in contention_keys if _matches_report_filter(key, report_match)]
        resource_total = sum(stats[key] for key in resource_keys)
        print(f"report contention_summary family=stall total={stall_total}")
        print(f"report contention_summary family=resource total={resource_total}")
        if report_limit is not None:
            contention_keys = sorted(contention_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in contention_keys:
            if key in resource_keys:
                print(
                    f"report contention key={key} value={stats[key]} pct={_format_percentage(stats[key], resource_total)}"
                )
                continue
            print(f"report contention key={key} value={stats[key]}")
        return
    if report_name == "stalls":
        stall_keys = sorted(key for key in stats if key.startswith("stall_"))
        stall_keys = [key for key in stall_keys if _matches_report_filter(key, report_match)]
        total_stalls = sum(stats[key] for key in stall_keys)
        print(f"report stalls_summary total={total_stalls} categories={len(stall_keys)}")
        if report_limit is not None:
            stall_keys = sorted(stall_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in stall_keys:
            print(f"report stalls key={key} value={stats[key]}")
        return
    if report_name == "pipeline":
        cycles = stats.get("cycles", 0)
        issued = stats.get("instructions_issued", 0)
        retired = stats.get("instructions_retired", 0)
        total_stalls = sum(value for key, value in stats.items() if key.startswith("stall_"))
        print(
            f"report pipeline cycles={cycles} issued={issued} retired={retired} issue_per_cycle={_format_per_cycle(issued, cycles)} retire_per_cycle={_format_per_cycle(retired, cycles)} total_stalls={total_stalls}"
        )
        return
    if report_name == "units":
        total_cycles = stats.get("cycles", 0)
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
        unit_names = [unit_name for unit_name in unit_names if _matches_report_filter(unit_name, report_match)]
        if report_limit is not None:
            unit_names = sorted(
                unit_names,
                key=lambda unit_name: (-stats.get(f"{unit_name}.busy_cycles", 0), unit_name),
            )[:report_limit]
        for unit_name in unit_names:
            busy_cycles = stats.get(f"{unit_name}.busy_cycles", 0)
            print(
                f"report units unit={unit_name} issued_ops={stats.get(f'{unit_name}.issued_ops', 0)} busy_cycles={busy_cycles} busy_pct={_format_percentage(busy_cycles, total_cycles)} max_queue_occupancy={stats.get(f'{unit_name}.max_queue_occupancy', 0)}"
            )
        return
    if report_name == "isa":
        rows: list[tuple[int, str, int]] = []
        sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
        for key in sample_keys:
            opcode = key.removeprefix("latency.").removesuffix(".samples")
            if not _matches_report_filter(opcode, report_match):
                continue
            samples = stats[key]
            total_cycles = stats.get(f"latency.{opcode}.total_cycles", 0)
            rows.append((total_cycles, opcode, samples))
        rows.sort(key=lambda row: (-row[0], row[1]))
        if report_limit is not None:
            rows = rows[:report_limit]
        for total_cycles, opcode, samples in rows:
            print(f"report isa opcode={opcode} issued={samples} total_cycles={total_cycles}")
        return
    raise ValueError(f"Unsupported report {report_name!r}.")


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
        "--dram-load",
        action="append",
        type=_parse_image_load_spec,
        default=[],
        help="Repeatable DRAM preload in the form ADDRESS:PATH.",
    )
    parser.add_argument(
        "--scratchpad-load",
        action="append",
        type=_parse_image_load_spec,
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
        choices=("summary", "latency", "occupancy", "memory", "contention", "stalls", "pipeline", "units", "isa"),
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
        program = _decode_program_from_path(args.program, args.base_address, decoder)
    engine = SimulatorEngine(config=AcceleratorConfig(), program=program)
    manifest_dram_loads: list[tuple[int, Path]] = []
    manifest_scratchpad_loads: list[tuple[int, Path]] = []
    if args.memory_loads_json is not None:
        manifest_dram_loads, manifest_scratchpad_loads = _load_memory_manifest(args.memory_loads_json)
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
            stats_json_path = _prepare_output_path(args.stats_json, args.output_dir)
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
            stats_csv_path = _prepare_output_path(args.stats_csv, args.output_dir)
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
            trace_json_path = _prepare_output_path(args.trace_json, args.output_dir)
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
            trace_csv_path = _prepare_output_path(args.trace_csv, args.output_dir)
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
        scratchpad_dump_path = _prepare_output_path(args.scratchpad_dump, args.output_dir)
        scratchpad_dump_path.write_bytes(payload)
        artifact_outputs["scratchpad_dump"] = str(scratchpad_dump_path.resolve())
    if args.dram_dump is not None:
        offset = args.dram_dump_offset
        if args.dram_dump_size > 0:
            size = args.dram_dump_size
        else:
            size = engine.config.dram.capacity_bytes - offset
        payload = engine.state.dram.read(offset, size)
        dram_dump_path = _prepare_output_path(args.dram_dump, args.output_dir)
        dram_dump_path.write_bytes(payload)
        artifact_outputs["dram_dump"] = str(dram_dump_path.resolve())
    if args.manifest_json is not None:
        manifest_destination = "stdout"
        if args.manifest_json != "-":
            manifest_destination = str(_prepare_output_path(args.manifest_json, args.output_dir).resolve())
        manifest_payload = {
            "program": program.name,
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
