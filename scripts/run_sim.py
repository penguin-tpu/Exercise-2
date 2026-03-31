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
    load_experiment_manifest,
    load_memory_manifest,
    parse_image_load_spec,
    prepare_output_path,
)
from perf_modeling.config import (
    available_config_names,
    describe_named_config,
    get_named_config,
    snapshot_config,
)
from perf_modeling.decode import Decoder
from perf_modeling.reporting import (
    SWEEP_SORT_FIELDS,
    build_run_summary,
    build_sweep_csv_rows,
    emit_config_report,
    emit_report,
    emit_sweep_report,
    extract_sweep_sort_value,
    sort_sweep_results,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the simulator entrypoint."""
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
        "--list-configs",
        action="store_true",
        help="Print the available named accelerator configuration presets and exit.",
    )
    parser.add_argument(
        "--sweep-config",
        action="append",
        choices=available_config_names(),
        default=[],
        help="Run the same program across multiple named configs. May be repeated.",
    )
    parser.add_argument(
        "--sweep-json",
        type=str,
        default=None,
        help="Optional path for a JSON export of multi-config sweep results, or '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--sweep-csv",
        type=str,
        default=None,
        help="Optional path for a CSV export of multi-config sweep results, or '-' to write CSV to stdout.",
    )
    parser.add_argument(
        "--sweep-sort",
        choices=SWEEP_SORT_FIELDS,
        default="config",
        help="Metric used to order sweep results before printing and export.",
    )
    parser.add_argument(
        "--sweep-desc",
        action="store_true",
        help="Reverse the selected sweep ordering metric.",
    )
    parser.add_argument(
        "--sweep-limit",
        type=int,
        default=0,
        help="Optional maximum number of ranked sweep results to print and export. Zero means no limit.",
    )
    parser.add_argument(
        "--sweep-manifest-json",
        type=Path,
        default=None,
        help="Optional JSON manifest describing a grouped sweep experiment.",
    )
    parser.add_argument(
        "--experiment-json",
        type=Path,
        default=None,
        help="Optional JSON manifest describing a grouped single-run or sweep experiment.",
    )
    parser.add_argument(
        "--sweep-report",
        action="append",
        choices=("summary", "delta"),
        default=[],
        help="Print a curated comparative report over ranked sweep results. May be repeated.",
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
        choices=("summary", "config", "latency", "occupancy", "events", "fetch", "memory", "contention", "stalls", "pipeline", "units", "isa"),
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
    return parser


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse command-line arguments."""
    parser = build_parser()
    return parser, parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    effective_sweep_configs: list[str],
    effective_sweep_limit: int,
) -> None:
    """Validate combinations of CLI arguments before starting simulation."""
    if effective_sweep_limit < 0:
        parser.error("--sweep-limit must be zero or greater.")
    if args.sweep_json is not None and not effective_sweep_configs:
        parser.error("--sweep-json requires at least one --sweep-config entry.")
    if args.sweep_csv is not None and not effective_sweep_configs:
        parser.error("--sweep-csv requires at least one --sweep-config entry.")
    if args.sweep_report and not effective_sweep_configs:
        parser.error("--sweep-report requires at least one --sweep-config entry.")
    if not effective_sweep_configs:
        return
    if args.stats_json is not None:
        parser.error("--stats-json is not supported together with --sweep-config.")
    if args.stats_csv is not None:
        parser.error("--stats-csv is not supported together with --sweep-config.")
    if args.trace_json is not None:
        parser.error("--trace-json is not supported together with --sweep-config.")
    if args.trace_csv is not None:
        parser.error("--trace-csv is not supported together with --sweep-config.")
    if args.manifest_json is not None:
        parser.error("--manifest-json is not supported together with --sweep-config.")
    if args.scratchpad_dump is not None:
        parser.error("--scratchpad-dump is not supported together with --sweep-config.")
    if args.dram_dump is not None:
        parser.error("--dram-dump is not supported together with --sweep-config.")
    if args.print_stats_prefix:
        parser.error("--print-stats-prefix is not supported together with --sweep-config.")
    if args.print_trace_limit > 0:
        parser.error("--print-trace-limit is not supported together with --sweep-config.")
    if args.report:
        parser.error("--report is not supported together with --sweep-config.")


def build_default_program() -> bytes:
    """Return a tiny built-in RV32I smoke-test binary."""
    words = [
        0x02A00513,
        0x00000073,
    ]
    return b"".join(word.to_bytes(4, byteorder="little", signed=False) for word in words)


def main() -> None:
    """Load and run one RV32I program image."""
    parser, args = parse_args()
    if args.list_configs:
        for config_name in available_config_names():
            print(f"config name={config_name} description={describe_named_config(config_name)}")
        return
    if args.experiment_json is not None and args.sweep_manifest_json is not None:
        parser.error("Use either --experiment-json or --sweep-manifest-json, not both.")
    experiment_manifest = None
    manifest_path = args.experiment_json
    if manifest_path is None:
        manifest_path = args.sweep_manifest_json
    if manifest_path is not None:
        experiment_manifest = load_experiment_manifest(manifest_path)
    effective_program_path = args.program
    if effective_program_path is None and experiment_manifest is not None:
        effective_program_path = experiment_manifest.program
    effective_base_address = args.base_address
    if effective_base_address == 0x1000 and experiment_manifest is not None and experiment_manifest.base_address is not None:
        effective_base_address = experiment_manifest.base_address
    effective_max_cycles = args.max_cycles
    if effective_max_cycles == 100000 and experiment_manifest is not None and experiment_manifest.max_cycles is not None:
        effective_max_cycles = experiment_manifest.max_cycles
    effective_sweep_configs = list(args.sweep_config)
    if not effective_sweep_configs and experiment_manifest is not None:
        effective_sweep_configs = list(experiment_manifest.sweep_configs)
    effective_sweep_sort = args.sweep_sort
    if effective_sweep_sort == "config" and experiment_manifest is not None and experiment_manifest.sweep_sort is not None:
        effective_sweep_sort = experiment_manifest.sweep_sort
    effective_sweep_desc = args.sweep_desc
    if not effective_sweep_desc and experiment_manifest is not None and experiment_manifest.sweep_desc is not None:
        effective_sweep_desc = experiment_manifest.sweep_desc
    effective_sweep_limit = args.sweep_limit
    if effective_sweep_limit == 0 and experiment_manifest is not None and experiment_manifest.sweep_limit is not None:
        effective_sweep_limit = experiment_manifest.sweep_limit
    effective_config_name = args.config
    if (
        effective_config_name == "baseline"
        and not effective_sweep_configs
        and experiment_manifest is not None
        and experiment_manifest.config is not None
    ):
        effective_config_name = experiment_manifest.config
    validate_args(args, parser, effective_sweep_configs, effective_sweep_limit)
    decoder = Decoder()
    if effective_program_path is None:
        blob = build_default_program()
        program = decoder.decode_bytes(blob, base_address=effective_base_address, name="builtin-smoke")
    else:
        program = decode_program_from_path(effective_program_path, effective_base_address, decoder, REPO_ROOT)
    manifest_dram_loads: list[tuple[int, Path]] = []
    manifest_scratchpad_loads: list[tuple[int, Path]] = []
    if args.memory_loads_json is not None:
        manifest_dram_loads, manifest_scratchpad_loads = load_memory_manifest(args.memory_loads_json)
    sweep_manifest_dram_loads = list(experiment_manifest.dram_loads) if experiment_manifest is not None else []
    sweep_manifest_scratchpad_loads = list(experiment_manifest.scratchpad_loads) if experiment_manifest is not None else []
    dram_images = [(address, path.read_bytes()) for address, path in sweep_manifest_dram_loads]
    dram_images.extend((address, path.read_bytes()) for address, path in manifest_dram_loads)
    dram_images.extend((address, path.read_bytes()) for address, path in args.dram_load)
    scratchpad_images = [(offset, path.read_bytes()) for offset, path in sweep_manifest_scratchpad_loads]
    scratchpad_images.extend((offset, path.read_bytes()) for offset, path in manifest_scratchpad_loads)
    scratchpad_images.extend((offset, path.read_bytes()) for offset, path in args.scratchpad_load)
    if effective_sweep_configs:
        sweep_results: list[dict[str, object]] = []
        print(f"sweep program={program.name} configs={len(effective_sweep_configs)}")
        for config_name in effective_sweep_configs:
            sweep_engine = SimulatorEngine(config=get_named_config(config_name), program=program)
            sweep_config_snapshot = snapshot_config(sweep_engine.config)
            for address, payload in dram_images:
                sweep_engine.state.dram.load_image(address, payload)
            for offset, payload in scratchpad_images:
                sweep_engine.state.scratchpad.load_image(offset, payload)
            sweep_stats = sweep_engine.run(max_cycles=effective_max_cycles).snapshot()
            sweep_summary = build_run_summary(sweep_stats)
            sweep_results.append(
                {
                    "config": config_name,
                    "config_snapshot": sweep_config_snapshot,
                    "halted": sweep_engine.state.halted,
                    "exit_code": sweep_engine.state.exit_code,
                    "trap": sweep_engine.state.trap_reason,
                    "cycles": sweep_stats.get("cycles", 0),
                    "summary": sweep_summary,
                    "stats": sweep_stats,
                }
            )
        sweep_results = sort_sweep_results(sweep_results, effective_sweep_sort, effective_sweep_desc)
        if effective_sweep_limit > 0:
            sweep_results = sweep_results[:effective_sweep_limit]
        for index, result in enumerate(sweep_results, start=1):
            sort_value = extract_sweep_sort_value(result, effective_sweep_sort)
            summary = result["summary"]
            assert isinstance(summary, dict)
            pipeline = summary["pipeline"]
            busiest_unit = summary["busiest_unit"]
            assert isinstance(pipeline, dict)
            assert isinstance(busiest_unit, dict)
            print(
                f"sweep rank={index} config={result['config']} sort={effective_sweep_sort} sort_value={sort_value} cycles={result['cycles']} retired={pipeline['retired']} halted={result['halted']} exit_code={result['exit_code']} busiest_unit={busiest_unit['name']}"
            )
        for report_name in args.sweep_report:
            emit_sweep_report(
                report_name,
                sweep_results,
                effective_sweep_sort,
                effective_sweep_desc,
                effective_sweep_limit if effective_sweep_limit > 0 else None,
            )
        if args.sweep_json is not None:
            sweep_payload = {
                "program": program.name,
                "sort": {
                    "field": effective_sweep_sort,
                    "descending": effective_sweep_desc,
                },
                "limit": effective_sweep_limit if effective_sweep_limit > 0 else None,
                "results": sweep_results,
            }
            serialized = json.dumps(sweep_payload, indent=2, sort_keys=True)
            if args.sweep_json == "-":
                print(serialized)
            else:
                sweep_json_path = prepare_output_path(args.sweep_json, args.output_dir)
                sweep_json_path.write_text(serialized + "\n")
        if args.sweep_csv is not None:
            rows = build_sweep_csv_rows(sweep_results)
            if args.sweep_csv == "-":
                writer = csv.writer(sys.stdout)
                writer.writerows(rows)
            else:
                sweep_csv_path = prepare_output_path(args.sweep_csv, args.output_dir)
                with sweep_csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerows(rows)
        return
    engine = SimulatorEngine(config=get_named_config(effective_config_name), program=program)
    config_snapshot = snapshot_config(engine.config)
    for address, payload in dram_images:
        engine.state.dram.load_image(address, payload)
    for offset, payload in scratchpad_images:
        engine.state.scratchpad.load_image(offset, payload)
    stats = engine.run(max_cycles=effective_max_cycles).snapshot()
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
        if report_name == "config":
            emit_config_report(
                args.config,
                config_snapshot,
                report_limit=args.report_limit if args.report_limit > 0 else None,
                report_match=args.report_match,
            )
            continue
        emit_report(
            report_name,
            stats,
            report_limit=args.report_limit if args.report_limit > 0 else None,
            report_match=args.report_match,
        )
    if args.stats_json is not None:
        stats_payload = {
            "program": program.name,
            "config": effective_config_name,
            "config_snapshot": config_snapshot,
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
            "config": effective_config_name,
            "config_snapshot": config_snapshot,
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
