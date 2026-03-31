"""Helper utilities shared by the simulator CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from perf_modeling.decode import Decoder
from toolchains.riscv32.assemble_to_elf import compile_to_elf
from toolchains.riscv32.gnu_toolchain import resolve_toolchain


SOURCE_SUFFIXES = frozenset({".s", ".asm", ".c"})
"""File suffixes treated as compileable RV32I sources for transient ELF generation."""


@dataclass(frozen=True)
class ExperimentManifest:
    """Resolved contents of one grouped experiment manifest."""

    program: Path | None
    """Optional program path used for the experiment."""
    base_address: int | None
    """Optional raw-binary base address override."""
    max_cycles: int | None
    """Optional cycle limit override."""
    config: str | None
    """Optional single-run named config."""
    report_names: tuple[str, ...]
    """Ordered single-run reports selected by the manifest."""
    report_limit: int | None
    """Optional single-run report detail limit."""
    report_match: str | None
    """Optional single-run report substring filter."""
    print_stats_prefixes: tuple[str, ...]
    """Optional single-run stat-prefix filters printed to stdout."""
    print_trace_limit: int | None
    """Optional single-run retained-trace tail length printed to stdout."""
    sweep_configs: tuple[str, ...]
    """Ordered named configs included in the sweep."""
    sweep_reports: tuple[str, ...]
    """Ordered sweep comparative reports selected by the manifest."""
    sweep_sort: str | None
    """Optional sweep ranking metric."""
    sweep_desc: bool | None
    """Optional reverse-order flag for sweep ranking."""
    sweep_limit: int | None
    """Optional top-N limit applied after ranking."""
    dram_loads: tuple[tuple[int, Path], ...]
    """Optional DRAM image preloads for each sweep run."""
    scratchpad_loads: tuple[tuple[int, Path], ...]
    """Optional scratchpad image preloads for each sweep run."""
    output_dir: str | None
    """Optional base directory for relative artifact outputs."""
    stats_json: str | None
    """Optional single-run stats JSON output."""
    stats_csv: str | None
    """Optional single-run stats CSV output."""
    trace_json: str | None
    """Optional single-run trace JSON output."""
    trace_csv: str | None
    """Optional single-run trace CSV output."""
    perfetto_trace: str | None
    """Optional single-run Perfetto trace JSON output."""
    manifest_json: str | None
    """Optional single-run manifest JSON output."""
    scratchpad_dump: str | None
    """Optional single-run scratchpad dump output."""
    scratchpad_dump_offset: int | None
    """Optional single-run scratchpad dump byte offset."""
    scratchpad_dump_size: int | None
    """Optional single-run scratchpad dump byte size."""
    dram_dump: str | None
    """Optional single-run DRAM dump output."""
    dram_dump_offset: int | None
    """Optional single-run DRAM dump byte offset."""
    dram_dump_size: int | None
    """Optional single-run DRAM dump byte size."""
    sweep_json: str | None
    """Optional sweep JSON output."""
    sweep_csv: str | None
    """Optional sweep CSV output."""


def parse_image_load_spec(value: str) -> tuple[int, Path]:
    """Parse one CLI memory-image load specification of the form `ADDRESS:PATH`."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected memory load spec in the form ADDRESS:PATH.")
    address_text, path_text = value.split(":", 1)
    return int(address_text, 0), Path(path_text)


def parse_int_like(value: object, field_name: str) -> int:
    """Parse one manifest integer field from either numeric or string JSON input."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError(f"Expected integer-like value for {field_name}.")


def load_memory_manifest(manifest_path: Path) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    """Load DRAM and scratchpad preload specs from one JSON manifest file."""
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Memory load manifest must contain a top-level JSON object.")
    manifest_root = manifest_path.parent
    dram_loads: list[tuple[int, Path]] = []
    for entry in payload.get("dram", []):
        if not isinstance(entry, dict):
            raise ValueError("Each DRAM manifest entry must be a JSON object.")
        address = parse_int_like(entry.get("address"), "dram.address")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        dram_loads.append((address, path))
    scratchpad_loads: list[tuple[int, Path]] = []
    for entry in payload.get("scratchpad", []):
        if not isinstance(entry, dict):
            raise ValueError("Each scratchpad manifest entry must be a JSON object.")
        offset = parse_int_like(entry.get("offset"), "scratchpad.offset")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        scratchpad_loads.append((offset, path))
    return dram_loads, scratchpad_loads


def load_experiment_manifest(manifest_path: Path) -> ExperimentManifest:
    """Load one grouped experiment manifest with program, configs, ordering, and preloads."""
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Experiment manifest must contain a top-level JSON object.")
    manifest_root = manifest_path.parent
    program = payload.get("program")
    program_path: Path | None = None
    if program is not None:
        program_path = Path(program)
        if not program_path.is_absolute():
            program_path = manifest_root / program_path
    sweep_configs_value = payload.get("sweep_configs", [])
    if not isinstance(sweep_configs_value, list):
        raise ValueError("sweep_configs must be a JSON array when present.")
    sweep_configs = tuple(str(config_name) for config_name in sweep_configs_value)
    sweep_reports_value = payload.get("sweep_reports", [])
    if not isinstance(sweep_reports_value, list):
        raise ValueError("sweep_reports must be a JSON array when present.")
    sweep_reports = tuple(str(report_name) for report_name in sweep_reports_value)
    config = None
    if "config" in payload:
        config = str(payload["config"])
    report_names_value = payload.get("reports", [])
    if not isinstance(report_names_value, list):
        raise ValueError("reports must be a JSON array when present.")
    report_names = tuple(str(report_name) for report_name in report_names_value)
    report_limit = None
    if "report_limit" in payload:
        report_limit = parse_int_like(payload["report_limit"], "report_limit")
    report_match = None
    if "report_match" in payload:
        report_match = str(payload["report_match"])
    print_stats_prefixes_value = payload.get("print_stats_prefix", [])
    if not isinstance(print_stats_prefixes_value, list):
        raise ValueError("print_stats_prefix must be a JSON array when present.")
    print_stats_prefixes = tuple(str(prefix) for prefix in print_stats_prefixes_value)
    print_trace_limit = None
    if "print_trace_limit" in payload:
        print_trace_limit = parse_int_like(payload["print_trace_limit"], "print_trace_limit")
    base_address = None
    if "base_address" in payload:
        base_address = parse_int_like(payload["base_address"], "base_address")
    max_cycles = None
    if "max_cycles" in payload:
        max_cycles = parse_int_like(payload["max_cycles"], "max_cycles")
    sweep_sort = None
    if "sweep_sort" in payload:
        sweep_sort = str(payload["sweep_sort"])
    sweep_desc = None
    if "sweep_desc" in payload:
        sweep_desc = bool(payload["sweep_desc"])
    sweep_limit = None
    if "sweep_limit" in payload:
        sweep_limit = parse_int_like(payload["sweep_limit"], "sweep_limit")
    dram_loads: list[tuple[int, Path]] = []
    for entry in payload.get("dram", []):
        if not isinstance(entry, dict):
            raise ValueError("Each DRAM experiment manifest entry must be a JSON object.")
        address = parse_int_like(entry.get("address"), "dram.address")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        dram_loads.append((address, path))
    scratchpad_loads: list[tuple[int, Path]] = []
    for entry in payload.get("scratchpad", []):
        if not isinstance(entry, dict):
            raise ValueError("Each scratchpad experiment manifest entry must be a JSON object.")
        offset = parse_int_like(entry.get("offset"), "scratchpad.offset")
        path = Path(entry.get("path"))
        if not path.is_absolute():
            path = manifest_root / path
        scratchpad_loads.append((offset, path))
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a JSON object when present.")
    output_dir = None
    if "output_dir" in artifacts:
        output_dir_path = Path(str(artifacts["output_dir"]))
        if not output_dir_path.is_absolute():
            output_dir_path = manifest_root / output_dir_path
        output_dir = str(output_dir_path)
    return ExperimentManifest(
        program=program_path,
        base_address=base_address,
        max_cycles=max_cycles,
        config=config,
        report_names=report_names,
        report_limit=report_limit,
        report_match=report_match,
        print_stats_prefixes=print_stats_prefixes,
        print_trace_limit=print_trace_limit,
        sweep_configs=sweep_configs,
        sweep_reports=sweep_reports,
        sweep_sort=sweep_sort,
        sweep_desc=sweep_desc,
        sweep_limit=sweep_limit,
        dram_loads=tuple(dram_loads),
        scratchpad_loads=tuple(scratchpad_loads),
        output_dir=output_dir,
        stats_json=None if "stats_json" not in artifacts else str(artifacts["stats_json"]),
        stats_csv=None if "stats_csv" not in artifacts else str(artifacts["stats_csv"]),
        trace_json=None if "trace_json" not in artifacts else str(artifacts["trace_json"]),
        trace_csv=None if "trace_csv" not in artifacts else str(artifacts["trace_csv"]),
        perfetto_trace=None if "perfetto_trace" not in artifacts else str(artifacts["perfetto_trace"]),
        manifest_json=None if "manifest_json" not in artifacts else str(artifacts["manifest_json"]),
        scratchpad_dump=None if "scratchpad_dump" not in artifacts else str(artifacts["scratchpad_dump"]),
        scratchpad_dump_offset=None
        if "scratchpad_dump_offset" not in artifacts
        else parse_int_like(artifacts["scratchpad_dump_offset"], "artifacts.scratchpad_dump_offset"),
        scratchpad_dump_size=None
        if "scratchpad_dump_size" not in artifacts
        else parse_int_like(artifacts["scratchpad_dump_size"], "artifacts.scratchpad_dump_size"),
        dram_dump=None if "dram_dump" not in artifacts else str(artifacts["dram_dump"]),
        dram_dump_offset=None
        if "dram_dump_offset" not in artifacts
        else parse_int_like(artifacts["dram_dump_offset"], "artifacts.dram_dump_offset"),
        dram_dump_size=None
        if "dram_dump_size" not in artifacts
        else parse_int_like(artifacts["dram_dump_size"], "artifacts.dram_dump_size"),
        sweep_json=None if "sweep_json" not in artifacts else str(artifacts["sweep_json"]),
        sweep_csv=None if "sweep_csv" not in artifacts else str(artifacts["sweep_csv"]),
    )


def load_sweep_experiment_manifest(manifest_path: Path) -> ExperimentManifest:
    """Backward-compatible wrapper for the older sweep-manifest-only loader name."""
    return load_experiment_manifest(manifest_path)


def path_is_under_directory(path: Path, directory: Path) -> bool:
    """Return whether one relative path already begins under the selected output directory."""
    if len(path.parts) < len(directory.parts):
        return False
    return path.parts[: len(directory.parts)] == directory.parts


def prepare_output_path(path_text: str, output_dir: str) -> Path:
    """Create parent directories for one file output target and return the normalized path."""
    path = Path(path_text)
    output_root = Path(output_dir)
    if not path.is_absolute() and not path_is_under_directory(path, output_root):
        path = output_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def decode_program_from_path(
    program_path: Path,
    base_address: int,
    decoder: Decoder,
    repo_root: Path,
):
    """Decode one user-supplied program path, auto-compiling source files first."""
    if program_path.suffix.lower() in SOURCE_SUFFIXES:
        toolchain = resolve_toolchain(repo_root)
        with tempfile.TemporaryDirectory() as temp_dir:
            elf_path = Path(temp_dir) / f"{program_path.stem}.elf"
            compile_to_elf(program_path, elf_path, toolchain.compiler, base_address)
            return decoder.decode_bytes(elf_path.read_bytes(), name=program_path.name)
    return decoder.decode_bytes(
        program_path.read_bytes(),
        base_address=base_address,
        name=program_path.name,
    )
