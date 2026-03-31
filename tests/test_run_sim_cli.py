"""CLI integration tests for `scripts/run_sim.py`."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

from perf_modeling import AcceleratorConfig
from perf_modeling.config import available_config_names, describe_named_config, get_named_config, snapshot_config
from perf_modeling.reporting import emit_config_report, emit_report


class TestRunSimCLI:
    """Verify structured CLI exports from the simulator entrypoint."""

    def test_run_sim_writes_stats_and_trace_json(self) -> None:
        """The CLI should emit JSON exports for stats and retained trace records."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stats_path = temp_path / "out" / "stats.json"
            trace_path = temp_path / "out" / "trace.json"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--stats-json",
                    str(stats_path),
                    "--trace-json",
                    str(trace_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            stats_payload = json.loads(stats_path.read_text())
            trace_payload = json.loads(trace_path.read_text())

        assert "program=builtin-smoke" in result.stdout
        assert stats_payload["program"] == "builtin-smoke"
        assert stats_payload["config"] == "baseline"
        assert stats_payload["config_snapshot"]["core"]["issue_width"] == 1
        assert stats_payload["config_snapshot"]["scratchpad"]["capacity_bytes"] == 1 << 20
        assert stats_payload["halted"] is True
        assert stats_payload["exit_code"] == 42
        assert stats_payload["stats"]["instructions_retired"] == 2
        assert isinstance(trace_payload, list)
        assert trace_payload
        assert any(record["kind"] == "issue" for record in trace_payload)

    def test_run_sim_writes_stats_csv(self) -> None:
        """The CLI should emit a flat CSV export for stats on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stats_csv_path = temp_path / "out" / "stats.csv"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--stats-csv",
                    str(stats_csv_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            with stats_csv_path.open(newline="") as handle:
                rows = list(csv.reader(handle))

        assert "program=builtin-smoke" in result.stdout
        assert rows[0] == ["key", "value"]
        assert ["instructions_retired", "2"] in rows
        assert ["latency.addi.samples", "1"] in rows

    def test_run_sim_writes_trace_csv(self) -> None:
        """The CLI should emit a CSV export for retained trace records on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace_csv_path = temp_path / "out" / "trace.csv"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--trace-csv",
                    str(trace_csv_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            with trace_csv_path.open(newline="") as handle:
                rows = list(csv.reader(handle))

        assert "program=builtin-smoke" in result.stdout
        assert rows[0] == ["cycle", "kind", "message"]
        assert any(row[1] == "issue" for row in rows[1:])
        assert any(row[1] == "complete" for row in rows[1:])

    def test_run_sim_writes_scratchpad_dump_for_workload_example(self) -> None:
        """The CLI should dump scratchpad results for an assembled workload example."""
        repo_root = Path(__file__).resolve().parent.parent
        assembler = repo_root / "toolchains" / "riscv32" / "assemble_to_elf.py"
        script = repo_root / "scripts" / "run_sim.py"
        source = repo_root / "tests" / "workload" / "scalar_int_matmul.S"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            program_path = temp_path / "scalar_int_matmul.elf"
            dump_path = temp_path / "out" / "results.bin"
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(assembler),
                    str(source),
                    "--output",
                    str(program_path),
                ],
                check=True,
                text=True,
                cwd=repo_root,
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(program_path),
                    "--scratchpad-dump",
                    str(dump_path),
                    "--scratchpad-dump-size",
                    "16",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            payload = dump_path.read_bytes()

        assert "program=scalar_int_matmul.elf" in result.stdout
        assert payload == (
            (19).to_bytes(4, byteorder="little", signed=False)
            + (22).to_bytes(4, byteorder="little", signed=False)
            + (43).to_bytes(4, byteorder="little", signed=False)
            + (50).to_bytes(4, byteorder="little", signed=False)
        )

    def test_run_sim_writes_dram_dump_for_store_program(self) -> None:
        """The CLI should dump DRAM bytes after a program stores through the normal memory path."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "dram_store.S"
            dump_path = temp_path / "out" / "dram.bin"
            source.write_text(
                "\n".join(
                    [
                        ".section .text",
                        ".globl _start",
                        "_start:",
                        "  addi t0, x0, 128",
                        "  lui t1, 0x12345",
                        "  addi t1, t1, 0x678",
                        "  sw t1, 0(t0)",
                        "  ebreak",
                        "",
                    ]
                )
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--dram-dump",
                    str(dump_path),
                    "--dram-dump-offset",
                    "128",
                    "--dram-dump-size",
                    "4",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            payload = dump_path.read_bytes()

        assert "program=dram_store.S" in result.stdout
        assert payload == b"\x78\x56\x34\x12"

    def test_run_sim_preloads_dram_from_image_file(self) -> None:
        """The CLI should preload DRAM bytes before execution via `--dram-load`."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.bin"
            source = temp_path / "dram_load.S"
            dump_path = temp_path / "out" / "roundtrip.bin"
            image_path.write_bytes(b"\x78\x56\x34\x12")
            source.write_text(
                "\n".join(
                    [
                        ".section .text",
                        ".globl _start",
                        "_start:",
                        "  addi t0, x0, 384",
                        "  lw a0, 0(t0)",
                        "  sw a0, 4(t0)",
                        "  ebreak",
                        "",
                    ]
                )
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--dram-load",
                    f"0x180:{image_path}",
                    "--dram-dump",
                    str(dump_path),
                    "--dram-dump-offset",
                    "388",
                    "--dram-dump-size",
                    "4",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            payload = dump_path.read_bytes()

        assert "program=dram_load.S" in result.stdout
        assert payload == b"\x78\x56\x34\x12"

    def test_run_sim_preloads_scratchpad_from_image_file(self) -> None:
        """The CLI should preload scratchpad bytes before execution via `--scratchpad-load`."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.bin"
            source = temp_path / "scratchpad_load.S"
            dump_path = temp_path / "out" / "roundtrip.bin"
            image_path.write_bytes(b"\xef\xbe\xad\xde")
            source.write_text(
                "\n".join(
                    [
                        ".section .text",
                        ".globl _start",
                        "_start:",
                        "  lui t0, 0x20000",
                        "  lw a0, 0(t0)",
                        "  addi t1, x0, 128",
                        "  sw a0, 0(t1)",
                        "  ebreak",
                        "",
                    ]
                )
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--scratchpad-load",
                    f"0x0:{image_path}",
                    "--dram-dump",
                    str(dump_path),
                    "--dram-dump-offset",
                    "128",
                    "--dram-dump-size",
                    "4",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            payload = dump_path.read_bytes()

        assert "program=scratchpad_load.S" in result.stdout
        assert payload == b"\xef\xbe\xad\xde"

    def test_run_sim_preloads_memory_from_json_manifest(self) -> None:
        """The CLI should accept one JSON manifest that seeds both DRAM and scratchpad."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dram_image_path = temp_path / "dram.bin"
            scratchpad_image_path = temp_path / "scratchpad.bin"
            manifest_path = temp_path / "loads.json"
            source = temp_path / "memory_manifest.S"
            dump_path = temp_path / "out" / "roundtrip.bin"
            dram_image_path.write_bytes(b"\x78\x56\x34\x12")
            scratchpad_image_path.write_bytes(b"\xef\xbe\xad\xde")
            manifest_path.write_text(
                json.dumps(
                    {
                        "dram": [
                            {"address": "0x180", "path": dram_image_path.name},
                        ],
                        "scratchpad": [
                            {"offset": "0x0", "path": scratchpad_image_path.name},
                        ],
                    }
                )
            )
            source.write_text(
                "\n".join(
                    [
                        ".section .text",
                        ".globl _start",
                        "_start:",
                        "  lui t0, 0x20000",
                        "  lw a0, 0(t0)",
                        "  addi t1, x0, 128",
                        "  sw a0, 0(t1)",
                        "  addi t2, x0, 384",
                        "  lw a1, 0(t2)",
                        "  sw a1, 4(t1)",
                        "  ebreak",
                        "",
                    ]
                )
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--memory-loads-json",
                    str(manifest_path),
                    "--dram-dump",
                    str(dump_path),
                    "--dram-dump-offset",
                    "128",
                    "--dram-dump-size",
                    "8",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            payload = dump_path.read_bytes()

        assert "program=memory_manifest.S" in result.stdout
        assert payload == b"\xef\xbe\xad\xde\x78\x56\x34\x12"

    def test_run_sim_accepts_assembly_input_directly(self) -> None:
        """The CLI should assemble one source file transiently when passed assembly input."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        source = repo_root / "tests" / "workload" / "scalar_int_matmul.S"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dump_path = temp_path / "out" / "results.bin"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--scratchpad-dump",
                    str(dump_path),
                    "--scratchpad-dump-size",
                    "16",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            payload = dump_path.read_bytes()

        assert "program=scalar_int_matmul.S" in result.stdout
        assert payload == (
            (19).to_bytes(4, byteorder="little", signed=False)
            + (22).to_bytes(4, byteorder="little", signed=False)
            + (43).to_bytes(4, byteorder="little", signed=False)
            + (50).to_bytes(4, byteorder="little", signed=False)
        )

    def test_run_sim_defaults_bare_relative_outputs_under_out_directory(self) -> None:
        """Bare relative output filenames should be materialized under `out/` by default."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        source = repo_root / "tests" / "workload" / "scalar_int_matmul.S"
        stats_path = repo_root / "out" / "test-default-stats.json"
        trace_path = repo_root / "out" / "test-default-trace.json"
        dump_path = repo_root / "out" / "test-default-results.bin"
        for path in (stats_path, trace_path, dump_path):
            if path.exists():
                path.unlink()
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--stats-json",
                    stats_path.name,
                    "--trace-json",
                    trace_path.name,
                    "--scratchpad-dump",
                    dump_path.name,
                    "--scratchpad-dump-size",
                    "16",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            assert "program=scalar_int_matmul.S" in result.stdout
            assert stats_path.exists()
            assert trace_path.exists()
            assert dump_path.read_bytes() == (
                (19).to_bytes(4, byteorder="little", signed=False)
                + (22).to_bytes(4, byteorder="little", signed=False)
                + (43).to_bytes(4, byteorder="little", signed=False)
                + (50).to_bytes(4, byteorder="little", signed=False)
            )
        finally:
            for path in (stats_path, trace_path, dump_path):
                if path.exists():
                    path.unlink()

    def test_run_sim_routes_relative_outputs_through_custom_output_dir(self) -> None:
        """A custom output directory should own all relative artifact paths."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        source = repo_root / "tests" / "workload" / "scalar_int_matmul.S"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "artifacts"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--output-dir",
                    str(output_dir),
                    "--stats-json",
                    "stats.json",
                    "--trace-json",
                    "trace.json",
                    "--scratchpad-dump",
                    "results.bin",
                    "--scratchpad-dump-size",
                    "16",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            stats_path = output_dir / "stats.json"
            trace_path = output_dir / "trace.json"
            dump_path = output_dir / "results.bin"

            assert "program=scalar_int_matmul.S" in result.stdout
            assert stats_path.exists()
            assert trace_path.exists()
            assert dump_path.read_bytes() == (
                (19).to_bytes(4, byteorder="little", signed=False)
                + (22).to_bytes(4, byteorder="little", signed=False)
                + (43).to_bytes(4, byteorder="little", signed=False)
                + (50).to_bytes(4, byteorder="little", signed=False)
            )

    def test_run_sim_writes_manifest_for_generated_artifacts(self) -> None:
        """The CLI should emit a machine-readable manifest for generated artifacts."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        source = repo_root / "tests" / "workload" / "scalar_int_matmul.S"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    str(source),
                    "--output-dir",
                    str(output_dir),
                    "--stats-json",
                    "stats.json",
                    "--trace-json",
                    "trace.json",
                    "--scratchpad-dump",
                    "results.bin",
                    "--scratchpad-dump-size",
                    "16",
                    "--manifest-json",
                    "manifest.json",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            manifest_path = output_dir / "manifest.json"
            manifest_payload = json.loads(manifest_path.read_text())

        assert "program=scalar_int_matmul.S" in result.stdout
        assert manifest_payload["program"] == "scalar_int_matmul.S"
        assert manifest_payload["config"] == "baseline"
        assert manifest_payload["config_snapshot"]["core"]["vector"]["lanes"] == 16
        assert manifest_payload["config_snapshot"]["dram"]["bytes_per_cycle"] == 32
        assert manifest_payload["halted"] is True
        assert manifest_payload["exit_code"] == 50
        assert manifest_payload["manifest"] == str(manifest_path.resolve())
        assert manifest_payload["summary"]["pipeline"]["cycles"] == manifest_payload["cycles"]
        assert manifest_payload["summary"]["pipeline"]["retired"] > 0
        assert "fetch_stall_cycles" in manifest_payload["summary"]["pipeline"]
        assert "top_reason" in manifest_payload["summary"]["fetch"]
        assert manifest_payload["summary"]["busiest_unit"]["name"] != ""
        assert manifest_payload["summary"]["latency_hotspot"]["opcode"] != ""
        assert "max_pending" in manifest_payload["summary"]["event_queue"]
        assert manifest_payload["artifacts"]["stats_json"] == str((output_dir / "stats.json").resolve())
        assert manifest_payload["artifacts"]["trace_json"] == str((output_dir / "trace.json").resolve())
        assert manifest_payload["artifacts"]["scratchpad_dump"] == str((output_dir / "results.bin").resolve())

    def test_run_sim_selects_named_config_preset(self) -> None:
        """The CLI should allow selecting one of the packaged hardware config presets."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stats_path = temp_path / "out" / "stats.json"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--config",
                    "tiny_debug",
                    "--stats-json",
                    str(stats_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            stats_payload = json.loads(stats_path.read_text())

        assert "program=builtin-smoke" in result.stdout
        assert stats_payload["config"] == "tiny_debug"
        assert stats_payload["config_snapshot"]["core"]["vector"]["lanes"] == 4
        assert stats_payload["config_snapshot"]["scratchpad"]["capacity_bytes"] == 1 << 16
        assert stats_payload["stats"]["cycles"] > 3

    def test_named_config_presets_are_exposed(self) -> None:
        """The package should expose stable named configuration presets for CLI selection."""
        assert available_config_names() == ("baseline", "tiny_debug", "balanced_ml", "throughput_ml")
        assert get_named_config("baseline") == AcceleratorConfig()
        assert "debug" in describe_named_config("tiny_debug")
        assert get_named_config("throughput_ml").core.vector.lanes > get_named_config("baseline").core.vector.lanes

    def test_run_sim_lists_named_config_presets(self) -> None:
        """The CLI should print the available named config presets without running a program."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--list-configs",
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "config name=baseline description=" in result.stdout
        assert "config name=tiny_debug description=" in result.stdout
        assert "program=" not in result.stdout

    def test_run_sim_sweeps_one_program_across_named_configs(self) -> None:
        """The CLI should run one workload across multiple named configs and export ordered sweep artifacts."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sweep_path = temp_path / "out" / "sweep.json"
            sweep_csv_path = temp_path / "out" / "sweep.csv"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--sweep-config",
                    "baseline",
                    "--sweep-config",
                    "tiny_debug",
                    "--sweep-sort",
                    "cycles",
                    "--sweep-desc",
                    "--sweep-json",
                    str(sweep_path),
                    "--sweep-csv",
                    str(sweep_csv_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            sweep_payload = json.loads(sweep_path.read_text())
            with sweep_csv_path.open(newline="") as handle:
                sweep_rows = list(csv.reader(handle))

        assert "sweep program=builtin-smoke configs=2" in result.stdout
        assert "sweep rank=1 config=tiny_debug sort=cycles sort_value=4" in result.stdout
        assert "sweep rank=2 config=baseline sort=cycles sort_value=3" in result.stdout
        assert "\nprogram=" not in result.stdout
        assert sweep_payload["program"] == "builtin-smoke"
        assert sweep_payload["sort"] == {"field": "cycles", "descending": True}
        assert [entry["config"] for entry in sweep_payload["results"]] == ["tiny_debug", "baseline"]
        assert sweep_payload["results"][0]["config_snapshot"]["core"]["vector"]["lanes"] == 4
        assert sweep_payload["results"][1]["config_snapshot"]["core"]["vector"]["lanes"] == 16
        assert sweep_payload["results"][0]["cycles"] > sweep_payload["results"][1]["cycles"]
        assert sweep_payload["results"][0]["summary"]["pipeline"]["retired"] == 2
        assert sweep_rows[0][:6] == ["config", "cycles", "issued", "retired", "halted", "exit_code"]
        assert sweep_rows[1][0] == "tiny_debug"
        assert sweep_rows[2][0] == "baseline"
        assert int(sweep_rows[1][1]) > int(sweep_rows[2][1])

    def test_run_sim_rejects_single_run_reports_during_config_sweep(self) -> None:
        """The CLI should reject ambiguous single-run report flags during multi-config sweep mode."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--sweep-config",
                "baseline",
                "--report",
                "summary",
            ],
            check=False,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert result.returncode != 0
        assert "--report is not supported together with --sweep-config" in result.stderr

    def test_run_sim_requires_sweep_config_for_sweep_csv(self) -> None:
        """The CLI should reject sweep CSV export when sweep mode is not enabled."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--sweep-csv",
                "-",
            ],
            check=False,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert result.returncode != 0
        assert "--sweep-csv requires at least one --sweep-config entry" in result.stderr

    def test_run_sim_applies_sweep_limit_after_sorting(self) -> None:
        """The CLI should be able to keep only the top-ranked sweep results."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sweep_path = temp_path / "out" / "limited.json"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--sweep-config",
                    "baseline",
                    "--sweep-config",
                    "tiny_debug",
                    "--sweep-sort",
                    "cycles",
                    "--sweep-desc",
                    "--sweep-limit",
                    "1",
                    "--sweep-json",
                    str(sweep_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            sweep_payload = json.loads(sweep_path.read_text())

        assert "sweep rank=1 config=tiny_debug sort=cycles sort_value=4" in result.stdout
        assert "sweep rank=2" not in result.stdout
        assert sweep_payload["limit"] == 1
        assert [entry["config"] for entry in sweep_payload["results"]] == ["tiny_debug"]

    def test_run_sim_loads_grouped_sweep_manifest(self) -> None:
        """The CLI should accept one grouped sweep manifest with a relative program path."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "manifest_sweep.S"
            manifest_path = temp_path / "sweep.json"
            output_path = temp_path / "artifacts" / "manifest-results.json"
            source.write_text(
                "\n".join(
                    [
                        ".section .text",
                        ".globl _start",
                        "_start:",
                        "  addi a0, x0, 7",
                        "  ebreak",
                        "",
                    ]
                )
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "program": source.name,
                        "sweep_configs": ["baseline", "tiny_debug"],
                        "sweep_sort": "cycles",
                        "sweep_desc": True,
                        "sweep_limit": 1,
                        "artifacts": {
                            "output_dir": "artifacts",
                            "sweep_json": "manifest-results.json",
                        },
                    }
                )
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--sweep-manifest-json",
                    str(manifest_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            sweep_payload = json.loads(output_path.read_text())

        assert "sweep program=manifest_sweep.S configs=2" in result.stdout
        assert "sweep rank=1 config=tiny_debug sort=cycles sort_value=4" in result.stdout
        assert sweep_payload["limit"] == 1
        assert [entry["config"] for entry in sweep_payload["results"]] == ["tiny_debug"]

    def test_run_sim_loads_grouped_single_run_manifest(self) -> None:
        """The CLI should accept one grouped single-run manifest with a relative program path."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "manifest_single.S"
            manifest_path = temp_path / "experiment.json"
            stats_path = temp_path / "artifacts" / "stats.json"
            source.write_text(
                "\n".join(
                    [
                        ".section .text",
                        ".globl _start",
                        "_start:",
                        "  addi a0, x0, 7",
                        "  ebreak",
                        "",
                    ]
                )
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "program": source.name,
                        "config": "tiny_debug",
                        "max_cycles": 1000,
                        "artifacts": {
                            "output_dir": "artifacts",
                            "stats_json": "stats.json",
                        },
                    }
                )
            )
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(script),
                    "--experiment-json",
                    str(manifest_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            stats_payload = json.loads(stats_path.read_text())

        assert "program=manifest_single.S" in result.stdout
        assert stats_payload["program"] == "manifest_single.S"
        assert stats_payload["config"] == "tiny_debug"
        assert stats_payload["stats"]["cycles"] > 3

    def test_run_sim_prints_comparative_sweep_reports(self) -> None:
        """The CLI should print comparative sweep summary and delta reports."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--sweep-config",
                "baseline",
                "--sweep-config",
                "tiny_debug",
                "--sweep-sort",
                "cycles",
                "--sweep-desc",
                "--sweep-report",
                "summary",
                "--sweep-report",
                "delta",
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "sweep report summary configs=2 sort=cycles descending=True limit=none" in result.stdout
        assert "sweep report winner config=tiny_debug cycles=4 retired=2 fetch_stall_cycles=2 latency_opcode=addi latency_avg_cycles=2.00" in result.stdout
        assert "sweep report trailing config=baseline cycles=3 retired=2 fetch_stall_cycles=1" in result.stdout
        assert "sweep report delta_reference config=tiny_debug sort=cycles sort_value=4" in result.stdout
        assert "sweep report delta config=baseline ref=tiny_debug cycles_delta=-1 retired_delta=0 fetch_stall_delta=-1 latency_avg_delta=-1.00" in result.stdout

    def test_run_sim_prints_filtered_stats_and_trace_tail(self) -> None:
        """The CLI should print filtered stat families and a bounded trace tail on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--print-stats-prefix",
                "latency.",
                "--print-trace-limit",
                "2",
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "stat[latency.addi.samples]=1" in result.stdout
        assert "stat[latency.addi.total_cycles]=1" in result.stdout
        assert "trace cycle=" in result.stdout

    def test_run_sim_reports_curated_latency_and_occupancy_views(self) -> None:
        """The CLI should print curated summary, latency, occupancy, fetch, memory, stall, pipeline, unit, and ISA reports on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--report",
                "summary",
                "--report",
                "config",
                "--report",
                "latency",
                "--report",
                "occupancy",
                "--report",
                "events",
                "--report",
                "fetch",
                "--report",
                "memory",
                "--report",
                "contention",
                "--report",
                "stalls",
                "--report",
                "pipeline",
                "--report",
                "units",
                "--report",
                "isa",
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "report summary pipeline cycles=3 issued=2 retired=2 total_stalls=0 fetch_stall_cycles=1 fetch_stall_pct=33.33" in result.stdout
        assert "report summary unit=scalar busy_cycles=2 busy_pct=66.67 issued_ops=2" in result.stdout
        assert "report summary latency opcode=addi avg_cycles=1.00 max_cycles=1" in result.stdout
        assert "report summary memory key=none total_bytes=0" in result.stdout
        assert "report summary contention key=none value=0" in result.stdout
        assert "report summary events samples=3 avg_pending=0.67 max_pending=1" in result.stdout
        assert "report summary fetch cycles=1 pct=33.33 top_reason=system top_reason_cycles=1" in result.stdout
        assert "report config_summary name=baseline fields=" in result.stdout
        assert "report config field=core.issue_width value=1" in result.stdout
        assert "report config field=core.vector.lanes value=16" in result.stdout
        assert "report latency opcode=addi" in result.stdout
        assert "avg_cycles=1.00" in result.stdout
        assert "report occupancy_summary unit=scalar samples=3 avg_depth=0.67 max_depth=1" in result.stdout
        assert "report events_summary samples=3 avg_pending=0.67 max_pending=1" in result.stdout
        assert "report fetch_summary cycles=1 pct=33.33 top_reason=system top_reason_cycles=1" in result.stdout
        assert "report fetch reason=system cycles=1 pct=100.00" in result.stdout
        assert "report memory_summary direction=read total_bytes=0" in result.stdout
        assert "report memory_summary direction=write total_bytes=0" in result.stdout
        assert "report contention_summary family=stall total=0" in result.stdout
        assert "report contention_summary family=resource total=0" in result.stdout
        assert "report stalls_summary total=0 categories=0" in result.stdout
        assert "report pipeline cycles=3 issued=2 retired=2 issue_per_cycle=0.67 retire_per_cycle=0.67 total_stalls=0 fetch_stall_cycles=1 fetch_stall_pct=33.33" in result.stdout
        assert "report occupancy unit=scalar depth=1" in result.stdout
        assert "report events pending=1 samples=2" in result.stdout
        assert "report units unit=scalar issued_ops=2 busy_cycles=2 busy_pct=66.67 max_queue_occupancy=1" in result.stdout
        assert "report isa opcode=addi issued=1 total_cycles=1" in result.stdout

    def test_run_sim_applies_report_limit(self) -> None:
        """The CLI should honor `--report-limit` for multi-row reports."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--report",
                "latency",
                "--report-limit",
                "1",
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "report latency opcode=addi" in result.stdout
        assert "report latency opcode=ecall" not in result.stdout

    def test_run_sim_applies_report_match(self) -> None:
        """The CLI should honor `--report-match` for multi-row reports."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(script),
                "--report",
                "latency",
                "--report-match",
                "ecall",
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "report latency opcode=ecall" in result.stdout
        assert "report latency opcode=addi" not in result.stdout

    def test_emit_report_prints_memory_and_contention_views(self, capsys: object) -> None:
        """The report helper should print curated memory, contention, unit, and ISA views from flat stats."""
        stats = {
            "cycles": 5,
            "dram.bytes_read": 32,
            "scratchpad.bytes_read": 16,
            "scratchpad.bytes_written": 16,
            "dram.contention_stalls": 2,
            "memory.contention.resource.mem_dram": 2,
            "scratchpad.bank_conflict.sp_bank_0": 1,
            "scratchpad.port_conflict.sp_read_port_0": 3,
            "scalar.issued_ops": 4,
            "scalar.busy_cycles": 4,
            "scalar.max_queue_occupancy": 2,
            "latency.addi.samples": 4,
            "latency.addi.total_cycles": 4,
        }

        emit_report("memory", stats)
        emit_report("contention", stats)
        emit_report("units", stats)
        emit_report("isa", stats)
        captured = capsys.readouterr()

        assert "report memory_summary direction=read total_bytes=48" in captured.out
        assert "report memory key=dram.bytes_read value=32 pct=66.67" in captured.out
        assert "report memory key=scratchpad.bytes_read value=16 pct=33.33" in captured.out
        assert "report memory_summary direction=write total_bytes=16" in captured.out
        assert "report memory key=scratchpad.bytes_written value=16 pct=100.00" in captured.out
        assert "report contention_summary family=stall total=0" in captured.out
        assert "report contention_summary family=resource total=6" in captured.out
        assert "report contention key=dram.contention_stalls value=2" in captured.out
        assert "report contention key=memory.contention.resource.mem_dram value=2 pct=33.33" in captured.out
        assert "report contention key=scratchpad.bank_conflict.sp_bank_0 value=1 pct=16.67" in captured.out
        assert "report contention key=scratchpad.port_conflict.sp_read_port_0 value=3 pct=50.00" in captured.out
        assert "report units unit=scalar issued_ops=4 busy_cycles=4 busy_pct=80.00 max_queue_occupancy=2" in captured.out
        assert "report isa opcode=addi issued=4 total_cycles=4" in captured.out

    def test_emit_config_report_prints_flattened_snapshot(self, capsys: object) -> None:
        """The config report helper should flatten and filter the resolved config snapshot."""
        emit_config_report(
            "tiny_debug",
            snapshot_config(get_named_config("tiny_debug")),
            report_limit=3,
            report_match="vector",
        )

        captured = capsys.readouterr()

        assert "report config_summary name=tiny_debug fields=" in captured.out
        assert "report config field=core.vector.lanes value=4" in captured.out
        assert "report config field=core.vector.max_vector_length value=128" in captured.out
        assert "core.scalar.lanes" not in captured.out

    def test_emit_report_prints_occupancy_summary_from_histogram(self, capsys: object) -> None:
        """The occupancy report should derive average and peak depth from histogram stats."""
        stats = {
            "scalar.queue_occupancy.0": 1,
            "scalar.queue_occupancy.2": 3,
            "scalar.max_queue_occupancy": 2,
        }

        emit_report("occupancy", stats)
        captured = capsys.readouterr()

        assert "report occupancy_summary unit=scalar samples=4 avg_depth=1.50 max_depth=2" in captured.out
        assert "report occupancy unit=scalar depth=0 samples=1" in captured.out
        assert "report occupancy unit=scalar depth=2 samples=3" in captured.out

    def test_emit_report_prints_event_queue_summary_from_histogram(self, capsys: object) -> None:
        """The events report should derive average and peak pending depth from histogram stats."""
        stats = {
            "event_queue.pending.0": 1,
            "event_queue.pending.2": 3,
            "event_queue.max_pending": 2,
        }

        emit_report("events", stats)
        captured = capsys.readouterr()

        assert "report events_summary samples=4 avg_pending=1.50 max_pending=2" in captured.out
        assert "report events pending=0 samples=1" in captured.out
        assert "report events pending=2 samples=3" in captured.out

    def test_emit_report_prints_fetch_stall_summary_from_reason_counters(self, capsys: object) -> None:
        """The fetch report should summarize aggregate and per-reason frontend stall cycles."""
        stats = {
            "cycles": 10,
            "fetch_stall_cycles": 4,
            "fetch_stall.branch_cycles": 3,
            "fetch_stall.system_cycles": 1,
        }

        emit_report("fetch", stats)
        captured = capsys.readouterr()

        assert "report fetch_summary cycles=4 pct=40.00 top_reason=branch top_reason_cycles=3" in captured.out
        assert "report fetch reason=branch cycles=3 pct=75.00" in captured.out
        assert "report fetch reason=system cycles=1 pct=25.00" in captured.out

    def test_emit_report_prints_stall_summary_and_categories(self, capsys: object) -> None:
        """The stall report should emit total and per-category counters."""
        stats = {
            "stall_fence": 2,
            "stall_scalar_dependency": 3,
        }

        emit_report("stalls", stats)
        captured = capsys.readouterr()

        assert "report stalls_summary total=5 categories=2" in captured.out
        assert "report stalls key=stall_fence value=2" in captured.out
        assert "report stalls key=stall_scalar_dependency value=3" in captured.out

    def test_emit_report_prints_pipeline_summary(self, capsys: object) -> None:
        """The pipeline report should summarize cycles, throughput, and total stalls."""
        stats = {
            "cycles": 10,
            "instructions_issued": 4,
            "instructions_retired": 3,
            "stall_fence": 2,
            "stall_scalar_dependency": 1,
            "fetch_stall_cycles": 2,
        }

        emit_report("pipeline", stats)
        captured = capsys.readouterr()

        assert "report pipeline cycles=10 issued=4 retired=3 issue_per_cycle=0.40 retire_per_cycle=0.30 total_stalls=3 fetch_stall_cycles=2 fetch_stall_pct=20.00" in captured.out

    def test_emit_report_prints_top_level_summary(self, capsys: object) -> None:
        """The summary report should surface top pipeline, unit, latency, memory, and contention highlights."""
        stats = {
            "cycles": 10,
            "instructions_issued": 5,
            "instructions_retired": 4,
            "fetch_stall_cycles": 2,
            "scalar.issued_ops": 3,
            "scalar.busy_cycles": 6,
            "dma.issued_ops": 2,
            "dma.busy_cycles": 4,
            "latency.addi.samples": 3,
            "latency.addi.total_cycles": 3,
            "latency.addi.max_cycles": 1,
            "latency.dma_copy.samples": 2,
            "latency.dma_copy.total_cycles": 8,
            "latency.dma_copy.max_cycles": 5,
            "dram.bytes_read": 64,
            "scratchpad.bytes_written": 32,
            "memory.contention.resource.mem_dram": 2,
            "scratchpad.port_conflict.sp_read_port_0": 3,
            "event_queue.pending.0": 1,
            "event_queue.pending.2": 3,
            "event_queue.max_pending": 2,
            "fetch_stall_cycles": 4,
            "fetch_stall.branch_cycles": 3,
            "fetch_stall.jump_cycles": 1,
            "stall_fence": 1,
        }

        emit_report("summary", stats)
        captured = capsys.readouterr()

        assert "report summary pipeline cycles=10 issued=5 retired=4 total_stalls=1 fetch_stall_cycles=4 fetch_stall_pct=40.00" in captured.out
        assert "report summary unit=scalar busy_cycles=6 busy_pct=60.00 issued_ops=3" in captured.out
        assert "report summary latency opcode=dma_copy avg_cycles=4.00 max_cycles=5" in captured.out
        assert "report summary memory key=dram.bytes_read total_bytes=64" in captured.out
        assert "report summary contention key=scratchpad.port_conflict.sp_read_port_0 value=3" in captured.out
        assert "report summary events samples=4 avg_pending=1.50 max_pending=2" in captured.out
        assert "report summary fetch cycles=4 pct=40.00 top_reason=branch top_reason_cycles=3" in captured.out

    def test_emit_report_respects_report_limit(self, capsys: object) -> None:
        """Multi-row reports should honor the optional report-limit argument."""
        stats = {
            "latency.addi.samples": 3,
            "latency.addi.total_cycles": 3,
            "latency.addi.max_cycles": 1,
            "latency.dma_copy.samples": 2,
            "latency.dma_copy.total_cycles": 8,
            "latency.dma_copy.max_cycles": 5,
        }

        emit_report("latency", stats, report_limit=1)
        captured = capsys.readouterr()

        assert "report latency opcode=dma_copy" in captured.out
        assert "report latency opcode=addi" not in captured.out

    def test_emit_report_respects_report_match(self, capsys: object) -> None:
        """Multi-row reports should honor the optional report-match argument."""
        stats = {
            "dram.bytes_read": 64,
            "scratchpad.bytes_written": 32,
            "scratchpad.bytes_read": 16,
        }

        emit_report("memory", stats, report_match="scratchpad")
        captured = capsys.readouterr()

        assert "report memory key=scratchpad.bytes_read value=16 pct=100.00" in captured.out
        assert "report memory key=scratchpad.bytes_written value=32 pct=100.00" in captured.out
        assert "report memory key=dram.bytes_read" not in captured.out
