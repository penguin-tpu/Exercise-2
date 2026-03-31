"""CLI integration tests for `scripts/run_sim.py`."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

from scripts.run_sim import emit_report


class TestRunSimCLI:
    """Verify structured CLI exports from the simulator entrypoint."""

    def test_run_sim_writes_stats_and_trace_json(self) -> None:
        """The CLI should emit JSON exports for stats and retained trace records."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stats_path = temp_path / "stats.json"
            trace_path = temp_path / "trace.json"
            result = subprocess.run(
                [
                    "python3",
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
            stats_csv_path = temp_path / "stats.csv"
            result = subprocess.run(
                [
                    "python3",
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

    def test_run_sim_prints_filtered_stats_and_trace_tail(self) -> None:
        """The CLI should print filtered stat families and a bounded trace tail on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "python3",
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
        """The CLI should print curated summary, latency, occupancy, memory, stall, pipeline, unit, and ISA reports on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "python3",
                str(script),
                "--report",
                "summary",
                "--report",
                "latency",
                "--report",
                "occupancy",
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

        assert "report summary pipeline cycles=3 issued=2 retired=2 total_stalls=0" in result.stdout
        assert "report summary unit=scalar busy_cycles=2 busy_pct=66.67 issued_ops=2" in result.stdout
        assert "report summary latency opcode=addi avg_cycles=1.00 max_cycles=1" in result.stdout
        assert "report summary memory key=none total_bytes=0" in result.stdout
        assert "report summary contention key=none value=0" in result.stdout
        assert "report latency opcode=addi" in result.stdout
        assert "avg_cycles=1.00" in result.stdout
        assert "report occupancy_summary unit=scalar samples=3 avg_depth=0.67 max_depth=1" in result.stdout
        assert "report memory_summary direction=read total_bytes=0" in result.stdout
        assert "report memory_summary direction=write total_bytes=0" in result.stdout
        assert "report contention_summary family=stall total=0" in result.stdout
        assert "report contention_summary family=resource total=0" in result.stdout
        assert "report stalls_summary total=0 categories=0" in result.stdout
        assert "report pipeline cycles=3 issued=2 retired=2 issue_per_cycle=0.67 retire_per_cycle=0.67 total_stalls=0" in result.stdout
        assert "report occupancy unit=scalar depth=1" in result.stdout
        assert "report units unit=scalar issued_ops=2 busy_cycles=2 busy_pct=66.67 max_queue_occupancy=1" in result.stdout
        assert "report isa opcode=addi issued=1 total_cycles=1" in result.stdout

    def test_run_sim_applies_report_limit(self) -> None:
        """The CLI should honor `--report-limit` for multi-row reports."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "python3",
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
                "python3",
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
        }

        emit_report("pipeline", stats)
        captured = capsys.readouterr()

        assert "report pipeline cycles=10 issued=4 retired=3 issue_per_cycle=0.40 retire_per_cycle=0.30 total_stalls=3" in captured.out

    def test_emit_report_prints_top_level_summary(self, capsys: object) -> None:
        """The summary report should surface top pipeline, unit, latency, memory, and contention highlights."""
        stats = {
            "cycles": 10,
            "instructions_issued": 5,
            "instructions_retired": 4,
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
            "stall_fence": 1,
        }

        emit_report("summary", stats)
        captured = capsys.readouterr()

        assert "report summary pipeline cycles=10 issued=5 retired=4 total_stalls=1" in captured.out
        assert "report summary unit=scalar busy_cycles=6 busy_pct=60.00 issued_ops=3" in captured.out
        assert "report summary latency opcode=dma_copy avg_cycles=4.00 max_cycles=5" in captured.out
        assert "report summary memory key=dram.bytes_read total_bytes=64" in captured.out
        assert "report summary contention key=scratchpad.port_conflict.sp_read_port_0 value=3" in captured.out

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
