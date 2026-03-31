"""CLI integration tests for `scripts/run_sim.py`."""

from __future__ import annotations

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
        """The CLI should print curated latency, occupancy, unit, and ISA reports on request."""
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "run_sim.py"
        result = subprocess.run(
            [
                "python3",
                str(script),
                "--report",
                "latency",
                "--report",
                "occupancy",
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

        assert "report latency opcode=addi" in result.stdout
        assert "avg_cycles=1.00" in result.stdout
        assert "report occupancy_summary unit=scalar samples=3 avg_depth=0.67 max_depth=1" in result.stdout
        assert "report occupancy unit=scalar depth=1" in result.stdout
        assert "report units unit=scalar issued_ops=2 busy_cycles=2 max_queue_occupancy=1" in result.stdout
        assert "report isa opcode=addi issued=1 total_cycles=1" in result.stdout

    def test_emit_report_prints_memory_and_contention_views(self, capsys: object) -> None:
        """The report helper should print curated memory, contention, unit, and ISA views from flat stats."""
        stats = {
            "dram.bytes_read": 32,
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

        assert "report memory key=dram.bytes_read value=32" in captured.out
        assert "report memory key=scratchpad.bytes_written value=16" in captured.out
        assert "report contention key=dram.contention_stalls value=2" in captured.out
        assert "report contention key=scratchpad.bank_conflict.sp_bank_0 value=1" in captured.out
        assert "report units unit=scalar issued_ops=4 busy_cycles=4 max_queue_occupancy=2" in captured.out
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
