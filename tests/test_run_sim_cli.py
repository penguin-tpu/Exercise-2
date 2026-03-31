"""CLI integration tests for `scripts/run_sim.py`."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


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
        """The CLI should print curated latency and occupancy reports on request."""
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
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )

        assert "report latency opcode=addi" in result.stdout
        assert "avg_cycles=1.00" in result.stdout
        assert "report occupancy unit=scalar depth=1" in result.stdout
