"""Human-readable reporting helpers for simulator stats."""

from __future__ import annotations


def format_average(total_cycles: int, samples: int) -> str:
    """Format one average value for human-readable reports."""
    if samples <= 0:
        return "0.00"
    return f"{total_cycles / samples:.2f}"


def format_percentage(numerator: int, denominator: int) -> str:
    """Format one percentage value for human-readable reports."""
    if denominator <= 0:
        return "0.00"
    return f"{(numerator * 100) / denominator:.2f}"


def format_per_cycle(count: int, cycles: int) -> str:
    """Format one per-cycle throughput value for human-readable reports."""
    if cycles <= 0:
        return "0.00"
    return f"{count / cycles:.2f}"


def matches_report_filter(value: str, report_match: str | None) -> bool:
    """Return whether one report field matches the optional substring filter."""
    if report_match is None or report_match == "":
        return True
    return report_match.lower() in value.lower()


def build_run_summary(stats: dict[str, int]) -> dict[str, object]:
    """Build one compact summary view from the flattened stats snapshot."""
    cycles = stats.get("cycles", 0)
    issued = stats.get("instructions_issued", 0)
    retired = stats.get("instructions_retired", 0)
    total_stalls = sum(value for key, value in stats.items() if key.startswith("stall_"))
    fetch_stall_cycles = stats.get("fetch_stall_cycles", 0)
    fetch_keys = sorted(key for key in stats if key.startswith("fetch_stall.") and key.endswith("_cycles"))

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
    event_keys = sorted(key for key in stats if key.startswith("event_queue.pending."))
    event_samples = sum(stats[key] for key in event_keys)
    event_weighted_depth = sum(int(key.removeprefix("event_queue.pending.")) * stats[key] for key in event_keys)
    top_fetch_reason = "none"
    top_fetch_cycles = -1
    for key in fetch_keys:
        if stats[key] > top_fetch_cycles:
            top_fetch_reason = key.removeprefix("fetch_stall.").removesuffix("_cycles")
            top_fetch_cycles = stats[key]

    return {
        "pipeline": {
            "cycles": cycles,
            "issued": issued,
            "retired": retired,
            "total_stalls": total_stalls,
            "fetch_stall_cycles": fetch_stall_cycles,
            "fetch_stall_pct": format_percentage(fetch_stall_cycles, cycles),
        },
        "busiest_unit": {
            "name": busiest_unit,
            "busy_cycles": max(busiest_busy_cycles, 0),
            "busy_pct": format_percentage(max(busiest_busy_cycles, 0), cycles),
            "issued_ops": stats.get(f"{busiest_unit}.issued_ops", 0) if busiest_unit != "none" else 0,
        },
        "latency_hotspot": {
            "opcode": top_opcode,
            "avg_cycles": format_average(latency_total_cycles, latency_samples),
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
        "event_queue": {
            "samples": event_samples,
            "avg_pending": format_average(event_weighted_depth, event_samples),
            "max_pending": stats.get("event_queue.max_pending", 0),
        },
        "fetch": {
            "cycles": fetch_stall_cycles,
            "pct": format_percentage(fetch_stall_cycles, cycles),
            "top_reason": top_fetch_reason,
            "top_reason_cycles": max(top_fetch_cycles, 0),
        },
    }


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
        event_queue = summary["event_queue"]
        fetch = summary["fetch"]
        print(
            f"report summary pipeline cycles={pipeline['cycles']} issued={pipeline['issued']} retired={pipeline['retired']} total_stalls={pipeline['total_stalls']} fetch_stall_cycles={pipeline['fetch_stall_cycles']} fetch_stall_pct={pipeline['fetch_stall_pct']}"
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
        print(
            f"report summary events samples={event_queue['samples']} avg_pending={event_queue['avg_pending']} max_pending={event_queue['max_pending']}"
        )
        print(
            f"report summary fetch cycles={fetch['cycles']} pct={fetch['pct']} top_reason={fetch['top_reason']} top_reason_cycles={fetch['top_reason_cycles']}"
        )
        return
    if report_name == "latency":
        rows: list[tuple[int, str, int, int, int]] = []
        sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
        for key in sample_keys:
            opcode = key.removeprefix("latency.").removesuffix(".samples")
            if not matches_report_filter(opcode, report_match):
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
                f"report latency opcode={opcode} samples={samples} total_cycles={total_cycles} max_cycles={max_cycles} avg_cycles={format_average(total_cycles, samples)}"
            )
        return
    if report_name == "occupancy":
        occupancy_keys = sorted(key for key in stats if ".queue_occupancy." in key)
        occupancy_by_unit: dict[str, list[tuple[int, int]]] = {}
        for key in occupancy_keys:
            unit_name, _, depth = key.partition(".queue_occupancy.")
            if not matches_report_filter(unit_name, report_match):
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
                f"report occupancy_summary unit={unit_name} samples={samples} avg_depth={format_average(weighted_depth, samples)} max_depth={max_depth}"
            )
        for key in occupancy_keys:
            unit_name, _, depth = key.partition(".queue_occupancy.")
            print(f"report occupancy unit={unit_name} depth={depth} samples={stats[key]}")
        return
    if report_name == "events":
        event_keys = sorted(key for key in stats if key.startswith("event_queue.pending."))
        event_keys = [key for key in event_keys if matches_report_filter(key, report_match)]
        samples = sum(stats[key] for key in event_keys)
        weighted_depth = sum(int(key.removeprefix("event_queue.pending.")) * stats[key] for key in event_keys)
        max_pending = stats.get("event_queue.max_pending", 0)
        print(
            f"report events_summary samples={samples} avg_pending={format_average(weighted_depth, samples)} max_pending={max_pending}"
        )
        if report_limit is not None:
            event_keys = sorted(event_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in event_keys:
            depth = key.removeprefix("event_queue.pending.")
            print(f"report events pending={depth} samples={stats[key]}")
        return
    if report_name == "fetch":
        fetch_keys = sorted(
            key
            for key in stats
            if key.startswith("fetch_stall.") and key.endswith("_cycles")
        )
        fetch_keys = [key for key in fetch_keys if matches_report_filter(key, report_match)]
        total_cycles = stats.get("fetch_stall_cycles", 0)
        top_reason = "none"
        top_reason_cycles = 0
        for key in fetch_keys:
            if stats[key] > top_reason_cycles:
                top_reason = key.removeprefix("fetch_stall.").removesuffix("_cycles")
                top_reason_cycles = stats[key]
        print(
            f"report fetch_summary cycles={total_cycles} pct={format_percentage(total_cycles, stats.get('cycles', 0))} top_reason={top_reason} top_reason_cycles={top_reason_cycles}"
        )
        if report_limit is not None:
            fetch_keys = sorted(fetch_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in fetch_keys:
            reason = key.removeprefix("fetch_stall.").removesuffix("_cycles")
            print(
                f"report fetch reason={reason} cycles={stats[key]} pct={format_percentage(stats[key], total_cycles)}"
            )
        return
    if report_name == "memory":
        read_keys = sorted(key for key in stats if key.endswith(".bytes_read"))
        write_keys = sorted(key for key in stats if key.endswith(".bytes_written"))
        read_keys = [key for key in read_keys if matches_report_filter(key, report_match)]
        write_keys = [key for key in write_keys if matches_report_filter(key, report_match)]
        total_read = sum(stats[key] for key in read_keys)
        total_write = sum(stats[key] for key in write_keys)
        print(f"report memory_summary direction=read total_bytes={total_read}")
        if report_limit is not None:
            read_keys = sorted(read_keys, key=lambda key: (-stats[key], key))[:report_limit]
            write_keys = sorted(write_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in read_keys:
            print(
                f"report memory key={key} value={stats[key]} pct={format_percentage(stats[key], total_read)}"
            )
        print(f"report memory_summary direction=write total_bytes={total_write}")
        for key in write_keys:
            print(
                f"report memory key={key} value={stats[key]} pct={format_percentage(stats[key], total_write)}"
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
        contention_keys = [key for key in contention_keys if matches_report_filter(key, report_match)]
        resource_total = sum(stats[key] for key in resource_keys)
        print(f"report contention_summary family=stall total={stall_total}")
        print(f"report contention_summary family=resource total={resource_total}")
        if report_limit is not None:
            contention_keys = sorted(contention_keys, key=lambda key: (-stats[key], key))[:report_limit]
        for key in contention_keys:
            if key in resource_keys:
                print(
                    f"report contention key={key} value={stats[key]} pct={format_percentage(stats[key], resource_total)}"
                )
                continue
            print(f"report contention key={key} value={stats[key]}")
        return
    if report_name == "stalls":
        stall_keys = sorted(key for key in stats if key.startswith("stall_"))
        stall_keys = [key for key in stall_keys if matches_report_filter(key, report_match)]
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
        fetch_stall_cycles = stats.get("fetch_stall_cycles", 0)
        print(
            f"report pipeline cycles={cycles} issued={issued} retired={retired} issue_per_cycle={format_per_cycle(issued, cycles)} retire_per_cycle={format_per_cycle(retired, cycles)} total_stalls={total_stalls} fetch_stall_cycles={fetch_stall_cycles} fetch_stall_pct={format_percentage(fetch_stall_cycles, cycles)}"
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
        unit_names = [unit_name for unit_name in unit_names if matches_report_filter(unit_name, report_match)]
        if report_limit is not None:
            unit_names = sorted(
                unit_names,
                key=lambda unit_name: (-stats.get(f"{unit_name}.busy_cycles", 0), unit_name),
            )[:report_limit]
        for unit_name in unit_names:
            busy_cycles = stats.get(f"{unit_name}.busy_cycles", 0)
            print(
                f"report units unit={unit_name} issued_ops={stats.get(f'{unit_name}.issued_ops', 0)} busy_cycles={busy_cycles} busy_pct={format_percentage(busy_cycles, total_cycles)} max_queue_occupancy={stats.get(f'{unit_name}.max_queue_occupancy', 0)}"
            )
        return
    if report_name == "isa":
        rows: list[tuple[int, str, int]] = []
        sample_keys = sorted(key for key in stats if key.startswith("latency.") and key.endswith(".samples"))
        for key in sample_keys:
            opcode = key.removeprefix("latency.").removesuffix(".samples")
            if not matches_report_filter(opcode, report_match):
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
