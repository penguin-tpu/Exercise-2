"""Perfetto-compatible trace export helpers."""

from __future__ import annotations

from dataclasses import dataclass

from perf_modeling.trace import TraceRecord


PROCESS_ID = 1
"""Stable process identifier used for Perfetto trace export."""


@dataclass(frozen=True)
class PerfettoTrack:
    """One named Perfetto thread/track descriptor."""

    name: str
    tid: int


TRACKS = (
    PerfettoTrack(name="frontend", tid=1),
    PerfettoTrack(name="scalar", tid=2),
    PerfettoTrack(name="load_store", tid=3),
    PerfettoTrack(name="dma", tid=4),
    PerfettoTrack(name="vector", tid=5),
    PerfettoTrack(name="mxu", tid=6),
    PerfettoTrack(name="system", tid=7),
    PerfettoTrack(name="summary", tid=8),
)
"""Stable ordered track list used by the Perfetto export."""


TRACK_IDS = {track.name: track.tid for track in TRACKS}
"""Mapping from logical track name to exported Perfetto thread identifier."""


SCALAR_OPCODES = frozenset(
    {
        "add",
        "addi",
        "sub",
        "and",
        "andi",
        "or",
        "ori",
        "xor",
        "xori",
        "sll",
        "slli",
        "srl",
        "srli",
        "sra",
        "srai",
        "slt",
        "slti",
        "sltu",
        "sltiu",
        "lui",
        "auipc",
        "jal",
        "jalr",
        "beq",
        "bne",
        "blt",
        "bge",
        "bltu",
        "bgeu",
    }
)
"""Opcodes emitted on the scalar track."""


LOAD_STORE_OPCODES = frozenset({"lb", "lbu", "lh", "lhu", "lw", "sb", "sh", "sw", "fence"})
"""Opcodes emitted on the load/store track."""


DMA_OPCODES = frozenset({"dma_copy"})
"""Opcodes emitted on the DMA track."""


VECTOR_OPCODES = frozenset({"vadd", "vmul", "vmax", "vrelu", "vreduce_sum"})
"""Opcodes emitted on the vector track."""


MXU_OPCODES = frozenset({"matmul"})
"""Opcodes emitted on the MXU track."""


SYSTEM_OPCODES = frozenset({"ecall", "ebreak", "mret"})
"""Opcodes emitted on the system track."""


def cycles_to_ns(cycle: int, frequency_hz: float) -> int:
    """Convert one cycle index into a Perfetto timestamp in nanoseconds."""
    if frequency_hz <= 0:
        return cycle
    return int(round((cycle * 1_000_000_000) / frequency_hz))


def opcode_from_message(message: str) -> str:
    """Extract the opcode-like prefix from one trace message."""
    if not message:
        return ""
    return message.split(" ", 1)[0]


def infer_track_name(kind: str, message: str) -> str:
    """Infer the most useful Perfetto track for one trace record."""
    if kind == "trap":
        return "system"
    if kind == "stall":
        if message.startswith("load_store ") or message.startswith("mem_") or message.startswith("sp_"):
            return "load_store"
        if message.startswith("dma "):
            return "dma"
        if message.startswith("vector "):
            return "vector"
        if message.startswith("mxu "):
            return "mxu"
        if message.startswith("scalar "):
            return "scalar"
        return "frontend"
    opcode = opcode_from_message(message)
    if opcode in SCALAR_OPCODES:
        return "scalar"
    if opcode in LOAD_STORE_OPCODES:
        return "load_store"
    if opcode in DMA_OPCODES:
        return "dma"
    if opcode in VECTOR_OPCODES:
        return "vector"
    if opcode in MXU_OPCODES:
        return "mxu"
    if opcode in SYSTEM_OPCODES:
        return "system"
    return "frontend"


def build_metadata_events(program_name: str, config_name: str) -> list[dict[str, object]]:
    """Build the fixed metadata events for one Perfetto trace payload."""
    events: list[dict[str, object]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": PROCESS_ID,
            "tid": 0,
            "args": {"name": "Architect Simulator"},
        },
        {
            "name": "process_labels",
            "ph": "M",
            "pid": PROCESS_ID,
            "tid": 0,
            "args": {"labels": f"program={program_name}, config={config_name}"},
        },
    ]
    for track in TRACKS:
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": PROCESS_ID,
                "tid": track.tid,
                "args": {"name": track.name},
            }
        )
    return events


def build_summary_counter_events(stats: dict[str, int], last_timestamp_ns: int) -> list[dict[str, object]]:
    """Build summary counter events placed at the end of the Perfetto timeline."""
    counter_values = {
        "cycles": stats.get("cycles", 0),
        "instructions_issued": stats.get("instructions_issued", 0),
        "instructions_retired": stats.get("instructions_retired", 0),
        "fetch_stall_cycles": stats.get("fetch_stall_cycles", 0),
        "event_queue.max_pending": stats.get("event_queue.max_pending", 0),
    }
    events: list[dict[str, object]] = []
    for name, value in counter_values.items():
        events.append(
            {
                "name": name,
                "ph": "C",
                "pid": PROCESS_ID,
                "tid": TRACK_IDS["summary"],
                "ts": last_timestamp_ns,
                "args": {"value": value},
            }
        )
    return events


def build_perfetto_trace(
    trace_records: list[TraceRecord],
    stats: dict[str, int],
    frequency_hz: float,
    program_name: str,
    config_name: str,
) -> dict[str, object]:
    """Build one Perfetto-compatible JSON payload from simulator trace records."""
    events = build_metadata_events(program_name, config_name)
    pending_issue_timestamps: dict[tuple[str, str], list[int]] = {}
    last_timestamp_ns = 0
    for record in trace_records:
        timestamp_ns = cycles_to_ns(record.cycle, frequency_hz)
        last_timestamp_ns = max(last_timestamp_ns, timestamp_ns)
        track_name = infer_track_name(record.kind, record.message)
        tid = TRACK_IDS[track_name]
        pending_key = (track_name, record.message)
        if record.kind == "issue":
            queue = pending_issue_timestamps.setdefault(pending_key, [])
            queue.append(timestamp_ns)
            continue
        if record.kind == "complete":
            queue = pending_issue_timestamps.get(pending_key, [])
            if queue:
                start_timestamp_ns = queue.pop(0)
                if not queue:
                    pending_issue_timestamps.pop(pending_key, None)
                duration_ns = max(1, timestamp_ns - start_timestamp_ns)
                events.append(
                    {
                        "name": record.message,
                        "cat": "instruction",
                        "ph": "X",
                        "pid": PROCESS_ID,
                        "tid": tid,
                        "ts": start_timestamp_ns,
                        "dur": duration_ns,
                        "args": {
                            "kind": "instruction",
                            "start_cycle": record.cycle - max(1, duration_ns // max(1, cycles_to_ns(1, frequency_hz))),
                            "end_cycle": record.cycle,
                        },
                    }
                )
                continue
        events.append(
            {
                "name": record.message,
                "cat": record.kind,
                "ph": "i",
                "s": "t",
                "pid": PROCESS_ID,
                "tid": tid,
                "ts": timestamp_ns,
                "args": {"kind": record.kind, "cycle": record.cycle},
            }
        )
    for (track_name, message), queue in pending_issue_timestamps.items():
        tid = TRACK_IDS[track_name]
        for timestamp_ns in queue:
            events.append(
                {
                    "name": message,
                    "cat": "issue",
                    "ph": "i",
                    "s": "t",
                    "pid": PROCESS_ID,
                    "tid": tid,
                    "ts": timestamp_ns,
                    "args": {"kind": "issue_unpaired"},
                }
            )
    events.extend(build_summary_counter_events(stats, last_timestamp_ns))
    return {
        "displayTimeUnit": "ns",
        "traceEvents": events,
    }
