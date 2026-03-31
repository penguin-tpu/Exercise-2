"""Event queue primitives for deferred operation completion."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable

from perf_modeling.types import Cycle, OpId


@dataclass(order=True)
class CompletionEvent:
    """Scheduled event that commits an in-flight operation."""

    ready_cycle: Cycle
    op_id: OpId
    callback: Callable[[], None] = field(compare=False)
    description: str = field(default="", compare=False)

    def fire(self) -> None:
        """Execute the completion callback for this event."""
        self.callback()


class EventQueue:
    """Min-heap based event queue keyed by completion cycle."""

    def __init__(self) -> None:
        self._events: list[CompletionEvent] = []

    def schedule(self, event: CompletionEvent) -> None:
        """Insert a completion event into the queue."""
        heappush(self._events, event)

    def pop_ready(self, cycle: Cycle) -> list[CompletionEvent]:
        """Remove and return all events that complete at or before the cycle."""
        ready: list[CompletionEvent] = []
        while self._events and self._events[0].ready_cycle <= cycle:
            ready.append(heappop(self._events))
        return ready

    def pending_count(self) -> int:
        """Return the number of queued completion events."""
        return len(self._events)

    def clear(self) -> None:
        """Drop all pending events from the queue."""
        self._events.clear()
