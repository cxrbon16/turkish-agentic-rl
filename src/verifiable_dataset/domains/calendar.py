"""Single-day calendar scheduler.

Times are integer minutes-from-midnight within one day for simplicity.
Events live in initial_state["events"]: list of {id, title, start, end}.
"""
from __future__ import annotations

import copy

from verifiable_dataset.base import BaseToolEnv, ToolCallError

TOOLS = [
    {
        "name": "list_events",
        "description": "List all currently scheduled events, sorted by start time.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "find_free_slot",
        "description": (
            "Find the earliest free slot of at least duration_minutes, searching "
            "from day_start_min to day_end_min (minutes from midnight)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "duration_minutes": {"type": "integer"},
                "day_start_min": {"type": "integer"},
                "day_end_min": {"type": "integer"},
            },
            "required": ["duration_minutes", "day_start_min", "day_end_min"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_event",
        "description": "Create an event. Fails if it overlaps an existing event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["title", "start", "end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_event",
        "description": "Cancel the event with the given id.",
        "parameters": {
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
            "additionalProperties": False,
        },
    },
]


class CalendarEnv(BaseToolEnv):
    def __init__(self, initial_state: dict):
        self.events = copy.deepcopy(initial_state.get("events", []))
        self._next_id = initial_state.get("next_id", 1)

    def _overlaps(self, start: int, end: int) -> bool:
        return any(start < e["end"] and e["start"] < end for e in self.events)

    def list_events(self) -> list[dict]:
        return sorted(self.events, key=lambda e: e["start"])

    def find_free_slot(self, duration_minutes: int, day_start_min: int, day_end_min: int) -> int | None:
        busy = sorted(self.events, key=lambda e: e["start"])
        cursor = day_start_min
        for e in busy:
            if e["start"] - cursor >= duration_minutes:
                return cursor
            cursor = max(cursor, e["end"])
        if day_end_min - cursor >= duration_minutes:
            return cursor
        return None

    def create_event(self, title: str, start: int, end: int) -> str:
        if end <= start:
            raise ToolCallError("event end must be after start")
        if self._overlaps(start, end):
            raise ToolCallError(f"event overlaps an existing event: {title} [{start},{end})")
        event_id = f"e{self._next_id}"
        self._next_id += 1
        self.events.append({"id": event_id, "title": title, "start": start, "end": end})
        return event_id

    def cancel_event(self, event_id: str) -> None:
        before = len(self.events)
        self.events = [e for e in self.events if e["id"] != event_id]
        if len(self.events) == before:
            raise ToolCallError(f"no such event: {event_id}")

    def state_dict(self) -> dict:
        events = sorted(copy.deepcopy(self.events), key=lambda e: e["start"])
        # id-agnostic signature so verification doesn't depend on the exact
        # auto-incremented ids a candidate trace happens to produce
        signatures = sorted([e["title"], e["start"], e["end"]] for e in events)
        return {"events": events, "event_signatures": signatures}


def make_env(initial_state: dict) -> CalendarEnv:
    return CalendarEnv(initial_state)
