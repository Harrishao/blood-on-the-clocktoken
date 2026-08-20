from __future__ import annotations

import asyncio

from clocktower.domain.events import EventRecord


class EventStream:
    """The complete, sequenced in-memory event history for one live game."""

    def __init__(self) -> None:
        self._events: list[EventRecord] = []
        self._next_seq = 1
        self._condition = asyncio.Condition()

    @property
    def next_seq(self) -> int:
        return self._next_seq

    async def publish(self, event: EventRecord) -> EventRecord:
        async with self._condition:
            record = self._assign_next(event)
            self._events.append(record)
            self._next_seq += 1
            self._condition.notify_all()
            return record

    def after(self, seq: int) -> tuple[EventRecord, ...]:
        return tuple(event for event in self._events if event.seq > seq)

    async def wait_for_after(self, seq: int) -> tuple[EventRecord, ...]:
        async with self._condition:
            await self._condition.wait_for(lambda: self._next_seq > seq + 1)
            return self.after(seq)

    def prepare(self, event: EventRecord) -> EventRecord:
        """Return the event at the next sequence without publishing it yet."""

        return self._assign_next(event)

    def _assign_next(self, event: EventRecord) -> EventRecord:
        if event.seq not in {0, self._next_seq}:
            raise ValueError(f"expected event sequence {self._next_seq}, got {event.seq}")
        return event.model_copy(update={"seq": self._next_seq})
