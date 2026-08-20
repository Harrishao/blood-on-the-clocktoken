from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

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
            record = self._assign_at(event, self._next_seq)
            self._events.append(record)
            self._next_seq += 1
            self._condition.notify_all()
            return record

    async def persist_and_publish(
        self,
        event_factory: Callable[[int], Sequence[EventRecord]],
        persist: Callable[[tuple[EventRecord, ...]], None],
    ) -> tuple[EventRecord, ...]:
        """Persist a contiguous event batch before making it visible to subscribers."""

        async with self._condition:
            events = tuple(event_factory(self._next_seq))
            records = tuple(
                self._assign_at(event, self._next_seq + index)
                for index, event in enumerate(events)
            )
            persist(records)
            self._events.extend(records)
            self._next_seq += len(records)
            if records:
                self._condition.notify_all()
            return records

    def after(self, seq: int) -> tuple[EventRecord, ...]:
        return tuple(event for event in self._events if event.seq > seq)

    async def wait_for_after(self, seq: int) -> tuple[EventRecord, ...]:
        async with self._condition:
            await self._condition.wait_for(lambda: self._next_seq > seq + 1)
            return self.after(seq)

    @staticmethod
    def _assign_at(event: EventRecord, sequence: int) -> EventRecord:
        if event.seq not in {0, sequence}:
            raise ValueError(f"expected event sequence {sequence}, got {event.seq}")
        return event.model_copy(update={"seq": sequence})
