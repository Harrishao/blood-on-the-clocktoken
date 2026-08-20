"""Shared causal contract for scheduler event persistence callbacks."""

from __future__ import annotations

from collections.abc import Sequence

from clocktower.domain.events import EventRecord


class EventSinkContractError(RuntimeError):
    """An event sink did not atomically commit the complete input batch."""


def validate_event_sink_result(
    drafts: tuple[EventRecord, ...],
    result: object,
) -> tuple[EventRecord, ...]:
    if not isinstance(result, Sequence):
        raise EventSinkContractError(
            f"event sink returned non-sequence result for {len(drafts)} events"
        )
    committed = tuple(result)
    if len(committed) != len(drafts):
        raise EventSinkContractError(
            f"event sink returned {len(committed)} records for {len(drafts)} events"
        )
    if not all(isinstance(event, EventRecord) for event in committed):
        raise EventSinkContractError("event sink returned a non-EventRecord item")
    return committed


__all__ = ["EventSinkContractError", "validate_event_sink_result"]
