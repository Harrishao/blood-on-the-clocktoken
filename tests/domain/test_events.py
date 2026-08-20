from clocktower.domain.events import Audience, EventRecord, ModelOutputSegment
from clocktower.event_stream import EventStream


def event(*, event_type: str, audience: Audience) -> EventRecord:
    return EventRecord(
        phase="day.discussion",
        type=event_type,
        actor="alice",
        audience=audience,
        payload={},
    )


def test_private_event_is_visible_only_to_recipients():
    private_event = event(
        event_type="chat.private_message",
        audience=Audience.players({"alice", "bob"}),
    )

    assert private_event.visible_to("alice")
    assert private_event.visible_to("bob")
    assert not private_event.visible_to("carol")


def test_observer_event_is_not_visible_to_any_player():
    observer_event = event(
        event_type="model.output_segment",
        audience=Audience.observer(),
    )

    assert not observer_event.visible_to("alice")
    assert not observer_event.visible_to("bob")


async def test_stream_assigns_monotonic_sequences_and_replays_after_cursor():
    stream = EventStream()

    first = await stream.publish(event(event_type="game.header", audience=Audience.public()))
    second = await stream.publish(event(event_type="player.public_message", audience=Audience.public()))

    assert (first.seq, second.seq) == (1, 2)
    assert stream.after(first.seq) == (second,)


def test_model_output_segment_preserves_provider_ordering_fields():
    segment = ModelOutputSegment(
        call_id="call-42",
        player_id="alice",
        call_purpose="formal_action",
        segment_index=0,
        kind="reasoning",
        source_field="reasoning_content",
        text="raw provider text",
    )

    assert segment.model_dump() == {
        "call_id": "call-42",
        "player_id": "alice",
        "call_purpose": "formal_action",
        "segment_index": 0,
        "kind": "reasoning",
        "source_field": "reasoning_content",
        "text": "raw provider text",
        "incomplete": False,
    }
