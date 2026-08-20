import json

import pytest

from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import Notebook
from clocktower.event_stream import EventStream
from clocktower.history import HistoryWriteError, HistoryWriter
from tests.builders import sample_game_state


def public_event(event_type: str) -> EventRecord:
    return EventRecord(
        phase="day.discussion",
        type=event_type,
        actor="alice",
        audience=Audience.public(),
        payload={},
    )


async def test_notebook_update_is_immediately_followed_by_checkpoint(tmp_path):
    stream = EventStream()
    writer = HistoryWriter(tmp_path / "game.jsonl", stream)
    state = sample_game_state()

    await writer.update_notebook(state, "alice", Notebook(notes="new"))

    records = [json.loads(line) for line in (tmp_path / "game.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == ["notebook.updated", "checkpoint"]
    assert records[1]["seq"] == records[0]["seq"] + 1
    assert state.players["alice"].notebook.notes == "new"


async def test_append_reopens_utf8_jsonl_and_keeps_sequences_monotonic(tmp_path):
    stream = EventStream()
    writer = HistoryWriter(tmp_path / "game.jsonl", stream)

    first = await writer.append(public_event("game.header"))
    second = await writer.append(public_event("player.public_message"))

    records = [json.loads(line) for line in (tmp_path / "game.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [1, 2]
    assert stream.after(0) == (first, second)


async def test_write_failure_raises_without_publishing_the_event(tmp_path):
    target_directory = tmp_path / "history-directory"
    target_directory.mkdir()
    stream = EventStream()
    writer = HistoryWriter(target_directory, stream)

    with pytest.raises(HistoryWriteError):
        await writer.append(public_event("game.header"))

    assert stream.after(0) == ()


async def test_serialization_failure_raises_history_error_without_publishing(tmp_path):
    stream = EventStream()
    writer = HistoryWriter(tmp_path / "game.jsonl", stream)
    event = public_event("game.header").model_copy(update={"payload": {"unserializable": object()}})

    with pytest.raises(HistoryWriteError):
        await writer.append(event)

    assert stream.after(0) == ()
