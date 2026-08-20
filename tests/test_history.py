import json

import pytest

from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import Notebook
from clocktower.event_stream import EventStream
from clocktower.history import CheckpointSnapshot, HistoryWriteError, HistoryWriter
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


def test_checkpoint_snapshot_deep_copies_game_level_role_state():
    state = sample_game_state()
    state.role_state.fortune_teller_red_herring = "bob"

    snapshot = CheckpointSnapshot.from_state(
        state,
        trigger_player_id="alice",
        trigger_event_seq=1,
        latest_event_seq=2,
    )
    state.role_state.fortune_teller_red_herring = "carol"

    assert snapshot.role_state.fortune_teller_red_herring == "bob"


async def test_append_reopens_utf8_jsonl_and_keeps_sequences_monotonic(tmp_path):
    stream = EventStream()
    writer = HistoryWriter(tmp_path / "game.jsonl", stream)

    first = await writer.append(public_event("game.header"))
    second = await writer.append(public_event("player.public_message"))

    records = [json.loads(line) for line in (tmp_path / "game.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [1, 2]
    assert stream.after(0) == (first, second)


async def test_append_many_rolls_back_the_entire_batch_when_the_second_line_fails(
    tmp_path,
    monkeypatch,
):
    stream = EventStream()
    history_path = tmp_path / "game.jsonl"
    writer = HistoryWriter(history_path, stream)
    writes = 0
    write_line = writer._write_line

    def fail_second_line_once(*args) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("second event write failed")
        write_line(*args)

    monkeypatch.setattr(writer, "_write_line", fail_second_line_once, raising=False)
    batch = (public_event("batch.first"), public_event("batch.second"))

    with pytest.raises(HistoryWriteError, match="could not write history"):
        await writer.append_many(batch)

    assert stream.after(0) == ()
    assert stream.next_seq == 1
    assert history_path.read_text(encoding="utf-8") == ""

    committed = await writer.append_many(batch)

    assert [event.type for event in committed] == ["batch.first", "batch.second"]
    assert [event.seq for event in committed] == [1, 2]
    assert stream.after(0) == committed


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


async def test_checkpoint_write_failure_rolls_back_notebook_batch(tmp_path, monkeypatch):
    stream = EventStream()
    history_path = tmp_path / "game.jsonl"
    writer = HistoryWriter(history_path, stream)
    state = sample_game_state()
    state.players["alice"].notebook = Notebook(notes="old")
    writes = 0
    write_line = writer._write_line

    def fail_second_line(*args) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("checkpoint write failed")
        write_line(*args)

    monkeypatch.setattr(writer, "_write_line", fail_second_line, raising=False)

    with pytest.raises(HistoryWriteError, match="could not write history"):
        await writer.update_notebook(state, "alice", Notebook(notes="new"))

    assert state.players["alice"].notebook.notes == "old"
    assert stream.after(0) == ()
    assert history_path.read_text(encoding="utf-8") == ""
