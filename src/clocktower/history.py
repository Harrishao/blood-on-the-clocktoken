from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState, PlayerState
from clocktower.event_stream import EventStream
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

if TYPE_CHECKING:
    from clocktower.domain.state import Notebook


class HistoryWriteError(RuntimeError):
    """A history event could not be durably appended."""


class CheckpointSnapshot(BaseModel):
    trigger_player_id: str
    trigger_event_seq: int
    day: int
    phase: str
    active_scene: str | None
    players: dict[str, PlayerState]
    latest_event_seq: int

    @classmethod
    def from_state(
        cls,
        state: GameState,
        *,
        trigger_player_id: str,
        trigger_event_seq: int,
        latest_event_seq: int,
    ) -> CheckpointSnapshot:
        return cls(
            trigger_player_id=trigger_player_id,
            trigger_event_seq=trigger_event_seq,
            day=state.day,
            phase=state.phase,
            active_scene=state.active_scene,
            players={player_id: player.model_copy(deep=True) for player_id, player in state.players.items()},
            latest_event_seq=latest_event_seq,
        )


class HistoryWriter:
    """UTF-8 JSONL persistence coupled to the authoritative live stream."""

    def __init__(self, path: Path, stream: EventStream) -> None:
        self.path = path
        self.stream = stream
        self._lock = asyncio.Lock()

    async def append(self, event: EventRecord) -> EventRecord:
        async with self._lock:
            return await self._append_locked(event)

    async def update_notebook(self, state: GameState, player_id: str, notebook: Notebook) -> None:
        async with self._lock:
            notebook_event = await self._append_locked(
                EventRecord(
                    phase=state.phase,
                    type="notebook.updated",
                    actor=player_id,
                    audience=Audience.player(player_id),
                    payload={"player_id": player_id, "notebook": notebook.model_dump(mode="json")},
                )
            )
            state.players[player_id].notebook = notebook
            checkpoint = CheckpointSnapshot.from_state(
                state,
                trigger_player_id=player_id,
                trigger_event_seq=notebook_event.seq,
                latest_event_seq=self.stream.next_seq,
            )
            await self._append_locked(
                EventRecord(
                    phase=state.phase,
                    type="checkpoint",
                    audience=Audience.observer(),
                    payload=checkpoint.model_dump(mode="json"),
                )
            )

    async def _append_locked(self, event: EventRecord) -> EventRecord:
        record = self.stream.prepare(event)
        self._write_record(record)
        return await self.stream.publish(record)

    def _write_record(self, record: EventRecord) -> None:
        try:
            line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as history:
                history.write(f"{line}\n")
                history.flush()
        except (OSError, TypeError, ValueError, PydanticSerializationError) as error:
            raise HistoryWriteError(f"could not write history to {self.path}") from error
