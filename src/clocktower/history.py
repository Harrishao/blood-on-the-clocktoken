from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState, PlayerState, RoleState
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
    role_state: RoleState
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
            role_state=state.role_state.model_copy(deep=True),
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
            records = await self.stream.persist_and_publish(
                lambda _first_seq: (event,),
                self._write_records,
            )
            return records[0]

    async def update_notebook(self, state: GameState, player_id: str, notebook: Notebook) -> None:
        async with self._lock:
            if player_id not in state.players:
                raise ValueError(f"unknown player: {player_id}")
            checkpoint_state = state.model_copy(deep=True)
            checkpoint_state.players[player_id].notebook = notebook.model_copy(deep=True)

            def notebook_batch(first_seq: int) -> tuple[EventRecord, EventRecord]:
                checkpoint = CheckpointSnapshot.from_state(
                    checkpoint_state,
                    trigger_player_id=player_id,
                    trigger_event_seq=first_seq,
                    latest_event_seq=first_seq + 1,
                )
                return (
                    EventRecord(
                        phase=state.phase,
                        type="notebook.updated",
                        actor=player_id,
                        audience=Audience.player(player_id),
                        payload={"player_id": player_id, "notebook": notebook.model_dump(mode="json")},
                    ),
                    EventRecord(
                        phase=state.phase,
                        type="checkpoint",
                        audience=Audience.observer(),
                        payload=checkpoint.model_dump(mode="json"),
                    ),
                )

            await self.stream.persist_and_publish(notebook_batch, self._write_records)
            state.players[player_id].notebook = notebook.model_copy(deep=True)

    def _write_records(self, records: tuple[EventRecord, ...]) -> None:
        try:
            lines = tuple(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
                for record in records
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+", encoding="utf-8", newline="\n") as history:
                history.seek(0, 2)
                batch_start = history.tell()
                try:
                    for line in lines:
                        self._write_line(history, line)
                except (OSError, TypeError, ValueError, PydanticSerializationError):
                    history.seek(batch_start)
                    history.truncate()
                    history.flush()
                    raise
        except (OSError, TypeError, ValueError, PydanticSerializationError) as error:
            raise HistoryWriteError(f"could not write history to {self.path}") from error

    @staticmethod
    def _write_line(history: TextIO, line: str) -> None:
        history.write(f"{line}\n")
        history.flush()
