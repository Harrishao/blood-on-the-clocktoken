from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clocktower.api import create_api_app, sse_events
from clocktower.domain.events import Audience, EventRecord
from clocktower.event_stream import EventStream
from clocktower.orchestrator import RuntimeStatus


class ControllableOrchestrator:
    def __init__(self, *, runtime_state: str = "running") -> None:
        self.runtime_state = runtime_state
        self.stop_requests = 0
        self.continue_requests = 0

    async def request_stop(self) -> None:
        self.stop_requests += 1
        if self.runtime_state != "ended":
            self.runtime_state = "stopped"

    async def continue_game(self) -> None:
        self.continue_requests += 1
        if self.runtime_state == "stopped":
            self.runtime_state = "running"

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            state=self.runtime_state,
            reason=None,
            phase="day.discussion",
            day=1,
            winner="good" if self.runtime_state == "ended" else None,
            history_path="history/game.jsonl",
        )


class DisconnectProbe:
    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def event(number: int) -> EventRecord:
    return EventRecord(
        phase="day.discussion",
        type="chat.public_message",
        actor="alice",
        audience=Audience.public(),
        payload={"number": number},
    )


def parse_sse_data(chunk: str) -> dict[str, object]:
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    return json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))


def test_state_and_stop_continue_are_the_only_control_routes() -> None:
    orchestrator = ControllableOrchestrator()
    app = create_api_app(orchestrator, EventStream())

    with TestClient(app) as client:
        state = client.get("/api/state")
        assert state.status_code == 200
        assert state.json() == {
            "state": "running",
            "reason": None,
            "phase": "day.discussion",
            "day": 1,
            "winner": None,
            "history_path": "history/game.jsonl",
        }

        assert client.post("/api/control/stop").status_code == 202
        assert client.post("/api/control/stop").status_code == 202
        assert orchestrator.status().state == "stopped"
        assert client.post("/api/control/continue").status_code == 202
        assert client.post("/api/control/continue").status_code == 202
        assert orchestrator.status().state == "running"

        assert client.post("/api/control/step").status_code == 404
        assert client.post("/api/control/speed").status_code == 404
        assert client.post("/api/control/restore").status_code == 404


def test_event_endpoint_replays_only_records_strictly_after_cursor() -> None:
    stream = EventStream()

    async def seed() -> None:
        for number in range(1, 6):
            await stream.publish(event(number))

    asyncio.run(seed())
    app = create_api_app(ControllableOrchestrator(runtime_state="ended"), stream)

    with TestClient(app) as client:
        response = client.get(
            "/api/events?after_seq=4",
            headers={"accept": "text/event-stream"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == f"data: {stream.after(4)[0].model_dump_json()}\n\n"
    assert '"seq":5' in response.text
    assert '"seq":4' not in response.text


@pytest.mark.parametrize("cursor", ["-1", "1.5", "abc"])
def test_event_endpoint_rejects_invalid_after_seq(cursor: str) -> None:
    app = create_api_app(ControllableOrchestrator(runtime_state="ended"), EventStream())
    with TestClient(app) as client:
        assert client.get(f"/api/events?after_seq={cursor}").status_code == 422


async def test_sse_replays_then_waits_for_live_persisted_records() -> None:
    stream = EventStream()
    for number in range(1, 6):
        await stream.publish(event(number))
    request = DisconnectProbe()
    subscription = sse_events(request, stream, after_seq=4)

    replay = parse_sse_data(await anext(subscription))
    assert replay == stream.after(4)[0].model_dump(mode="json")

    live_chunk = asyncio.create_task(anext(subscription))
    await asyncio.sleep(0)
    published = await stream.publish(event(6))
    live = parse_sse_data(await asyncio.wait_for(live_chunk, timeout=1))
    assert live == published.model_dump(mode="json")
    await subscription.aclose()


async def test_sse_stops_cleanly_when_observer_is_disconnected() -> None:
    stream = EventStream()
    await stream.publish(event(1))
    request = DisconnectProbe()
    request.disconnected = True
    subscription = sse_events(request, stream, after_seq=0)

    with pytest.raises(StopAsyncIteration):
        await anext(subscription)


def test_lifespan_owns_one_orchestrator_task_and_static_assets_are_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from clocktower import main

    class LifecycleOrchestrator(ControllableOrchestrator):
        def __init__(self) -> None:
            super().__init__()
            self.started = 0
            self.cancelled = 0
            self.running = asyncio.Event()

        async def run(self) -> None:
            self.started += 1
            self.running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

    orchestrator = LifecycleOrchestrator()
    stream = EventStream()
    config_path = tmp_path / "config.toml"
    missing_dist = tmp_path / "missing-dist"
    build_calls: list[Path] = []

    def build_runtime(path: Path):
        build_calls.append(path)
        return orchestrator, stream

    monkeypatch.setattr(main, "build_runtime", build_runtime)
    app = main.create_application(config_path=config_path, web_dist=missing_dist)

    with TestClient(app) as client:
        assert client.get("/api/state").status_code == 200
        assert client.get("/").status_code == 404
        assert build_calls == [config_path]
        assert orchestrator.started == 1

    assert orchestrator.cancelled == 1
