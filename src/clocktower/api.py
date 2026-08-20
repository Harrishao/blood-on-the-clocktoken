"""Local read-only observer surface plus Stop/Continue controls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import StreamingResponse

from clocktower.event_stream import EventStream
from clocktower.orchestrator import RuntimeStatus


class RuntimeController(Protocol):
    async def request_stop(self) -> None: ...

    async def continue_game(self) -> None: ...

    def status(self) -> RuntimeStatus: ...


async def sse_events(
    request: Any,
    stream: EventStream,
    after_seq: int = 0,
    *,
    finished: Callable[[], bool] | None = None,
) -> AsyncIterator[str]:
    """Replay persisted events after a cursor, then follow the live stream."""

    cursor = after_seq
    while True:
        if await request.is_disconnected():
            return

        records = stream.after(cursor)
        if not records:
            if finished is not None and finished():
                return
            records = await stream.wait_for_after(cursor)

        for record in records:
            if await request.is_disconnected():
                return
            cursor = record.seq
            yield f"data: {record.model_dump_json()}\n\n"


router = APIRouter(prefix="/api")


@router.get("/state", response_model=RuntimeStatus)
async def state(request: Request) -> RuntimeStatus:
    return request.app.state.orchestrator.status()


@router.get("/events")
async def events(
    request: Request,
    after_seq: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    orchestrator: RuntimeController = request.app.state.orchestrator
    stream: EventStream = request.app.state.stream
    return StreamingResponse(
        sse_events(
            request,
            stream,
            after_seq,
            finished=lambda: orchestrator.status().state == "ended",
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/control/stop", status_code=202)
async def stop(request: Request) -> dict[str, str]:
    await request.app.state.orchestrator.request_stop()
    return {"state": "stop_requested"}


@router.post("/control/continue", status_code=202)
async def continue_game(request: Request) -> dict[str, str]:
    await request.app.state.orchestrator.continue_game()
    return {"state": "continue_requested"}


def create_api_app(orchestrator: RuntimeController, stream: EventStream) -> FastAPI:
    """Create the route surface around an already-owned single-game runtime."""

    app = FastAPI()
    app.state.orchestrator = orchestrator
    app.state.stream = stream
    app.include_router(router)
    return app


__all__ = ["create_api_app", "router", "sse_events"]
