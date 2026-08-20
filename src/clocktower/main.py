"""Production assembly for the single-process AI Clocktower service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from clocktower.agents.player import PlayerAgent
from clocktower.api import router
from clocktower.config import AppConfig
from clocktower.event_stream import EventStream
from clocktower.history import HistoryWriter
from clocktower.models.openai_compat import OpenAICompatibleAdapter
from clocktower.orchestrator import GameOrchestrator
from clocktower.rules.engine import RuleEngine


def build_runtime(config_path: Path) -> tuple[GameOrchestrator, EventStream]:
    """Load one game configuration and assemble its only live runtime."""

    config = AppConfig.load(config_path)
    stream = EventStream()
    rules = RuleEngine.start_game(
        config.game.player_ids,
        seed=config.game.seed,
        model_config_snapshot=config.model_dump(mode="json", by_alias=True),
    )
    history_name = datetime.now().astimezone().strftime("game-%Y%m%d-%H%M%S-%f.jsonl")
    history = HistoryWriter(config.game.history_directory / history_name, stream)
    adapter = OpenAICompatibleAdapter()
    current_config = {"value": config}

    def resolve_model(player_id: str, short: bool):
        return current_config["value"].resolve_model(player_id, short=short)

    agents = {
        player_id: PlayerAgent(
            player_id=player_id,
            state_provider=lambda: rules.state,
            resolve_model=resolve_model,
            adapter=adapter,
            history=history,
        )
        for player_id in config.game.player_ids
    }

    def reload_model_config() -> None:
        current_config["value"] = AppConfig.load(config_path)

    orchestrator = GameOrchestrator(
        rules=rules,
        agents=agents,
        history=history,
        game_config=config.game,
        reload_model_config=reload_model_config,
    )
    return orchestrator, stream


def create_application(
    *,
    config_path: Path = Path("config.toml"),
    web_dist: Path = Path("web/dist"),
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        orchestrator, stream = build_runtime(config_path)
        application.state.orchestrator = orchestrator
        application.state.stream = stream
        application.state.orchestrator_task = asyncio.create_task(orchestrator.run())
        try:
            yield
        finally:
            task: asyncio.Task[None] = application.state.orchestrator_task
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    application = FastAPI(lifespan=lifespan)
    application.include_router(router)
    if web_dist.is_dir():
        @application.api_route(
            "/api/{unsupported_path:path}",
            methods=("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"),
            include_in_schema=False,
        )
        async def unsupported_api_route(unsupported_path: str) -> None:
            del unsupported_path
            raise HTTPException(status_code=404, detail="Not Found")

        application.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return application


app = create_application()


__all__ = ["app", "build_runtime", "create_application"]
