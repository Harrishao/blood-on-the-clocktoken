"""Configuration loading and provider/model selection."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class ProviderConfig(BaseModel):
    """Connection details for an OpenAI-compatible provider."""

    base_url: str
    api_key_env: str
    reasoning_fields: tuple[str, ...] = ("reasoning_content", "thinking")


class ModelConfig(BaseModel):
    """The provider and model name used for one model call type."""

    provider: str
    name: str


class PlayerModelOverrides(BaseModel):
    """Optional per-player model selections."""

    model: ModelConfig | None = None
    short_model: ModelConfig | None = None


class GameConfig(BaseModel):
    """Deterministic game and scheduler bounds."""

    seed: StrictInt = 17
    player_ids: tuple[str, ...] = ("alice", "bob", "carol", "david", "eve")
    history_directory: Path = Path("history")
    discussion_action_budget: int = Field(default=40, gt=0)
    discussion_quiet_windows: int = Field(default=3, gt=0)
    private_chat_action_budget: int = Field(default=8, gt=0)
    private_chat_quiet_windows: int = Field(default=2, gt=0)

    @field_validator("player_ids")
    @classmethod
    def validate_player_ids(cls, player_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not 5 <= len(player_ids) <= 15:
            raise ValueError("player_ids must contain 5 to 15 entries")
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("player_ids must be unique")
        if any(not player_id.strip() for player_id in player_ids):
            raise ValueError("player_ids must not contain empty entries")
        return player_ids

    @field_validator("history_directory")
    @classmethod
    def validate_history_directory(cls, directory: Path) -> Path:
        if not str(directory).strip() or str(directory) == ".":
            raise ValueError("history_directory must name a directory")
        return directory


class ModelSelections(BaseModel):
    global_model: ModelConfig = Field(alias="global")
    global_short: ModelConfig | None = None

    model_config = ConfigDict(populate_by_name=True)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model selection with the provider settings needed to invoke it."""

    provider: str
    name: str
    base_url: str
    api_key_env: str
    api_key: str | None
    reasoning_fields: tuple[str, ...]
    source: str

    @classmethod
    def from_config(
        cls,
        model: ModelConfig,
        source: str,
        providers: dict[str, ProviderConfig],
    ) -> ResolvedModel:
        try:
            provider = providers[model.provider]
        except KeyError as error:
            raise ValueError(f"Unknown provider: {model.provider}") from error
        return cls(
            provider=model.provider,
            name=model.name,
            base_url=provider.base_url,
            api_key_env=provider.api_key_env,
            api_key=os.getenv(provider.api_key_env),
            reasoning_fields=provider.reasoning_fields,
            source=source,
        )


class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    game: GameConfig = Field(default_factory=GameConfig)
    models: ModelSelections
    players: dict[str, PlayerModelOverrides] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        with path.open("rb") as config_file:
            return cls.model_validate(tomllib.load(config_file))

    def resolve_model(self, player_id: str, short: bool) -> ResolvedModel:
        player = self.players.get(player_id)
        choices = (
            [
                (player.short_model, f"players.{player_id}.short_model") if player else None,
                (self.models.global_short, "models.global_short"),
                (player.model, f"players.{player_id}.model") if player else None,
                (self.models.global_model, "models.global"),
            ]
            if short
            else [
                (player.model, f"players.{player_id}.model") if player else None,
                (self.models.global_model, "models.global"),
            ]
        )
        for choice in choices:
            if choice is not None and choice[0] is not None:
                return ResolvedModel.from_config(choice[0], choice[1], self.providers)
        raise ValueError(f"No {'short' if short else 'normal'} model configured")
