from pathlib import Path

import pytest
from pydantic import ValidationError

from clocktower.config import AppConfig


def write_config(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_short_model_prefers_global_short_over_player_normal(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_config(path, """
[providers.main]
base_url = "https://example.test/v1"
api_key_env = "TEST_KEY"
reasoning_fields = ["reasoning_content", "thinking"]
[game]
seed = 17
player_ids = ["alice", "bob", "carol", "david", "eve"]
history_directory = "history"
[models.global]
provider = "main"
name = "normal"
[models.global_short]
provider = "main"
name = "fast"
[players.alice.model]
provider = "main"
name = "alice-normal"
""")

    resolved = AppConfig.load(path).resolve_model("alice", short=True)

    assert (resolved.name, resolved.source) == ("fast", "models.global_short")


def test_player_short_has_highest_short_priority(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_config(path, """
[providers.main]
base_url = "https://example.test/v1"
api_key_env = "TEST_KEY"
[models.global]
provider = "main"
name = "normal"
[models.global_short]
provider = "main"
name = "fast"
[players.alice.model]
provider = "main"
name = "alice-normal"
[players.alice.short_model]
provider = "main"
name = "alice-fast"
""")

    resolved = AppConfig.load(path).resolve_model("alice", short=True)

    assert (resolved.name, resolved.source) == ("alice-fast", "players.alice.short_model")


def test_normal_model_prefers_player_normal_over_global(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_config(path, """
[providers.main]
base_url = "https://example.test/v1"
api_key_env = "TEST_KEY"
[models.global]
provider = "main"
name = "normal"
[players.alice.model]
provider = "main"
name = "alice-normal"
""")

    resolved = AppConfig.load(path).resolve_model("alice", short=False)

    assert (resolved.name, resolved.source) == ("alice-normal", "players.alice.model")


def test_short_model_falls_back_to_global_normal(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_config(path, """
[providers.main]
base_url = "https://example.test/v1"
api_key_env = "TEST_KEY"
[models.global]
provider = "main"
name = "normal"
""")

    resolved = AppConfig.load(path).resolve_model("alice", short=True)

    assert (resolved.name, resolved.source) == ("normal", "models.global")


def test_unknown_model_provider_is_rejected(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_config(path, """
[models.global]
provider = "missing"
name = "normal"
""")

    with pytest.raises(ValueError, match="Unknown provider: missing"):
        AppConfig.load(path).resolve_model("alice", short=False)


def test_game_config_rejects_duplicate_or_out_of_range_player_ids(tmp_path: Path):
    path = tmp_path / "config.toml"
    write_config(path, """
[models.global]
provider = "main"
name = "normal"
[game]
player_ids = ["alice", "alice", "bob", "carol"]
""")

    with pytest.raises(ValidationError):
        AppConfig.load(path)
