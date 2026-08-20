"""Deterministic, auditable choices among rule-provided legal options."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


LegalOption = Any


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One policy choice whose options were already validated by the rules."""

    key: str
    options: tuple[Any, ...]
    reason_code: str = "seeded_legal_choice"


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """The complete audit record for one deterministic policy choice."""

    request_key: str
    options: tuple[Any, ...]
    selected: Any
    reason_code: str


class StorytellerPolicy:
    """Choose by a stable hash of seed, request key, and the complete option set.

    Decisions do not consume a mutable RNG cursor.  An unrelated request can
    therefore never change the result of a later, otherwise identical request.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._decisions: list[DecisionRecord] = []

    @property
    def decisions(self) -> tuple[DecisionRecord, ...]:
        return tuple(self._decisions)

    def choose(self, request: DecisionRequest) -> LegalOption:
        if not request.options:
            raise ValueError("Storyteller decision requires at least one legal option")
        stable_options = _jsonable(request.options)
        material = json.dumps(
            {
                "seed": self.seed,
                "key": request.key,
                "options": stable_options,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(
            request.options
        )
        selected = request.options[index]
        self._decisions.append(
            DecisionRecord(
                request_key=request.key,
                options=deepcopy(request.options),
                selected=deepcopy(selected),
                reason_code=request.reason_code,
            )
        )
        return selected


def decision_jsonable(value: Any) -> Any:
    """Return a stable JSON-compatible representation for event payloads."""

    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported Storyteller option type: {type(value).__name__}")
