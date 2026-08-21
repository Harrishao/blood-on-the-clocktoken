# Autonomous AI Clocktower Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, single-game Trouble Brewing simulator in which isolated AI players autonomously play while a browser observer watches raw reasoning, tool calls, rules events, history, and notebook checkpoints.

**Architecture:** A Python/FastAPI modular monolith owns the only live game and appends every fact to one typed event stream and JSONL history file. React/Vite consumes live SSE events or a selected history file through the same view model; deterministic rules and Storyteller policy remain independent of model providers.

**Tech Stack:** Python 3.12+, FastAPI 0.141.1, Pydantic 2.13.4, HTTPX 0.28.1, Uvicorn 0.52.4, pytest 9.1.1, pytest-asyncio 1.4.0, React 19.2.8, Vite 8.2.1, TypeScript 7.0.2, Vitest 4.1.11, Testing Library React 16.3.2.

**Spec:** `docs/superpowers/specs/2026-08-20-ai-clocktower-design.md`

## Global Constraints

- Implement only Trouble Brewing; do not add other scripts, Travellers, Fabled characters, custom scripts, or a generic role-plugin system.
- Run one live game per Python process; do not add a database, message queue, job system, microservices, multi-game manager, or long-term statistics.
- Human controls are Stop and Continue only; checkpoints are read-only and cannot restore or fork a game.
- Persist one append-only JSONL file per game; every successful notebook mutation must be followed immediately by a checkpoint record.
- Use only an OpenAI-compatible Chat Completions adapter; API keys come from environment variables and never enter history files.
- Preserve returned `reasoning_content`, `thinking`, tool calls, tool results, and final content in original order for observer display.
- Never expose observer-only truth, another player's notebook, private-chat content, or another player's reasoning to a player context.
- Use ordinary models for formal game choices and short models only for response/closure probes.
- Backend support floor is Python 3.12; frontend support floor is Node.js 22.
- Use TDD for every behavior, deterministic seeds in all rules/scheduler tests, and scoped commits after every task.

---

## Planned File Structure

```text
pyproject.toml                         Python dependencies and pytest settings
config.example.toml                   Safe, runnable configuration template
src/clocktower/config.py              TOML parsing and four-level model resolution
src/clocktower/domain/events.py       Event, audience, model segment, checkpoint models
src/clocktower/domain/state.py        GameState, PlayerState, Notebook, scene state
src/clocktower/domain/actions.py      Canonical player action intents
src/clocktower/history.py             Atomic JSONL append and checkpoint creation
src/clocktower/event_stream.py        In-memory ordered publication and replay
src/clocktower/rules/setup.py         Trouble Brewing setup counts and role assignment
src/clocktower/rules/voting.py        Nomination, voting, execution, dead-vote rules
src/clocktower/rules/roles/*.py       Focused Trouble Brewing role handlers
src/clocktower/rules/engine.py        Ability dispatch, effects, death and win checks
src/clocktower/storyteller.py         Seeded selection among legal options
src/clocktower/models/protocol.py     Provider-neutral call/segment interfaces
src/clocktower/models/openai_compat.py Chat Completions streaming adapter
src/clocktower/agents/context.py      Per-player authorized prompt projection
src/clocktower/agents/tools.py        Tool schema and action validation bridge
src/clocktower/agents/player.py       Full-call and short-probe lifecycle
src/clocktower/scheduler/scoring.py   Deterministic features and seeded choice
src/clocktower/scheduler/discussion.py Public discussion and bounded termination
src/clocktower/scheduler/private_chat.py Two-player private subscene
src/clocktower/orchestrator.py        Setup/night/day/pause/error game loop
src/clocktower/api.py                 HTTP control, state and SSE routes
src/clocktower/main.py                Config loading, startup and static serving
web/src/types.ts                      Frontend mirror of persisted event envelope
web/src/lib/live.ts                   SSE replay/reconnect client
web/src/lib/history.ts                Browser-local JSONL parser
web/src/components/EventStream.tsx    Ordered single-window conversation view
web/src/components/ModelTrace.tsx     Think/tool/result/final rendering
web/src/components/CheckpointPanel.tsx Read-only checkpoint inspection
web/src/components/Toolbar.tsx        Filters, file open, Stop/Continue
web/src/App.tsx                       Live/history mode composition
tests/                                Backend tests mirroring source responsibilities
tests/builders.py                     Deterministic GameState/event builders shared by tests
web/src/**/*.test.tsx                 Frontend component and integration tests
web/src/test/fixtures.ts              Typed event and history-file fixtures
```

---

### Task 1: Project Scaffold and Model Configuration

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `config.example.toml`
- Create: `src/clocktower/__init__.py`
- Create: `src/clocktower/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: environment variables named by `ProviderConfig.api_key_env`.
- Produces: `AppConfig.load(path: Path) -> AppConfig`; `AppConfig.resolve_model(player_id: str, short: bool) -> ResolvedModel`.

- [ ] **Step 1: Write the failing configuration tests**

```python
from pathlib import Path
from clocktower.config import AppConfig

def test_short_model_prefers_global_short_over_player_normal(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("""
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
""", encoding="utf-8")
    resolved = AppConfig.load(path).resolve_model("alice", short=True)
    assert (resolved.name, resolved.source) == ("fast", "models.global_short")

def test_player_short_has_highest_short_priority(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("""
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
""", encoding="utf-8")
    resolved = AppConfig.load(path).resolve_model("alice", short=True)
    assert (resolved.name, resolved.source) == ("alice-fast", "players.alice.short_model")
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"; .\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

Expected: FAIL during collection because `clocktower.config` does not exist.

- [ ] **Step 3: Add exact package metadata and focused configuration models**

```toml
[project]
name = "blood-on-the-clocktoken"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi==0.141.1",
  "pydantic==2.13.4",
  "httpx==0.28.1",
  "uvicorn==0.52.4",
]
[project.optional-dependencies]
dev = ["pytest==9.1.1", "pytest-asyncio==1.4.0"]
[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
```

Implement `ProviderConfig`, `ModelConfig`, `PlayerModelOverrides`, `GameConfig`, `AppConfig`, and immutable `ResolvedModel`. `GameConfig` validates 5–15 unique player IDs, integer seed, history directory, discussion/private-chat budgets, and defaults that are copied into `config.example.toml`. `resolve_model` must implement exactly:

```python
def resolve_model(self, player_id: str, short: bool) -> ResolvedModel:
    player = self.players.get(player_id)
    choices = (
        [(player.short_model, f"players.{player_id}.short_model") if player else None,
         (self.models.global_short, "models.global_short"),
         (player.model, f"players.{player_id}.model") if player else None,
         (self.models.global_model, "models.global")]
        if short else
        [(player.model, f"players.{player_id}.model") if player else None,
         (self.models.global_model, "models.global")]
    )
    for choice in choices:
        if choice is not None and choice[0] is not None:
            return ResolvedModel.from_config(choice[0], choice[1], self.providers)
    raise ValueError(f"No {'short' if short else 'normal'} model configured")
```

`.gitignore` must include `.venv/`, `__pycache__/`, `.pytest_cache/`, `.superpowers/`, `history/`, `config.toml`, `web/node_modules/`, and `web/dist/`.

- [ ] **Step 4: Run configuration tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

Expected: PASS, including all four fallback paths and rejection of unknown providers.

- [ ] **Step 5: Commit the scaffold**

```powershell
git add .gitignore pyproject.toml config.example.toml src/clocktower tests/test_config.py
git commit -m "chore: scaffold Python service and model config"
```

---

### Task 2: Typed Event Stream, History, and Checkpoints

**Files:**
- Create: `src/clocktower/domain/events.py`
- Create: `src/clocktower/domain/state.py`
- Create: `src/clocktower/event_stream.py`
- Create: `src/clocktower/history.py`
- Create: `tests/builders.py`
- Test: `tests/domain/test_events.py`
- Test: `tests/test_history.py`

**Interfaces:**
- Produces: `EventRecord`, `Audience`, `ModelOutputSegment`, `CheckpointSnapshot`, `EventStream.publish(...)`, `EventStream.after(seq)`, `HistoryWriter.append(event)`, `HistoryWriter.update_notebook(...)`.
- Invariant: `HistoryWriter.update_notebook` writes `notebook.updated` then `checkpoint` under one async lock with adjacent sequence numbers.

- [ ] **Step 1: Write failing ordering and audience tests**

```python
async def test_notebook_update_is_immediately_followed_by_checkpoint(tmp_path):
    stream = EventStream()
    writer = HistoryWriter(tmp_path / "game.jsonl", stream)
    state = sample_game_state()
    await writer.update_notebook(state, "alice", Notebook(notes="new"))
    records = [json.loads(line) for line in (tmp_path / "game.jsonl").read_text().splitlines()]
    assert [r["type"] for r in records] == ["notebook.updated", "checkpoint"]
    assert records[1]["seq"] == records[0]["seq"] + 1

def test_private_event_is_visible_only_to_recipients():
    event = EventRecord(type="chat.private_message", audience=Audience.players({"alice", "bob"}), payload={})
    assert event.visible_to("alice")
    assert not event.visible_to("carol")
```

- [ ] **Step 2: Verify tests fail before models exist**

Run: `.\.venv\Scripts\python.exe -m pytest tests/domain/test_events.py tests/test_history.py -v`

Expected: FAIL importing `clocktower.domain.events`.

- [ ] **Step 3: Implement minimal event and state contracts**

```python
class Audience(BaseModel):
    kind: Literal["public", "players", "player", "observer"]
    player_ids: frozenset[str] = frozenset()

class EventRecord(BaseModel):
    seq: int = 0
    time: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    phase: str
    type: str
    actor: str | None = None
    audience: Audience
    payload: dict[str, Any]

    def visible_to(self, player_id: str) -> bool:
        return self.audience.kind == "public" or player_id in self.audience.player_ids
```

Define `Notebook(notes: str, attention: AttentionState)`, `PlayerState`, and `GameState` with no persistence methods. Add deterministic test constructors `GameState.from_assignments(assignments, dead, alive_count, seed)` and `GameState.from_alignments(alignments, seed)`; production setup must continue to go through Task 3's `build_setup`. `EventStream` owns sequence assignment, the complete in-memory event list for the one live game, and `asyncio.Condition` for subscribers; complete retention guarantees SSE reconnect can replay any earlier sequence without a second store.

Create deterministic builders used by later tests with these exact signatures:

```python
def sample_game_state(*, roles: dict[str, str] | None = None,
                      dead: set[str] | None = None,
                      alive_count: int | None = None) -> GameState:
    assigned = roles or {"alice":"washerwoman", "bob":"chef", "carol":"recluse", "david":"poisoner", "eve":"imp"}
    return GameState.from_assignments(assigned, dead=dead or set(), alive_count=alive_count, seed=17)

def sample_voting_state(*, dead: set[str] | None = None) -> GameState:
    return sample_game_state(dead=dead)

def game_with_roles(**roles: str) -> GameState:
    return sample_game_state(roles=roles)

def game_with_seats(alignments: list[str]) -> GameState:
    return GameState.from_alignments(alignments, seed=17)

def public_claim(*, actor: str, mentions: set[str]) -> EventRecord:
    return EventRecord(phase="day.discussion", type="claim.public", actor=actor,
                       audience=Audience(kind="public"), payload={"mentions": sorted(mentions)})

def private_message(participants: set[str]) -> EventRecord:
    return EventRecord(phase="day.private", type="chat.private_message", actor=None,
                       audience=Audience(kind="players", player_ids=frozenset(participants)), payload={})
```

Each builder must use fixed player IDs `alice`, `bob`, `carol`, `david`, `eve`, fixed seat order, and seed `17`; builders must not call the model adapter.

`HistoryWriter` opens the target with UTF-8, serializes one compact JSON object per line, flushes after each line, and raises `HistoryWriteError` without publishing a partially written event.

- [ ] **Step 4: Run event/history tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/domain/test_events.py tests/test_history.py -v`

Expected: PASS for audience filtering, monotonically increasing sequence numbers, JSONL reopening, adjacent checkpoints, and write-failure propagation.

- [ ] **Step 5: Commit event persistence**

```powershell
git add src/clocktower/domain src/clocktower/event_stream.py src/clocktower/history.py tests/builders.py tests/domain tests/test_history.py
git commit -m "feat: add event stream history and checkpoints"
```

---

### Task 3: Core Trouble Brewing State, Setup, Nominations, and Votes

**Files:**
- Create: `src/clocktower/domain/actions.py`
- Create: `src/clocktower/rules/setup.py`
- Create: `src/clocktower/rules/voting.py`
- Test: `tests/rules/test_setup.py`
- Test: `tests/rules/test_voting.py`

**Interfaces:**
- Produces: `build_setup(player_count, selected_roles, seed) -> SetupResult`; `NominationTracker.nominate(...)`; `NominationTracker.cast_vote(...)`; `NominationTracker.resolve_execution()`.
- Consumes: `GameState`, `PlayerState`, `EventRecord` from Task 2.

- [ ] **Step 1: Write failing setup and voting tests**

```python
@pytest.mark.parametrize("count,expected", [
    (5, (3, 0, 1, 1)), (6, (3, 1, 1, 1)), (7, (5, 0, 1, 1)),
    (8, (5, 1, 1, 1)), (9, (5, 2, 1, 1)), (10, (7, 0, 2, 1)),
    (11, (7, 1, 2, 1)), (12, (7, 2, 2, 1)), (13, (9, 0, 3, 1)),
    (14, (9, 1, 3, 1)), (15, (9, 2, 3, 1)),
])
def test_official_setup_counts(count, expected):
    assert setup_counts(count) == expected

def test_tie_clears_execution_candidate():
    tracker = NominationTracker(alive_count=7)
    tracker.record_tally("alice", 4)
    tracker.record_tally("bob", 4)
    assert tracker.resolve_execution() is None

def test_dead_player_spends_only_remaining_vote():
    state = sample_voting_state(dead={"bob"})
    cast_vote(state, "bob", "nom-1", True)
    with pytest.raises(IllegalAction, match="dead vote already spent"):
        cast_vote(state, "bob", "nom-2", True)
```

- [ ] **Step 2: Verify focused failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules/test_setup.py tests/rules/test_voting.py -v`

Expected: FAIL because setup and voting modules are missing.

- [ ] **Step 3: Implement official setup and voting primitives**

```python
SETUP_COUNTS = {
    5:(3,0,1,1), 6:(3,1,1,1), 7:(5,0,1,1), 8:(5,1,1,1),
    9:(5,2,1,1), 10:(7,0,2,1), 11:(7,1,2,1), 12:(7,2,2,1),
    13:(9,0,3,1), 14:(9,1,3,1), 15:(9,2,3,1),
}

def qualifying_tally(votes: int, alive_count: int) -> bool:
    return votes >= math.ceil(alive_count / 2)
```

Enforce alive-only nomination, one nomination per player per day, one nomination of each player per day, clockwise vote order ending with the nominee, unlimited alive votes, one lifetime dead vote, strict-highest execution, and no execution on a tie.

`build_setup` must reject counts outside 5–15, enforce the category counts after Baron's `+2 Outsiders`, and assign seats with `random.Random(seed)` only.

- [ ] **Step 4: Run the complete setup/voting matrix**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules/test_setup.py tests/rules/test_voting.py -v`

Expected: PASS, including nomination limits, threshold rounding, tie, and dead-vote consumption.

- [ ] **Step 5: Commit core day rules**

```powershell
git add src/clocktower/domain/actions.py src/clocktower/rules/setup.py src/clocktower/rules/voting.py tests/rules
git commit -m "feat: add Trouble Brewing setup and voting rules"
```

---

### Task 4: Information and Registration Role Handlers

**Files:**
- Create: `src/clocktower/rules/roles/base.py`
- Create: `src/clocktower/rules/roles/information.py`
- Create: `src/clocktower/rules/roles/registration.py`
- Test: `tests/rules/roles/test_information.py`
- Test: `tests/rules/roles/test_registration.py`

**Interfaces:**
- Produces: `AbilityContext`, `Observation`, `RoleHandler`; handlers for Washerwoman, Librarian, Investigator, Chef, Empath, Fortune Teller, Undertaker, Ravenkeeper, Spy, and Recluse.
- Registration API: `registrations_for(player_id, query, context) -> tuple[Registration, ...]`.

- [ ] **Step 1: Write failing table-driven role tests**

```python
def test_empath_counts_alive_neighbours_only():
    game = game_with_seats(["good", "evil", "good", "evil"])
    game.players["bob"].alive = False
    assert Empath().observe(game, actor_seat=0).number == 1

def test_fortune_teller_red_herring_registers_as_demon():
    game = game_with_roles(alice="fortune_teller", bob="townsfolk", carol="imp")
    game.role_state.fortune_teller_red_herring = "bob"
    assert FortuneTeller().choose(game, "alice", ["bob", "alice"]).yes is True

def test_recluse_can_register_evil_for_investigator_when_policy_selects_it():
    options = registrations_for("recluse", RegistrationQuery.ALIGNMENT, AbilityContext.from_state(game_with_roles(alice="recluse"), "alice"))
    assert {o.alignment for o in options} == {"good", "evil"}
```

- [ ] **Step 2: Confirm role handlers are absent**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules/roles/test_information.py tests/rules/roles/test_registration.py -v`

Expected: FAIL importing role modules.

- [ ] **Step 3: Implement role handlers through legal observations**

```python
class RoleHandler(Protocol):
    role: str
    first_night_order: int | None
    other_night_order: int | None
    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]: ...
    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]: ...
```

Implement exact Trouble Brewing behavior:

- Washerwoman/Librarian/Investigator: one correct character among two players; Librarian may learn zero Outsiders.
- Chef: number of adjacent evil pairs around the circular seating order.
- Empath: evil alive neighbours only.
- Fortune Teller: selected pair yields yes for Demon, red herring, or allowed Recluse registration.
- Undertaker: learns the character of the player executed today, subject to Spy/Recluse registration.
- Ravenkeeper: if killed at night, chooses a player and learns a legal registered character.
- Spy: receives the complete Grimoire snapshot as a private observation; it never enters another player context.
- Recluse/Spy: return legal registration alternatives; deterministic Storyteller policy chooses one when an ability queries them.
- Drunk/poisoned information: generate all rule-legal false observations and delegate the choice to Storyteller policy rather than embedding a lie in the handler.

- [ ] **Step 4: Run role and information-leak tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules/roles/test_information.py tests/rules/roles/test_registration.py -v`

Expected: PASS for seating edges, dead-neighbour skipping, false-information option sets, Spy observer projection, and registration alternatives.

- [ ] **Step 5: Commit information roles**

```powershell
git add src/clocktower/rules/roles tests/rules/roles
git commit -m "feat: implement Trouble Brewing information roles"
```

---

### Task 5: Action, Protection, Outsider, Minion, and Demon Roles

**Files:**
- Create: `src/clocktower/rules/roles/action.py`
- Create: `src/clocktower/rules/roles/outsiders.py`
- Create: `src/clocktower/rules/roles/evil.py`
- Test: `tests/rules/roles/test_action.py`
- Test: `tests/rules/roles/test_outsiders.py`
- Test: `tests/rules/roles/test_evil.py`

**Interfaces:**
- Produces handlers for Monk, Virgin, Slayer, Soldier, Mayor, Butler, Drunk, Saint, Poisoner, Scarlet Woman, Baron, and Imp.
- Consumes `RoleHandler`, `RuleEffect`, registration choices, and setup counts from Tasks 3–4.

- [ ] **Step 1: Write failing interaction tests**

```python
def test_virgin_executes_first_townsfolk_nominator_and_ends_day():
    game = game_with_roles(alice="washerwoman", bob="virgin", eve="imp")
    effects = Virgin().on_nominated(AbilityContext.for_nomination(game, nominator="alice", nominee="bob"))
    assert effects == [Execute("nominator"), EndDay(reason="virgin")]

def test_scarlet_woman_becomes_imp_when_five_or_more_alive():
    game = sample_game_state(roles={"imp":"imp", "scarlet":"scarlet_woman"}, alive_count=7)
    kill(game, "imp")
    assert game.players["scarlet"].role == "imp"

def test_saint_execution_is_evil_win():
    result = resolve_execution(game_with_roles(alice="saint"), "alice")
    assert result.winner == "evil"
```

- [ ] **Step 2: Verify handlers do not exist**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules/roles/test_action.py tests/rules/roles/test_outsiders.py tests/rules/roles/test_evil.py -v`

Expected: FAIL importing the new modules.

- [ ] **Step 3: Implement the remaining role effects**

Use focused effect types such as `Protect`, `Poison`, `Kill`, `Execute`, `TransformRole`, `RedirectDeath`, `MarkUsed`, and `DeclareWinner`.

Implement:

- Monk protects one other player until dawn.
- Virgin triggers only on first nomination while healthy and executes a Townsfolk nominator immediately.
- Slayer has one public use; a legal Demon registration dies when healthy.
- Soldier cannot die from Demon attack while healthy.
- Mayor may redirect night death to a legal alternative selected by Storyteller policy and wins with exactly three alive and no execution.
- Butler chooses a master nightly and may vote only when the master is voting.
- Drunk occupies an Outsider slot, receives a not-in-play Townsfolk identity, and has no ability.
- Saint execution declares evil victory.
- Poisoner selects one player each night; poison lasts through the next day.
- Scarlet Woman becomes Imp when the Demon dies with at least five alive.
- Baron modifies setup by +2 Outsiders and -2 Townsfolk.
- Imp kills nightly; self-kill transfers the Imp role to a living Minion; Demon death without a valid continuation gives good victory.

- [ ] **Step 4: Run all remaining role tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules/roles -v`

Expected: PASS for sober/healthy and drunk/poisoned branches, once-per-game state, Mayor redirect, Butler vote gating, Baron setup, and Imp transfer.

- [ ] **Step 5: Commit remaining roles**

```powershell
git add src/clocktower/rules/roles tests/rules/roles
git commit -m "feat: implement remaining Trouble Brewing roles"
```

---

### Task 6: Deterministic Storyteller and Complete Rule Engine

**Files:**
- Create: `src/clocktower/storyteller.py`
- Create: `src/clocktower/rules/night.py`
- Create: `src/clocktower/rules/engine.py`
- Test: `tests/test_storyteller.py`
- Test: `tests/rules/test_engine.py`

**Interfaces:**
- Produces: `StorytellerPolicy.choose(request) -> LegalOption`; `RuleEngine.start_game(...)`; `RuleEngine.apply_action(...)`; `RuleEngine.advance_night_step(...)`; `RuleEngine.check_winner(...)`.

- [ ] **Step 1: Write failing deterministic and lifecycle tests**

```python
def test_storyteller_same_seed_and_request_is_reproducible():
    request = DecisionRequest(key="false-info:1", options=("a", "b", "c"))
    assert StorytellerPolicy(44).choose(request) == StorytellerPolicy(44).choose(request)

def test_virgin_execution_ends_day_before_more_nominations():
    engine = RuleEngine.for_test({"alice":"washerwoman", "bob":"virgin", "eve":"imp"}, seed=17)
    effects = engine.apply_action(Nominate(actor="alice", target="bob", accusation="test"))
    assert engine.state.phase == "night"
    assert any(e.type == "execution.resolved" for e in effects)
```

- [ ] **Step 2: Confirm engine failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_storyteller.py tests/rules/test_engine.py -v`

Expected: FAIL importing `StorytellerPolicy` and `RuleEngine`.

- [ ] **Step 3: Implement one authoritative mutation boundary**

```python
class RuleEngine:
    def apply_action(self, action: PlayerAction) -> list[EventRecord]:
        self._validate_phase_and_actor(action)
        effects = self._dispatch(action)
        self._apply_effects_atomically(effects)
        events = self._events_for(effects)
        winner = self.check_winner()
        return events + ([self._winner_event(winner)] if winner else [])
```

All handlers return effects; only `RuleEngine._apply_effects_atomically` mutates `GameState`. Define `RuleEngine.for_test(assignments, seed)` as a deterministic constructor around `GameState.from_assignments`; production setup still uses `start_game`. Define explicit first-night and other-night order arrays for all Trouble Brewing roles. Storyteller decisions must log request key, legal options, selected option, and reason code. `start_game` must publish `game.header` before any setup fact, and game completion must publish exactly one `game.ended` record.

- [ ] **Step 4: Run the entire backend rules suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/rules tests/test_storyteller.py -v`

Expected: PASS with no model calls and fixed-seed reproducibility.

- [ ] **Step 5: Commit the complete rules engine**

```powershell
git add src/clocktower/storyteller.py src/clocktower/rules tests/test_storyteller.py tests/rules
git commit -m "feat: complete deterministic Trouble Brewing engine"
```

---

### Task 7: OpenAI-Compatible Streaming Model Adapter

**Files:**
- Create: `src/clocktower/models/protocol.py`
- Create: `src/clocktower/models/openai_compat.py`
- Create: `tests/models/fakes.py`
- Test: `tests/models/test_openai_compat.py`

**Interfaces:**
- Produces: `ModelRequest`, `ModelSegment`, `ModelAdapter.stream(request) -> AsyncIterator[ModelSegment]`.
- Preserves segment kinds `reasoning`, `tool_call`, `tool_result`, `final_message`, `provider_metadata` and exact `source_field`.

- [ ] **Step 1: Write failing streaming parser tests**

```python
async def test_reasoning_tool_and_content_order_is_preserved(mock_transport):
    mock_transport.stream([
        delta(reasoning_content="Think A"),
        delta(tool_calls=[tool_delta("update_notebook", '{"notes":"x"}')]),
        delta(reasoning_content="Think B"),
        delta(content="Public text"),
    ])
    segments = [s async for s in adapter(mock_transport).stream(sample_request())]
    assert [(s.kind, s.source_field, s.text) for s in segments] == [
        ("reasoning", "reasoning_content", "Think A"),
        ("tool_call", "tool_calls", '{"notes":"x"}'),
        ("reasoning", "reasoning_content", "Think B"),
        ("final_message", "content", "Public text"),
    ]
```

`tests/models/fakes.py` must define `ScriptedSSETransport.stream(chunks)`, `delta(**fields)`, `tool_delta(name, arguments)`, and `sample_request()`; the `mock_transport` fixture returns `ScriptedSSETransport` and never opens a socket.

- [ ] **Step 2: Confirm adapter import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/models/test_openai_compat.py -v`

Expected: FAIL importing `clocktower.models.openai_compat`.

- [ ] **Step 3: Implement direct HTTPX/SSE parsing**

```python
@dataclass(frozen=True)
class ModelSegment:
    call_id: str
    index: int
    kind: Literal["reasoning", "tool_call", "tool_result", "final_message", "provider_metadata"]
    source_field: str
    text: str
    incomplete: bool = False
```

Parse `choices[0].delta` directly. Candidate reasoning field names come from provider config. Accumulate contiguous chunks of the same semantic field, flush when kind changes, and emit an `incomplete=True` segment before raising `ModelCallError` on an interrupted stream. Never log request headers or API keys.

- [ ] **Step 4: Run parser and redaction tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/models/test_openai_compat.py -v`

Expected: PASS for `reasoning_content`, `thinking`, simultaneous fields, split tool arguments, usage metadata, interruption, timeout, and secret redaction.

- [ ] **Step 5: Commit the model adapter**

```powershell
git add src/clocktower/models tests/models
git commit -m "feat: add OpenAI-compatible reasoning stream adapter"
```

---

### Task 8: Player Context, Notebook Tooling, and Call Lifecycle

**Files:**
- Create: `src/clocktower/agents/context.py`
- Create: `src/clocktower/agents/tools.py`
- Create: `src/clocktower/agents/player.py`
- Test: `tests/agents/test_context.py`
- Test: `tests/agents/test_player.py`

**Interfaces:**
- Produces: `project_context(player_id, state, events) -> PlayerContext`; `PlayerAgent.run_action(scene) -> AgentOutcome`; `PlayerAgent.probe(event) -> ReactionProbe`.
- Consumes configuration resolver, model adapter, history writer, event visibility, and canonical actions.
- Conversion boundary: every provider `ModelSegment` from Task 7 becomes one Task 2 `ModelOutputSegment` payload inside an `EventRecord(type="model.output_segment")`; provider objects never enter rules or frontend code directly.

- [ ] **Step 1: Write failing isolation and lifecycle tests**

```python
def test_context_excludes_other_players_private_and_reasoning_events():
    ctx = project_context("alice", state, [public_event(), private_event({"bob","carol"}), bob_reasoning_event()])
    assert [e.type for e in ctx.events] == ["chat.public_message"]

async def test_short_probe_does_not_mutate_notebook_or_normal_continuation():
    before = deepcopy(agent.state)
    result = await agent.probe(public_claim_event())
    assert result.decision in {"respond", "defer", "silent"}
    assert agent.state.notebook == before.notebook
    assert agent.state.continuation == before.continuation
```

- [ ] **Step 2: Verify missing agent failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agents/test_context.py tests/agents/test_player.py -v`

Expected: FAIL importing player-agent modules.

- [ ] **Step 3: Implement authorization-first prompt construction**

```python
def project_context(player_id: str, state: GameState, events: Sequence[EventRecord]) -> PlayerContext:
    visible = tuple(e for e in events if e.visible_to(player_id))
    player = state.players[player_id]
    return PlayerContext(
        identity=player.perceived_identity,
        alignment=player.known_alignment,
        ability_text=player.perceived_ability_text,
        notebook=player.notebook,
        events=visible,
        tools=tools_for(state.phase, player),
    )
```

`run_action` may call `update_notebook` repeatedly but accepts at most one outward game action, four tool round trips, and one illegal-action correction. Every accepted notebook patch calls `HistoryWriter.update_notebook`. `probe` uses the short resolver, stateless messages, no tools, and a strict Pydantic `ReactionProbe` response.

Use this exact adapter-to-domain conversion:

```python
def segment_event(player_id: str, purpose: str, segment: ModelSegment, phase: str) -> EventRecord:
    payload = ModelOutputSegment(
        call_id=segment.call_id, player_id=player_id, call_purpose=purpose,
        segment_index=segment.index, kind=segment.kind,
        source_field=segment.source_field, text=segment.text,
        incomplete=segment.incomplete,
    )
    return EventRecord(phase=phase, type="model.output_segment", actor=player_id,
                       audience=Audience(kind="observer"), payload=payload.model_dump())
```

- [ ] **Step 4: Run agent isolation and tool-budget tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agents -v`

Expected: PASS for public/private/observer filtering, perceived Drunk identity, tool availability by phase, notebook checkpointing, one outward action, correction limit, and short-call isolation.

- [ ] **Step 5: Commit player agents**

```powershell
git add src/clocktower/agents tests/agents
git commit -m "feat: add isolated AI player contexts and tools"
```

---

### Task 9: Event-Driven Scoring and Public Discussion

**Files:**
- Create: `src/clocktower/scheduler/scoring.py`
- Create: `src/clocktower/scheduler/discussion.py`
- Test: `tests/scheduler/test_scoring.py`
- Test: `tests/scheduler/test_discussion.py`

**Interfaces:**
- Produces: `score_candidates(event, state) -> list[CandidateScore]`; `choose_candidate(scores, seed_state) -> str | None`; `DiscussionScheduler.step() -> list[EventRecord]`.
- Consumes `PlayerAgent.probe`, action budgets, notebook attention metadata, and rule actions.

- [ ] **Step 1: Write failing deterministic scoring tests**

```python
def test_mentioned_player_scores_above_unrelated_player():
    scores = score_candidates(public_claim(actor="alice", mentions={"bob"}), sample_game_state())
    by_id = {s.player_id: s.total for s in scores}
    assert by_id["bob"] > by_id["carol"]

async def test_only_top_two_candidates_are_probed():
    scheduler = discussion_scheduler_with_spies()
    await scheduler.step()
    assert scheduler.probed_player_ids == scheduler.initial_ranking[:2]
```

- [ ] **Step 2: Confirm scheduler modules are absent**

Run: `.\.venv\Scripts\python.exe -m pytest tests/scheduler/test_scoring.py tests/scheduler/test_discussion.py -v`

Expected: FAIL importing scheduler modules.

- [ ] **Step 3: Implement auditable features and bounded selection**

```python
WEIGHTS = {
    "direct_target": 40, "mentioned": 25, "trigger": 20,
    "pending_action": 15, "fairness": 10, "recent_speaker": -20,
    "repeat_risk": -25, "budget_pressure": -15,
}
```

Each `CandidateScore` stores every feature contribution. Probe only the highest two base scores, cap urgency adjustment at ±15, filter below threshold, then perform seeded weighted choice. Increment quiet count when no eligible candidate remains; end the scene after configured quiet windows or hard action budgets.

- [ ] **Step 4: Run scheduler determinism and termination tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/scheduler/test_scoring.py tests/scheduler/test_discussion.py -v`

Expected: PASS for feature reasons, top-two probes, short-call failure fallback, cooldown, seeded repeatability, quiet termination, and hard limits.

- [ ] **Step 5: Commit public discussion scheduling**

```powershell
git add src/clocktower/scheduler tests/scheduler
git commit -m "feat: add event-driven discussion scheduler"
```

---

### Task 10: Bounded Two-Player Private Chat

**Files:**
- Create: `src/clocktower/scheduler/private_chat.py`
- Test: `tests/scheduler/test_private_chat.py`

**Interfaces:**
- Produces: `PrivateChatScheduler.request(inviter, invitee)`; `PrivateChatScheduler.run(chat_id)`; private events whose `Audience.players` contains exactly the two participants.

- [ ] **Step 1: Write failing privacy and termination tests**

```python
async def test_private_messages_are_visible_only_to_two_participants():
    events = await private_chat(alice_accepts=True).run()
    message = next(e for e in events if e.type == "chat.private_message")
    assert message.audience.player_ids == frozenset({"alice", "bob"})

async def test_private_chat_ends_after_both_players_have_no_new_information():
    chat = private_chat(probes=["silent", "silent"])
    await chat.run()
    assert chat.end_reason == "quiet"
```

- [ ] **Step 2: Verify missing private-chat scheduler**

Run: `.\.venv\Scripts\python.exe -m pytest tests/scheduler/test_private_chat.py -v`

Expected: FAIL importing `clocktower.scheduler.private_chat`.

- [ ] **Step 3: Implement one active subscene with separate notebook updates**

```python
@dataclass
class PrivateChatScene:
    chat_id: str
    participant_ids: tuple[str, str]
    action_count: int = 0
    quiet_count: int = 0
```

Invitee uses a short call to accept, reject, or defer. Accepted chats reuse scoring/probe rules limited to the two participants, enforce the private action budget, publish only a public `chat.private_started/ended` shell, and never create a shared summary. Participant notebook updates remain independent and create their own checkpoints.

- [ ] **Step 4: Run private-chat tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/scheduler/test_private_chat.py -v`

Expected: PASS for accept/reject/defer, audience isolation, public shell events, action cap, quiet end, and separate checkpoints.

- [ ] **Step 5: Commit private chat**

```powershell
git add src/clocktower/scheduler/private_chat.py tests/scheduler/test_private_chat.py
git commit -m "feat: add bounded private chat scenes"
```

---

### Task 11: Autonomous Game Orchestrator, Pause, and Error Policy

**Files:**
- Create: `src/clocktower/orchestrator.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/integration/test_headless_game.py`

**Interfaces:**
- Produces: `GameOrchestrator.run()`, `request_stop()`, `continue_game()`, `status() -> RuntimeStatus`.
- Consumes rules engine, player agents, discussion/private schedulers, history, config reload, and event stream.

- [ ] **Step 1: Write failing lifecycle tests with scripted fake agents**

```python
async def test_stop_waits_for_atomic_action_then_blocks_next_step():
    orchestrator = scripted_orchestrator(actions=[slow_valid_action(), next_action()])
    task = asyncio.create_task(orchestrator.run())
    await orchestrator.request_stop()
    await orchestrator.wait_until_stopped()
    assert orchestrator.completed_action_count == 1
    assert orchestrator.status().state == "stopped"
    task.cancel()

async def test_required_choice_failure_stops_instead_of_random_fallback():
    orchestrator = orchestrator_with_required_model_failure()
    await orchestrator.run_until_blocked()
    assert orchestrator.status().reason == "required_model_call_failed"
```

- [ ] **Step 2: Confirm orchestrator import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py tests/integration/test_headless_game.py -v`

Expected: FAIL importing `clocktower.orchestrator`.

- [ ] **Step 3: Implement the single-game state machine**

```python
async def run(self) -> None:
    await self._setup_and_first_night()
    while self.rules.check_winner() is None:
        await self._safe_point()
        await self._run_day()
        if self.rules.check_winner() is not None:
            break
        await self._safe_point()
        await self._run_night()
    await self._publish_game_ended()
```

Day runs public discussion/private scenes/nominations until quiet, then probes every still-eligible living nominator before execution resolution. Voting calls ordinary models in official seat order. Optional call failure retries once then yields; short failure retries once then uses base score; required choice failure retries once then stops. Continue reloads only provider/model connection configuration and records `model_config_reloaded`.

- [ ] **Step 4: Run fake-model full-game tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py tests/integration/test_headless_game.py -v`

Expected: PASS for setup-to-winner flow, final nomination probe, stop/continue safety, optional/required errors, config reload, and a deterministic five-player complete game without network access.

- [ ] **Step 5: Commit autonomous orchestration**

```powershell
git add src/clocktower/orchestrator.py tests/test_orchestrator.py tests/integration/test_headless_game.py
git commit -m "feat: orchestrate a complete autonomous game"
```

---

### Task 12: FastAPI Control and SSE Surface

**Files:**
- Create: `src/clocktower/api.py`
- Create: `src/clocktower/main.py`
- Test: `tests/api/test_api.py`

**Interfaces:**
- Produces: `GET /api/state`, `GET /api/events?after_seq=N`, `POST /api/control/stop`, `POST /api/control/continue`.
- SSE payload is the persisted `EventRecord.model_dump_json()`; no second wire model.

- [ ] **Step 1: Write failing API tests**

```python
def test_stop_and_continue_are_the_only_control_routes(client):
    assert client.post("/api/control/stop").status_code == 202
    assert client.post("/api/control/continue").status_code == 202
    assert client.post("/api/control/step").status_code == 404

def test_event_replay_starts_after_requested_sequence(client, seeded_stream):
    response = client.get("/api/events?after_seq=4", headers={"accept":"text/event-stream"})
    assert '"seq":5' in response.text
    assert '"seq":4' not in response.text
```

- [ ] **Step 2: Confirm API module is missing**

Run: `.\.venv\Scripts\python.exe -m pytest tests/api/test_api.py -v`

Expected: FAIL importing `clocktower.api`.

- [ ] **Step 3: Implement routes and lifespan ownership**

```python
@router.post("/api/control/stop", status_code=202)
async def stop(request: Request):
    await request.app.state.orchestrator.request_stop()
    return {"state": "stop_requested"}

@router.get("/api/events")
async def events(request: Request, after_seq: int = 0):
    return StreamingResponse(sse_events(request.app.state.stream, after_seq), media_type="text/event-stream")
```

FastAPI lifespan loads `config.toml`, creates one orchestrator, starts it as one application task, and cancels cleanly on shutdown. Serve `web/dist` only when it exists; API tests run without frontend assets.

- [ ] **Step 4: Run API reconnect and control tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/api/test_api.py -v`

Expected: PASS for state, control idempotency, replay, live SSE, disconnect tolerance, and absent unsupported control routes.

- [ ] **Step 5: Commit the local server**

```powershell
git add src/clocktower/api.py src/clocktower/main.py tests/api
git commit -m "feat: expose local control and SSE API"
```

---

### Task 13: React Event Stream, Raw Model Trace, and History File Mode

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/types.ts`
- Create: `web/src/lib/live.ts`
- Create: `web/src/lib/history.ts`
- Create: `web/src/components/ModelTrace.tsx`
- Create: `web/src/components/EventStream.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/test/fixtures.ts`
- Test: `web/src/lib/history.test.ts`
- Test: `web/src/components/EventStream.test.tsx`

**Interfaces:**
- Produces: `parseHistory(file: File) -> Promise<EventRecord[]>`; `connectLive(afterSeq, onEvent) -> () => void`; `<EventStream events filter />`.
- Consumes exact backend `EventRecord` JSON fields.

- [ ] **Step 1: Write failing parser and render tests**

```tsx
it("renders reasoning, tool call, result, and public text in segment order", () => {
  render(<EventStream events={orderedFixture} filter="all" />)
  expect(screen.getAllByTestId("trace-kind").map(n => n.textContent)).toEqual([
    "Think", "Tool", "Tool result", "Public message"
  ])
})

it("rejects a malformed history line with its line number", async () => {
  await expect(parseHistory(fileOf('{"seq":1}\nnot-json'))).rejects.toThrow("line 2")
})
```

- [ ] **Step 2: Install frontend dependencies and verify failing tests**

Run: `cd web; npm install; npm test -- --run`

Use exact dependencies:

```json
{
  "dependencies": {"react":"19.2.8","react-dom":"19.2.8"},
  "devDependencies": {
    "@testing-library/jest-dom":"7.0.1",
    "@testing-library/react":"16.3.2",
    "@vitejs/plugin-react":"6.0.5",
    "jsdom":"30.0.1",
    "typescript":"7.0.2",
    "vite":"8.2.1",
    "vitest":"4.1.11"
  }
}
```

Expected: FAIL because components and parsers are missing.

- [ ] **Step 3: Implement one view model for live and history events**

```ts
export type EventRecord = {
  seq: number
  time: string
  phase: string
  type: string
  actor: string | null
  audience: { kind: "public" | "players" | "player" | "observer"; player_ids: string[] }
  payload: Record<string, unknown>
}
```

`ModelTrace` groups only adjacent `model.output_segment` records sharing a `call_id`, retains `segment_index`, labels the actual `source_field`, and HTML-escapes through React text nodes. `EventStream` distinguishes players with stable CSS classes derived from seat index, not random colors. `parseHistory` reads browser-local file content and validates every line before replacing the current view.

`web/src/test/fixtures.ts` exports `orderedFixture: EventRecord[]` containing one reasoning segment, one tool call, one tool result, and one public message with sequences 1–4, plus:

```ts
export const fileOf = (text: string) => new File([text], "game.jsonl", {type: "application/x-ndjson"})
```

- [ ] **Step 4: Run frontend tests and build**

Run: `cd web; npm test -- --run; npm run build`

Expected: PASS and a successful production build with no TypeScript errors.

- [ ] **Step 5: Commit the conversation viewer**

```powershell
git add web
git commit -m "feat: add live and history conversation viewer"
```

---

### Task 14: Stop/Continue Toolbar, Filters, Checkpoint Panel, and Final Acceptance

**Files:**
- Create: `web/src/components/Toolbar.tsx`
- Create: `web/src/components/CheckpointPanel.tsx`
- Create: `web/src/styles.css`
- Modify: `web/src/App.tsx`
- Modify: `web/src/main.tsx`
- Test: `web/src/components/Toolbar.test.tsx`
- Test: `web/src/components/CheckpointPanel.test.tsx`
- Test: `web/src/App.test.tsx`
- Test: `tests/integration/test_history_replay.py`
- Create: `README.md`

**Interfaces:**
- Produces final single-window UI and documented commands.
- Consumes API/state, event stream, history parser, checkpoint payload, and existing event components.

- [ ] **Step 1: Write failing interaction tests**

```tsx
it("offers Stop while running and Continue while stopped, never step or speed controls", () => {
  const { rerender } = render(<Toolbar mode="live" runtime="running" {...handlers} />)
  expect(screen.getByRole("button", {name:"Stop"})).toBeVisible()
  expect(screen.queryByText(/speed|step|next phase/i)).toBeNull()
  rerender(<Toolbar mode="live" runtime="stopped" {...handlers} />)
  expect(screen.getByRole("button", {name:"Continue"})).toBeVisible()
})

it("checkpoint selection shows every player's notebook but no restore action", () => {
  render(<CheckpointPanel checkpoint={checkpointFixture} />)
  expect(screen.getByText("Alice notebook")).toBeVisible()
  expect(screen.queryByRole("button", {name:/restore|resume/i})).toBeNull()
})
```

Add the backend replay assertion in `tests/integration/test_history_replay.py`:

```python
def test_completed_history_has_contiguous_checkpoints_and_authorized_prompts(completed_game_path):
    records = [json.loads(line) for line in completed_game_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["type"] == "game.header"
    assert records[-1]["type"] == "game.ended"
    for index, record in enumerate(records[:-1]):
        if record["type"] == "notebook.updated":
            assert records[index + 1]["type"] == "checkpoint"
    assert all_prompt_records_respect_audience(records)
```

- [ ] **Step 2: Verify the new tests fail**

Run: `cd web; npm test -- --run`

Expected: FAIL because toolbar and checkpoint components are missing.

- [ ] **Step 3: Implement the final approved interaction only**

```tsx
export function Toolbar({mode, runtime, onStop, onContinue, onOpenHistory}: Props) {
  return <header>
    <button onClick={onOpenHistory}>Open history</button>
    {mode === "live" && runtime === "running" && <button onClick={onStop}>Stop</button>}
    {mode === "live" && runtime === "stopped" && <button onClick={onContinue}>Continue</button>}
  </header>
}
```

Add filters for all/body/reasoning/tools, a “back to latest” affordance that appears only after manual upward scrolling, history/live mode switching without stopping the game, and a read-only checkpoint side panel with player selector, notes, role state, trigger event, and locate-event action. Do not add dashboards, multi-game lists, restore, export, speed, or phase navigation.

README must contain exact setup and run commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item config.example.toml config.toml
cd web
npm install
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn clocktower.main:app --host 127.0.0.1 --port 4396
```

- [ ] **Step 4: Run all automated verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
cd web
npm test -- --run
npm run build
cd ..
```

Expected: all backend tests pass, all frontend tests pass, and Vite production build succeeds.

- [ ] **Step 5: Run the fake-provider acceptance path**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_headless_game.py tests/integration/test_history_replay.py -v`

Expected: a deterministic five-player game reaches `game.ended`; its JSONL reopens successfully; every notebook mutation is immediately followed by a checkpoint; no unauthorized event appears in any player prompt fixture.

- [ ] **Step 6: Perform manual local-browser acceptance**

Run the README command, then verify at `http://127.0.0.1:4396`:

1. Live events append in order and player identities are visually distinct.
2. `reasoning_content`/`thinking`, tools, results, and public text appear in returned order.
3. Stop takes effect after the current atomic action; Continue resumes.
4. Opening a completed JSONL file creates a scrollable history conversation.
5. A notebook checkpoint opens all-player notes and state without restore controls.
6. Returning to live view requests events after the last seen sequence.

- [ ] **Step 7: Commit the completed MVP**

```powershell
git add web README.md tests/integration/test_history_replay.py
git commit -m "feat: complete AI Clocktower observer console"
```

---

## Final Verification Gate

Before claiming the implementation complete, run:

```powershell
git status --short
.\.venv\Scripts\python.exe -m pytest -v
cd web
npm test -- --run
npm run build
cd ..
```

Then inspect one generated history file and assert:

- It begins with `game.header` and ends with `game.ended`.
- Event sequence numbers are strictly increasing.
- Every `notebook.updated` is immediately followed by `checkpoint`.
- Every model call preserves segment ordering and source field names.
- Private and observer events never appear in unauthorized player prompt records.
- The working tree contains no API key, `config.toml`, generated history, `.venv`, `node_modules`, `dist`, or `.superpowers` artifacts.

Finally, with a user-supplied API key and configured compatible model, run one five-player game to `game.ended` and perform the six manual browser checks from Task 14. If credentials or provider access are unavailable, report real-model/game acceptance as unverified; passing fake-provider tests is not evidence that an external provider works end to end.
