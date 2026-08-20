from __future__ import annotations

import pytest

from clocktower.domain.actions import CastVote, IllegalAction, Nominate, UseAbility
from clocktower.domain.events import Audience
from clocktower.event_stream import EventStream
from clocktower.rules.engine import RuleEngine
from clocktower.rules.night import FIRST_NIGHT_ORDER, OTHER_NIGHT_ORDER
from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.roles.evil import Imp
from clocktower.rules.setup import build_setup


def _vote_all(engine: RuleEngine, nomination_id: str, *, yes: set[str]) -> None:
    for voter in engine.current_vote_order:
        engine.apply_action(
            CastVote(actor=voter, nomination_id=nomination_id, vote=voter in yes)
        )


def _advance_to_day(engine: RuleEngine) -> None:
    for _ in range(32):
        if engine.state.phase.startswith("day"):
            return
        events = engine.advance_night_step()
        request = next((event for event in events if event.type == "ability.choice_requested"), None)
        if request is not None:
            targets = tuple(request.payload["legal_targets"][0])
            engine.apply_action(
                UseAbility(
                    actor=request.payload["actor_id"],
                    action=request.payload["role"],
                    targets=targets,
                )
            )
    raise AssertionError("night did not reach dawn")


def test_trouble_brewing_night_orders_are_explicit_and_complete():
    """Moving an information role before death resolution changes legal information."""

    assert FIRST_NIGHT_ORDER == (
        "minion_info",
        "demon_info",
        "poisoner",
        "spy",
        "washerwoman",
        "librarian",
        "investigator",
        "chef",
        "empath",
        "fortune_teller",
        "butler",
        "dawn",
    )
    assert OTHER_NIGHT_ORDER == (
        "poisoner",
        "monk",
        "spy",
        "scarlet_woman",
        "imp",
        "ravenkeeper",
        "undertaker",
        "empath",
        "fortune_teller",
        "butler",
        "dawn",
    )


async def test_start_game_uses_policy_roles_and_header_is_first_stream_record():
    """Publishing setup truth before the header breaks the append-only history contract."""

    player_ids = ("alice", "bob", "carol", "david", "eve")
    first = RuleEngine.start_game(player_ids, seed=23, model_config_snapshot={"model": "test"})
    second = RuleEngine.start_game(player_ids, seed=23, model_config_snapshot={"model": "test"})

    assert [player.role for player in first.state.players.values()] == [
        player.role for player in second.state.players.values()
    ]
    assert first.events[0].type == "game.header"
    assert first.events[0].payload["seed"] == 23
    assert any(event.type == "storyteller.decision" for event in first.events[1:])

    selected_roles = tuple(player.role for player in first.state.players.values())
    build_setup(len(player_ids), selected_roles, seed=23)

    stream = EventStream()
    published = [await stream.publish(event) for event in first.events]
    assert published[0].type == "game.header"
    assert [event.seq for event in published] == list(range(1, len(published) + 1))


def test_demon_bluffs_exclude_the_drunk_perceived_identity():
    """Offering the Drunk's token as a bluff would reveal or duplicate a used identity."""

    engine = RuleEngine.start_game(tuple(f"p{seat}" for seat in range(7)), seed=4)
    drunk = next(player for player in engine.state.players.values() if player.role == "drunk")
    assert drunk.perceived_identity == "soldier"

    engine.advance_night_step()  # Minion information
    events = engine.advance_night_step()  # Demon information and bluffs

    decision = next(
        event
        for event in events
        if event.type == "storyteller.decision"
        and event.payload["request_key"].startswith("demon_bluffs:")
    )
    assert all(
        drunk.perceived_identity not in bluff_set
        for bluff_set in decision.payload["options"]
    )


def test_start_game_applies_baron_counts_before_setup_validation_for_every_seed():
    """Selecting roles against base counts would reject or misbuild Baron games."""

    saw_baron = False
    for seed in range(12):
        engine = RuleEngine.start_game(
            ("alice", "bob", "carol", "david", "eve"), seed=seed
        )
        roles = tuple(player.role for player in engine.state.players.values())
        build_setup(5, roles, seed=seed)
        saw_baron = saw_baron or "baron" in roles

    assert saw_baron


def test_atomic_effect_failure_leaves_state_policy_and_events_unchanged():
    """Committing an early effect before a later failure creates an unaudited partial action."""

    engine = RuleEngine.for_test(
        {"alice": "slayer", "bob": "chef", "carol": "imp"}, seed=17
    )
    before_state = engine.state.model_copy(deep=True)
    before_events = engine.events
    before_decisions = engine.storyteller.decisions

    with pytest.raises(ValueError, match="unsupported rule effect"):
        engine._apply_effects_atomically(
            (
                RuleEffect("mark_used", {"player_id": "alice", "ability": "slayer"}),
                RuleEffect("not_a_real_effect", {}),
            ),
            actor_id="alice",
        )

    assert engine.state == before_state
    assert engine.events == before_events
    assert engine.storyteller.decisions == before_decisions


def test_dead_vote_record_and_token_consumption_commit_together():
    """Consuming a ghost vote before its ordered vote record succeeds breaks atomicity."""

    engine = RuleEngine.for_test(
        {"alice": "chef", "bob": "empath", "carol": "imp", "david": "monk"},
        dead={"bob"},
        seed=17,
    )
    nomination_event = engine.apply_action(
        Nominate(actor="alice", target="bob", accusation="test")
    )[0]
    nomination_id = nomination_event.payload["nomination_id"]

    engine.apply_action(CastVote(actor="carol", nomination_id=nomination_id, vote=False))
    engine.apply_action(CastVote(actor="david", nomination_id=nomination_id, vote=False))
    engine.apply_action(CastVote(actor="alice", nomination_id=nomination_id, vote=False))
    vote_events = engine.apply_action(
        CastVote(actor="bob", nomination_id=nomination_id, vote=True)
    )

    assert engine.state.players["bob"].dead_vote_available is False
    assert vote_events[0].payload["consumes_dead_vote"] is True


def test_invalid_ordered_dead_vote_rolls_back_token_and_events():
    """A rejected out-of-order vote must not spend the dead player's only vote."""

    engine = RuleEngine.for_test(
        {"alice": "chef", "bob": "empath", "carol": "imp", "david": "monk"},
        dead={"bob"},
        seed=17,
    )
    nomination_id = engine.apply_action(
        Nominate(actor="alice", target="bob", accusation="test")
    )[0].payload["nomination_id"]
    before_events = engine.events

    with pytest.raises(IllegalAction, match="expected vote from carol"):
        engine.apply_action(CastVote(actor="bob", nomination_id=nomination_id, vote=True))

    assert engine.state.players["bob"].dead_vote_available is True
    assert engine.events == before_events


def test_butler_none_falls_back_to_ordinary_vote_rules_when_poisoned():
    """Treating Butler None as false would illegally block a disabled Butler's vote."""

    engine = RuleEngine.for_test(
        {"alice": "butler", "bob": "chef", "carol": "imp"}, seed=17
    )
    engine._apply_effects_atomically(
        (RuleEffect("set_master", {"butler_id": "alice", "master_id": "bob"}),),
        actor_id="alice",
    )
    nomination_id = engine.apply_action(
        Nominate(actor="carol", target="bob", accusation="test")
    )[0].payload["nomination_id"]
    engine.apply_action(CastVote(actor="carol", nomination_id=nomination_id, vote=False))

    with pytest.raises(IllegalAction, match="Butler master"):
        engine.apply_action(CastVote(actor="alice", nomination_id=nomination_id, vote=True))

    engine._apply_effects_atomically(
        (RuleEffect("poison", {"target_id": "alice", "source": "test"}),),
        actor_id="carol",
    )
    engine.apply_action(CastVote(actor="alice", nomination_id=nomination_id, vote=True))

    assert engine.current_vote_order[-1] == "bob"


def test_virgin_policy_resolution_executes_then_ends_day_in_event_order():
    """Continuing nominations after Virgin's immediate execution violates the day lifecycle."""

    engine = RuleEngine.for_test(
        {"alice": "washerwoman", "bob": "virgin", "eve": "imp"}, seed=17
    )

    events = engine.apply_action(
        Nominate(actor="alice", target="bob", accusation="test")
    )

    assert engine.state.phase == "night"
    assert engine.state.players["alice"].alive is False
    assert "virgin_used" in engine.state.players["bob"].reminders
    types = [event.type for event in events]
    assert types.index("storyteller.decision") < types.index("execution.resolved")
    assert types.index("execution.resolved") < types.index("player.died")
    assert types.index("player.died") < types.index("day.ended")
    assert sum(event.type == "game.ended" for event in engine.events) == 1

    with pytest.raises(IllegalAction, match="game has ended"):
        engine.apply_action(Nominate(actor="eve", target="bob", accusation="too late"))
    assert sum(event.type == "game.ended" for event in engine.events) == 1


def test_information_policy_logs_all_candidates_but_projects_only_selected_private_result():
    """Putting candidate observations in a player event leaks registration and false-info choices."""

    engine = RuleEngine.for_test(
        {"alice": "washerwoman", "bob": "chef", "carol": "imp"},
        seed=17,
        phase="night",
        first_night=True,
    )
    new_events = []
    for _ in range(8):
        new_events.extend(engine.advance_night_step())
        if any(event.type == "information.received" for event in new_events):
            break

    decision = next(event for event in new_events if event.type == "storyteller.decision")
    received = next(event for event in new_events if event.type == "information.received")
    assert decision.audience == Audience.observer()
    assert decision.payload["request_key"].startswith("information:washerwoman:alice")
    assert decision.payload["options"]
    assert received.audience == Audience.player("alice")
    assert "options" not in received.payload
    assert not received.visible_to("bob")


def test_spy_grimoire_never_enters_another_players_visible_events():
    """A public or wrongly scoped Spy observation reveals the complete game truth."""

    engine = RuleEngine.for_test(
        {"alice": "spy", "bob": "chef", "carol": "imp"},
        seed=17,
        phase="night",
        first_night=True,
    )
    for _ in range(5):
        engine.advance_night_step()

    spy_event = next(event for event in engine.events if event.type == "information.received")
    assert spy_event.audience == Audience.player("alice")
    assert "grimoire" in spy_event.payload
    assert all(
        "grimoire" not in str(event.payload)
        for event in engine.events
        if event.visible_to("bob")
    )


def test_poison_lasts_through_next_day_and_expires_before_next_poisoner_choice():
    """Clearing poison at dawn would restore the target one day too early."""

    engine = RuleEngine.for_test(
        {"alice": "poisoner", "bob": "chef", "carol": "imp"},
        seed=17,
        phase="night",
        first_night=True,
    )
    for _ in range(3):
        events = engine.advance_night_step()
    request = next(event for event in events if event.type == "ability.choice_requested")
    assert request.payload["role"] == "poisoner"
    engine.apply_action(UseAbility(actor="alice", action="poisoner", targets=("bob",)))

    _advance_to_day(engine)
    assert "poisoned" in engine.state.players["bob"].reminders

    engine.end_day()
    poisoner_step = engine.advance_night_step()
    assert next(event for event in poisoner_step if event.type == "ability.choice_requested")
    assert "poisoned" not in engine.state.players["bob"].reminders


def test_monk_protection_prevents_imp_death_and_expires_at_dawn():
    """Applying an Imp kill without the same-night protection check kills a safe player."""

    engine = RuleEngine.for_test(
        {"alice": "monk", "bob": "chef", "carol": "imp"},
        seed=17,
        phase="night",
        first_night=False,
    )
    engine.advance_night_step()  # absent Poisoner
    monk_request = engine.advance_night_step()
    assert next(event for event in monk_request if event.type == "ability.choice_requested")
    engine.apply_action(UseAbility(actor="alice", action="monk", targets=("bob",)))
    engine.advance_night_step()  # absent Spy
    engine.advance_night_step()  # Scarlet Woman notification slot
    imp_request = engine.advance_night_step()
    assert next(event for event in imp_request if event.type == "ability.choice_requested")
    kill_events = engine.apply_action(UseAbility(actor="carol", action="imp", targets=("bob",)))

    assert engine.state.players["bob"].alive is True
    assert any(event.type == "death.prevented" for event in kill_events)
    _advance_to_day(engine)
    assert "protected" not in engine.state.players["bob"].reminders


def test_poisoned_ravenkeeper_remains_misinformed_for_its_night_death_trigger():
    """Clearing poison on death would incorrectly restore Ravenkeeper's triggered information."""

    engine = RuleEngine.for_test(
        {
            "alice": "poisoner",
            "bob": "imp",
            "carol": "ravenkeeper",
            "david": "chef",
            "eve": "monk",
        },
        seed=17,
        phase="night",
        first_night=False,
    )
    poison_request = engine.advance_night_step()
    assert next(event for event in poison_request if event.type == "ability.choice_requested")
    engine.apply_action(
        UseAbility(actor="alice", action="poisoner", targets=("carol",))
    )
    monk_request = engine.advance_night_step()
    assert next(event for event in monk_request if event.type == "ability.choice_requested")
    engine.apply_action(UseAbility(actor="eve", action="monk", targets=("david",)))
    engine.advance_night_step()  # absent Spy
    engine.advance_night_step()  # Scarlet Woman notification slot
    imp_request = engine.advance_night_step()
    assert next(event for event in imp_request if event.type == "ability.choice_requested")
    engine.apply_action(UseAbility(actor="bob", action="imp", targets=("carol",)))
    raven_request = engine.advance_night_step()
    assert next(event for event in raven_request if event.type == "ability.choice_requested")

    events = engine.apply_action(
        UseAbility(actor="carol", action="ravenkeeper", targets=("david",))
    )

    decision = next(event for event in events if event.type == "storyteller.decision")
    assert "poisoned" in engine.state.players["carol"].reminders
    assert any(option["truthful"] is False for option in decision.payload["options"])


def test_poisoned_slayer_spends_use_without_killing_or_triggering_demon_death():
    """Checking health after applying the kill would let a disabled Slayer end the game."""

    engine = RuleEngine.for_test(
        {"alice": "slayer", "bob": "chef", "carol": "imp"}, seed=17
    )
    engine._apply_effects_atomically(
        (RuleEffect("poison", {"target_id": "alice", "source": "test"}),),
        actor_id="carol",
    )

    events = engine.apply_action(
        UseAbility(actor="alice", action="slayer", targets=("carol",))
    )

    assert "slayer_used" in engine.state.players["alice"].reminders
    assert engine.state.players["carol"].alive is True
    assert not any(event.type == "game.ended" for event in events)


def test_imp_self_kill_confirms_death_then_uses_policy_successor_without_good_win():
    """Calling the Demon-death hook before the successor settles produces a false good win."""

    engine = RuleEngine.for_test(
        {
            "alice": "imp",
            "bob": "poisoner",
            "carol": "scarlet_woman",
            "david": "chef",
            "eve": "monk",
        },
        seed=17,
        phase="night",
        first_night=False,
    )
    for _ in range(5):
        events = engine.advance_night_step()
        if any(event.type == "ability.choice_requested" for event in events):
            request = next(event for event in events if event.type == "ability.choice_requested")
            if request.payload["role"] == "imp":
                break
            engine.apply_action(
                UseAbility(
                    actor=request.payload["actor_id"],
                    action=request.payload["role"],
                    targets=(
                        ("david",)
                        if request.payload["role"] == "poisoner"
                        else ("bob",)
                    ),
                )
            )

    resolved = engine.apply_action(
        UseAbility(actor="alice", action="imp", targets=("alice",))
    )

    assert engine.state.players["alice"].alive is False
    assert engine.state.players["carol"].role == "imp"
    assert engine.check_winner() is None
    assert any(event.type == "storyteller.decision" for event in resolved)
    assert not any(event.type == "game.ended" for event in resolved)


def test_prevented_imp_self_kill_does_not_create_a_second_living_imp():
    """A successor effect must depend on confirmed self-kill, not merely a healthy Imp."""

    engine = RuleEngine.for_test(
        {
            "alice": "imp",
            "bob": "poisoner",
            "carol": "chef",
            "david": "empath",
            "eve": "monk",
        },
        seed=17,
        phase="night",
        first_night=False,
    )
    engine._apply_effects_atomically(
        (
            RuleEffect(
                "protect",
                {"target_id": "alice", "source": "monk"},
            ),
        ),
        actor_id="eve",
    )
    imp_effects = Imp().apply(
        AbilityContext.from_state(engine.state, "alice"),
        AbilityChoice("alice", ("alice",)),
    )

    engine._apply_effects_atomically(imp_effects, actor_id="alice")

    assert engine.state.players["alice"].alive is True
    assert [
        player.player_id
        for player in engine.state.players.values()
        if player.alive and player.role == "imp"
    ] == ["alice"]


def test_execution_of_saint_declares_evil_and_emits_exactly_one_game_end():
    """Treating Saint as an ordinary death misses its immediate loss condition."""

    engine = RuleEngine.for_test(
        {"alice": "chef", "bob": "saint", "carol": "imp"}, seed=17
    )
    nomination_id = engine.apply_action(
        Nominate(actor="alice", target="bob", accusation="test")
    )[0].payload["nomination_id"]
    _vote_all(engine, nomination_id, yes={"alice", "bob", "carol"})
    events = engine.end_day()

    assert engine.check_winner() == "evil"
    assert engine.state.players["bob"].alive is False
    assert [event.type for event in events].count("game.ended") == 1
    assert sum(event.type == "game.ended" for event in engine.events) == 1


def test_newly_dead_player_receives_exactly_one_available_dead_vote():
    """Leaving the living-player token state unchanged would deny every in-game death a ghost vote."""

    engine = RuleEngine.for_test(
        {
            "alice": "chef",
            "bob": "empath",
            "carol": "monk",
            "david": "imp",
        },
        seed=17,
    )
    nomination_id = engine.apply_action(
        Nominate(actor="bob", target="alice", accusation="test")
    )[0].payload["nomination_id"]
    _vote_all(engine, nomination_id, yes={"alice", "bob", "carol"})

    engine.end_day()

    assert engine.state.players["alice"].alive is False
    assert engine.state.players["alice"].dead_vote_available is True


def test_later_nomination_uses_new_alive_threshold_without_requalifying_earlier_tally():
    """A daytime death changes later thresholds but must not retroactively qualify an old vote."""

    engine = RuleEngine.for_test(
        {
            "alice": "chef",
            "bob": "empath",
            "carol": "monk",
            "david": "imp",
            "eve": "washerwoman",
        },
        seed=17,
    )
    first = engine.apply_action(
        Nominate(actor="alice", target="bob", accusation="first")
    )[0].payload["nomination_id"]
    _vote_all(engine, first, yes={"carol", "david"})  # 2 of 5: not qualifying

    engine._apply_effects_atomically(
        (RuleEffect("kill", {"target_id": "eve", "source": "test"}),),
        actor_id="alice",
    )
    second = engine.apply_action(
        Nominate(actor="carol", target="alice", accusation="second")
    )[0].payload["nomination_id"]
    _vote_all(engine, second, yes={"bob", "carol"})  # 2 of 4: qualifying

    engine.end_day()

    assert engine.state.players["alice"].alive is False
    assert engine.state.players["bob"].alive is True


def test_imp_execution_at_five_alive_promotes_scarlet_before_demon_win_check():
    """Checking for a living Demon before Scarlet Woman resolves ends a continuing game."""

    engine = RuleEngine.for_test(
        {
            "alice": "chef",
            "bob": "empath",
            "carol": "monk",
            "david": "scarlet_woman",
            "eve": "imp",
        },
        seed=17,
    )
    nomination_id = engine.apply_action(
        Nominate(actor="alice", target="eve", accusation="test")
    )[0].payload["nomination_id"]
    _vote_all(engine, nomination_id, yes=set(engine.state.players))
    events = engine.end_day()

    assert engine.state.players["eve"].alive is False
    assert engine.state.players["david"].role == "imp"
    assert engine.check_winner() is None
    assert not any(event.type == "game.ended" for event in events)


def test_mayor_wins_with_exactly_three_alive_and_no_execution():
    """Skipping the no-execution day-end hook loses Mayor's alternate victory."""

    engine = RuleEngine.for_test(
        {"alice": "mayor", "bob": "chef", "carol": "imp"}, seed=17
    )

    events = engine.end_day()

    assert engine.check_winner() == "good"
    assert [event.payload.get("reason") for event in events if event.type == "game.ended"] == [
        "mayor"
    ]


def test_two_living_players_is_evil_win_while_demon_lives():
    """Waiting for one living player would miss the base evil victory threshold."""

    engine = RuleEngine.for_test(
        {"alice": "chef", "bob": "saint", "carol": "imp"},
        dead={"alice"},
        seed=17,
    )

    assert engine.check_winner() == "evil"
