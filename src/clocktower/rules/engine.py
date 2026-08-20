"""Authoritative deterministic Trouble Brewing rule engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

from clocktower.domain.actions import (
    CastVote,
    IllegalAction,
    Nominate,
    PlayerAction,
    SpeakPublic,
    UseAbility,
    YieldAction,
)
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState
from clocktower.rules.night import FIRST_NIGHT_ORDER, OTHER_NIGHT_ORDER
from clocktower.rules.roles.action import Mayor, Monk, Slayer, Soldier, Virgin
from clocktower.rules.roles.base import AbilityChoice, AbilityContext, Observation, RuleEffect
from clocktower.rules.roles.evil import Imp, Poisoner, ScarletWoman
from clocktower.rules.roles.information import (
    Chef,
    Empath,
    FortuneTeller,
    Investigator,
    Librarian,
    Ravenkeeper,
    Spy,
    Undertaker,
    Washerwoman,
)
from clocktower.rules.roles.outsiders import Butler, Drunk, Saint
from clocktower.rules.roles.registration import Registration, RegistrationQuery, registrations_for
from clocktower.rules.setup import ROLE_CATEGORIES, build_setup, setup_counts
from clocktower.rules.voting import NominationTracker
from clocktower.storyteller import (
    DecisionRecord,
    DecisionRequest,
    StorytellerPolicy,
    decision_jsonable,
)


_TOWNSFOLK = tuple(role for role, category in ROLE_CATEGORIES.items() if category == "townsfolk")
_OUTSIDERS = tuple(role for role, category in ROLE_CATEGORIES.items() if category == "outsider")
_MINIONS = tuple(role for role, category in ROLE_CATEGORIES.items() if category == "minion")
_DEMONS = tuple(role for role, category in ROLE_CATEGORIES.items() if category == "demon")

_ACTIVE_NIGHT_HANDLERS = {
    "poisoner": Poisoner,
    "monk": Monk,
    "fortune_teller": FortuneTeller,
    "butler": Butler,
    "imp": Imp,
    "ravenkeeper": Ravenkeeper,
}

_PASSIVE_INFORMATION_HANDLERS = {
    "washerwoman": Washerwoman,
    "librarian": Librarian,
    "investigator": Investigator,
    "chef": Chef,
    "empath": Empath,
    "spy": Spy,
    "undertaker": Undertaker,
}


@dataclass(frozen=True, slots=True)
class _PendingEffect:
    effect: RuleEffect
    actor_id: str | None = None


@dataclass(frozen=True, slots=True)
class _DemonDeath:
    demon_id: str
    alive_before: int


class RuleEngine:
    """The only production boundary allowed to replace authoritative GameState."""

    def __init__(
        self,
        state: GameState,
        storyteller: StorytellerPolicy,
        *,
        events: Sequence[EventRecord] = (),
        nominations: NominationTracker | None = None,
    ) -> None:
        self.state = state
        self.storyteller = storyteller
        self._events = list(events)
        self._nominations = nominations

    @classmethod
    def start_game(
        cls,
        player_ids: Sequence[str],
        *,
        seed: int,
        model_config_snapshot: dict[str, Any] | None = None,
    ) -> RuleEngine:
        """Create production setup using policy-selected roles, never caller roles."""

        ordered_player_ids = tuple(player_ids)
        if not 5 <= len(ordered_player_ids) <= 15:
            raise ValueError("Trouble Brewing requires 5 to 15 players")
        if len(set(ordered_player_ids)) != len(ordered_player_ids):
            raise ValueError("player IDs must be unique")

        policy = StorytellerPolicy(seed)
        selected_roles = _select_setup_roles(len(ordered_player_ids), policy)
        setup = build_setup(len(ordered_player_ids), selected_roles, seed)
        assignments = dict(zip(ordered_player_ids, setup.roles_by_seat, strict=True))
        initial_state = GameState.from_assignments(assignments, seed=seed)
        state = initial_state.model_copy(
            update={
                "phase": "night",
                "role_state": initial_state.role_state.model_copy(
                    update={"first_night": True}, deep=True
                ),
            },
            deep=True,
        )

        for player in state.players.values():
            if player.role == "drunk":
                options = Drunk().legal_perceived_identities(
                    AbilityContext.from_state(state, player.player_id)
                )
                perceived = policy.choose(
                    DecisionRequest(
                        key=f"setup:drunk_identity:{player.player_id}",
                        options=options,
                        reason_code="drunk_not_in_play_townsfolk",
                    )
                )
                state = state.model_copy(
                    update={
                        "players": {
                            **state.players,
                            player.player_id: player.model_copy(
                                update={"perceived_identity": perceived}, deep=True
                            ),
                        }
                    },
                    deep=True,
                )

        fortune_tellers = [
            player.player_id
            for player in state.players.values()
            if player.perceived_identity == "fortune_teller"
        ]
        if fortune_tellers:
            red_herring_options = tuple(
                player.player_id for player in state.players.values() if player.alignment == "good"
            )
            red_herring = policy.choose(
                DecisionRequest(
                    key="setup:fortune_teller_red_herring",
                    options=red_herring_options,
                    reason_code="fortune_teller_red_herring",
                )
            )
            state = state.model_copy(
                update={
                    "role_state": state.role_state.model_copy(
                        update={"fortune_teller_red_herring": red_herring}, deep=True
                    )
                },
                deep=True,
            )

        header = EventRecord(
            phase="setup",
            type="game.header",
            audience=Audience.observer(),
            payload={
                "schema_version": 1,
                "rules_version": "trouble-brewing-1",
                "script": "trouble_brewing",
                "seed": seed,
                "model_config_snapshot": deepcopy(model_config_snapshot or {}),
                "players": [
                    {"player_id": player_id, "seat": seat, "display_name": player_id}
                    for seat, player_id in enumerate(ordered_player_ids)
                ],
            },
        )
        decision_events = tuple(
            _decision_event(record, phase="setup") for record in policy.decisions
        )
        setup_event = EventRecord(
            phase="setup",
            type="setup.completed",
            audience=Audience.observer(),
            payload={
                "roles_by_player": {
                    player_id: player.role for player_id, player in state.players.items()
                },
                "category_counts": setup.category_counts,
            },
        )
        assignment_events = tuple(
            EventRecord(
                phase="setup",
                type="role.assigned",
                actor=player.player_id,
                audience=Audience.player(player.player_id),
                payload={
                    "player_id": player.player_id,
                    "identity": player.perceived_identity,
                    "alignment": player.known_alignment,
                },
            )
            for player in sorted(state.players.values(), key=lambda item: item.seat)
        )
        return cls(
            state,
            policy,
            events=(header, *decision_events, setup_event, *assignment_events),
        )

    @classmethod
    def for_test(
        cls,
        assignments: dict[str, str],
        *,
        seed: int,
        dead: set[str] | None = None,
        phase: str = "day.discussion",
        first_night: bool = False,
    ) -> RuleEngine:
        initial_state = GameState.from_assignments(assignments, dead=dead, seed=seed)
        state = initial_state.model_copy(
            update={
                "phase": phase,
                "role_state": initial_state.role_state.model_copy(
                    update={"first_night": first_night}, deep=True
                ),
            },
            deep=True,
        )
        return cls(state, StorytellerPolicy(seed))

    @property
    def events(self) -> tuple[EventRecord, ...]:
        return tuple(self._events)

    @property
    def current_vote_order(self) -> tuple[str, ...]:
        return () if self._nominations is None else self._nominations.current_vote_order

    def apply_action(self, action: PlayerAction) -> list[EventRecord]:
        if self.state.role_state.game_ended:
            raise IllegalAction("game has ended")
        before = len(self._events)
        effects = self._dispatch(action)
        self._apply_effects_atomically(effects)
        return list(self._events[before:])

    def advance_night_step(self) -> list[EventRecord]:
        if self.state.role_state.game_ended:
            raise IllegalAction("game has ended")
        if self.state.phase != "night":
            raise IllegalAction("night step is only available at night")
        if self.state.role_state.pending_night_role is not None:
            raise IllegalAction("night ability choice is still pending")

        order = FIRST_NIGHT_ORDER if self.state.role_state.first_night else OTHER_NIGHT_ORDER
        index = self.state.role_state.night_step_index
        if index >= len(order):
            raise IllegalAction("night order is already complete")
        role = order[index]
        effects = self._night_step_effects(role)
        before = len(self._events)
        self._apply_effects_atomically(effects)
        return list(self._events[before:])

    def end_day(self) -> list[EventRecord]:
        if self.state.role_state.game_ended:
            raise IllegalAction("game has ended")
        if not self.state.phase.startswith("day"):
            raise IllegalAction("day can end only during the day")
        if self._nominations is not None and self._nominations.active_nomination_id is not None:
            raise IllegalAction("active nomination voting is not complete")

        execution = None if self._nominations is None else self._nominations.resolve_execution()
        pending: list[_PendingEffect] = []
        if execution is None:
            pending.append(
                _PendingEffect(
                    RuleEffect(
                        "emit_event",
                        {
                            "type": "execution.none",
                            "audience": Audience.public(),
                            "payload": {"day": self.state.day},
                        },
                    )
                )
            )
            for player in self.state.players.values():
                if player.role == "mayor" and player.alive:
                    pending.extend(
                        _tagged(
                            Mayor().end_of_day_effects(
                                AbilityContext.from_state(self.state, player.player_id),
                                no_execution=True,
                            ),
                            player.player_id,
                        )
                    )
        else:
            pending.append(
                _PendingEffect(
                    RuleEffect("execute", {"target_id": execution, "source": "vote"})
                )
            )
        pending.append(_PendingEffect(RuleEffect("end_day", {"reason": "day_complete"})))

        before = len(self._events)
        self._apply_effects_atomically(tuple(pending))
        return list(self._events[before:])

    def check_winner(self) -> str | None:
        return _winner_for(self.state)

    def _dispatch(self, action: PlayerAction) -> tuple[_PendingEffect, ...]:
        if action.actor not in self.state.players:
            raise IllegalAction(f"unknown actor: {action.actor}")
        if isinstance(action, Nominate):
            return self._dispatch_nomination(action)
        if isinstance(action, CastVote):
            return self._dispatch_vote(action)
        if isinstance(action, UseAbility):
            return self._dispatch_ability(action)
        if isinstance(action, SpeakPublic):
            if not self.state.phase.startswith("day"):
                raise IllegalAction("public speech is available only during the day")
            return (
                _PendingEffect(
                    RuleEffect(
                        "emit_event",
                        {
                            "type": "player.public_message",
                            "actor": action.actor,
                            "audience": Audience.public(),
                            "payload": {"text": action.text},
                        },
                    ),
                    action.actor,
                ),
            )
        if isinstance(action, YieldAction):
            return (
                _PendingEffect(
                    RuleEffect(
                        "emit_event",
                        {
                            "type": "player.yielded",
                            "actor": action.actor,
                            "audience": Audience.public(),
                            "payload": {"reason": action.reason},
                        },
                    ),
                    action.actor,
                ),
            )
        raise IllegalAction(f"action is not resolved by RuleEngine: {action.kind}")

    def _dispatch_nomination(self, action: Nominate) -> tuple[_PendingEffect, ...]:
        if self.state.phase != "day.discussion":
            raise IllegalAction("nominations require day discussion")
        tracker = deepcopy(self._nominations) or NominationTracker(
            alive_count=self.state.alive_count,
            nomination_id_prefix=f"nom-day-{self.state.day}",
        )
        tracker.update_alive_count(self.state.alive_count)
        nomination = tracker.nominate(
            self.state,
            action.actor,
            action.target,
            accusation=action.accusation,
        )
        effects: list[_PendingEffect] = [
            _PendingEffect(
                RuleEffect(
                    "emit_event",
                    {
                        "type": "nomination.opened",
                        "actor": action.actor,
                        "audience": Audience.public(),
                        "payload": {
                            "nomination_id": nomination.nomination_id,
                            "nominator": nomination.nominator,
                            "nominee": nomination.nominee,
                            "accusation": nomination.accusation,
                            "vote_order": nomination.vote_order,
                        },
                    },
                ),
                action.actor,
            ),
            _PendingEffect(RuleEffect("replace_nomination_tracker", {"tracker": tracker})),
            _PendingEffect(RuleEffect("set_phase", {"phase": "day.nomination"})),
        ]
        nominee = self.state.players[action.target]
        if nominee.role == "virgin":
            virgin_effects = Virgin().on_nominated(
                AbilityContext.for_nomination(
                    self.state, nominator=action.actor, nominee=action.target
                )
            )
            effects.extend(_tagged(virgin_effects, action.target))
        return tuple(effects)

    def _dispatch_vote(self, action: CastVote) -> tuple[_PendingEffect, ...]:
        if self.state.phase != "day.nomination" or self._nominations is None:
            raise IllegalAction("there is no active nomination")
        tracker = deepcopy(self._nominations)
        record = tracker.cast_vote(
            self.state, action.actor, action.nomination_id, action.vote
        )
        public_resolutions: tuple[dict[str, Any], ...] = ()
        rule_resolutions: tuple[dict[str, Any], ...] = ()
        if tracker.active_nomination_id is None:
            tally, public_resolutions, rule_resolutions = (
                self._resolve_completed_tally(tracker, action.nomination_id)
            )
            tracker.record_resolved_tally(action.nomination_id, tally)

        effects: list[_PendingEffect] = [
            _PendingEffect(RuleEffect("replace_nomination_tracker", {"tracker": tracker})),
        ]
        if record.consumes_dead_vote:
            effects.append(
                _PendingEffect(
                    RuleEffect("consume_dead_vote", {"player_id": action.actor}),
                    action.actor,
                )
            )
        effects.append(
            _PendingEffect(
                RuleEffect(
                    "emit_event",
                    {
                        "type": "vote.cast",
                        "actor": action.actor,
                        "audience": Audience.public(),
                        "payload": {
                            "voter": action.actor,
                            "nomination_id": action.nomination_id,
                            "intended": action.vote,
                            "counted": None,
                            "consumes_dead_vote": record.consumes_dead_vote,
                        },
                    },
                ),
                action.actor,
            )
        )
        if tracker.active_nomination_id is None:
            effects.extend(
                _PendingEffect(
                    RuleEffect(
                        "emit_event",
                        {
                            "type": "vote.resolved",
                            "actor": resolution["voter"],
                            "audience": Audience.public(),
                            "payload": resolution,
                        },
                    )
                )
                for resolution in public_resolutions
            )
            effects.extend(
                _PendingEffect(
                    RuleEffect(
                        "emit_event",
                        {
                            "type": "vote.rule_resolved",
                            "audience": Audience.observer(),
                            "payload": resolution,
                        },
                    )
                )
                for resolution in rule_resolutions
            )
            effects.extend(
                (
                    _PendingEffect(
                        RuleEffect(
                            "emit_event",
                            {
                                "type": "nomination.closed",
                                "audience": Audience.public(),
                                "payload": {
                                    "nomination_id": action.nomination_id,
                                    "tally": tracker.tally_for(action.nomination_id),
                                },
                            },
                        )
                    ),
                    _PendingEffect(RuleEffect("set_phase", {"phase": "day.discussion"})),
                )
            )
        return tuple(effects)

    def _resolve_completed_tally(
        self,
        tracker: NominationTracker,
        nomination_id: str,
    ) -> tuple[
        int,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ]:
        votes = tracker.votes_for(nomination_id)
        master_votes = {vote.voter: vote.vote for vote in votes}
        tally = 0
        public_resolutions: list[dict[str, Any]] = []
        rule_resolutions: list[dict[str, Any]] = []
        for vote in votes:
            counted = vote.vote
            player = self.state.players[vote.voter]
            if vote.vote and player.role == "butler":
                master_id = self.state.role_state.butler_masters.get(vote.voter)
                may_vote = Butler().may_vote(
                    AbilityContext.from_state(self.state, vote.voter),
                    master_is_voting=bool(
                        master_id is not None and master_votes.get(master_id, False)
                    ),
                )
                counted = may_vote is not False
                rule_resolutions.append(
                    {
                        "voter": vote.voter,
                        "nomination_id": nomination_id,
                        "vote": True,
                        "counted": counted,
                        "reason": "butler_master_vote",
                    }
                )
            tally += int(counted)
            public_resolutions.append(
                {
                    "voter": vote.voter,
                    "nomination_id": nomination_id,
                    "intended": vote.vote,
                    "counted": bool(counted),
                }
            )
        return tally, tuple(public_resolutions), tuple(rule_resolutions)

    def _dispatch_ability(self, action: UseAbility) -> tuple[_PendingEffect, ...]:
        if self.state.phase == "night":
            pending_role = self.state.role_state.pending_night_role
            pending_actor = self.state.role_state.pending_night_actor_id
            if (action.action, action.actor) != (pending_role, pending_actor):
                raise IllegalAction("action does not match the pending night ability")
            return self._night_ability_effects(action)

        if self.state.phase.startswith("day") and action.action == "slayer":
            player = self.state.players[action.actor]
            if player.perceived_identity != "slayer":
                raise IllegalAction("actor does not perceive the Slayer ability")
            choice = AbilityChoice(action.actor, action.targets)
            return tuple(_tagged(Slayer().apply(AbilityContext.from_state(self.state, action.actor), choice), action.actor))
        raise IllegalAction("ability is not available in the current phase")

    def _night_step_effects(self, role: str) -> tuple[_PendingEffect, ...]:
        if role == "dawn":
            return (_PendingEffect(RuleEffect("dawn", {})),)
        if role == "minion_info":
            effects = self._evil_information_effects(minions=True)
            return (*effects, _PendingEffect(RuleEffect("advance_night_step", {})))
        if role == "demon_info":
            effects = self._evil_information_effects(minions=False)
            return (*effects, _PendingEffect(RuleEffect("advance_night_step", {})))
        if role == "poisoner":
            prefix = [_PendingEffect(RuleEffect("clear_poison", {}))]
            actor_id = self._night_actor(role)
            if actor_id is None:
                return (*prefix, _PendingEffect(RuleEffect("advance_night_step", {})))
            choices = Poisoner().legal_choices(AbilityContext.from_state(self.state, actor_id))
            if not choices:
                return (*prefix, _PendingEffect(RuleEffect("advance_night_step", {})))
            return (
                *prefix,
                _PendingEffect(
                    RuleEffect(
                        "request_night_choice",
                        {
                            "role": role,
                            "actor_id": actor_id,
                            "legal_targets": tuple(choice.targets for choice in choices),
                        },
                    ),
                    actor_id,
                ),
            )
        if role == "scarlet_woman":
            actor = next(
                (
                    player
                    for player in self.state.players.values()
                    if "became_imp" in player.reminders
                ),
                None,
            )
            effects: list[_PendingEffect] = []
            if actor is not None:
                effects.append(
                    _PendingEffect(
                        RuleEffect(
                            "notify_new_imp",
                            {
                                "player_id": actor.player_id,
                                "source": "scarlet_woman",
                            },
                        ),
                        actor.player_id,
                    )
                )
            effects.append(_PendingEffect(RuleEffect("advance_night_step", {})))
            return tuple(effects)

        if role in _ACTIVE_NIGHT_HANDLERS:
            actor_id = self._night_actor(role)
            if actor_id is None:
                return (_PendingEffect(RuleEffect("advance_night_step", {})),)
            handler = _ACTIVE_NIGHT_HANDLERS[role]()
            choices = handler.legal_choices(AbilityContext.from_state(self.state, actor_id))
            if not choices:
                return (_PendingEffect(RuleEffect("advance_night_step", {})),)
            return (
                _PendingEffect(
                    RuleEffect(
                        "request_night_choice",
                        {
                            "role": role,
                            "actor_id": actor_id,
                            "legal_targets": tuple(choice.targets for choice in choices),
                        },
                    ),
                    actor_id,
                ),
            )

        if role in _PASSIVE_INFORMATION_HANDLERS:
            actor_id = self._night_actor(role)
            if actor_id is None:
                return (_PendingEffect(RuleEffect("advance_night_step", {})),)
            options = self._information_options(role, actor_id)
            effects: list[_PendingEffect] = []
            if options:
                effects.append(
                    _PendingEffect(
                        RuleEffect(
                            "select_observation",
                            {
                                "role": role,
                                "actor_id": actor_id,
                                "options": options,
                            },
                        ),
                        actor_id,
                    )
                )
            effects.append(_PendingEffect(RuleEffect("advance_night_step", {})))
            return tuple(effects)
        raise ValueError(f"unknown night order entry: {role}")

    def _night_ability_effects(self, action: UseAbility) -> tuple[_PendingEffect, ...]:
        context = AbilityContext.from_state(self.state, action.actor)
        choice = AbilityChoice(action.actor, action.targets)
        role = action.action
        if role == "fortune_teller":
            handler = FortuneTeller()
            if choice not in handler.legal_choices(context):
                raise IllegalAction("illegal Fortune Teller targets")
            effects = [
                _PendingEffect(
                    RuleEffect(
                        "select_observation",
                        {
                            "role": role,
                            "actor_id": action.actor,
                            "options": handler.legal_observations(context, action.targets),
                        },
                    ),
                    action.actor,
                )
            ]
        elif role == "ravenkeeper":
            handler = Ravenkeeper()
            if choice not in handler.legal_choices(context):
                raise IllegalAction("illegal Ravenkeeper targets")
            observation = handler.choose(context, action.targets[0])
            effects = [
                _PendingEffect(
                    RuleEffect(
                        "select_observation",
                        {
                            "role": role,
                            "actor_id": action.actor,
                            "options": observation.options,
                            "target_ids": observation.player_ids,
                        },
                    ),
                    action.actor,
                )
            ]
        else:
            try:
                handler = _ACTIVE_NIGHT_HANDLERS[role]()
            except KeyError as error:
                raise IllegalAction(f"unsupported night ability: {role}") from error
            try:
                effects = list(_tagged(handler.apply(context, choice), action.actor))
            except ValueError as error:
                raise IllegalAction(str(error)) from error
        effects.append(_PendingEffect(RuleEffect("complete_night_choice", {}), action.actor))
        return tuple(effects)

    def _evil_information_effects(self, *, minions: bool) -> tuple[_PendingEffect, ...]:
        if len(self.state.players) < 7:
            return ()
        seated = sorted(self.state.players.values(), key=lambda player: player.seat)
        demon_ids = tuple(player.player_id for player in seated if player.role == "imp")
        minion_ids = tuple(
            player.player_id
            for player in seated
            if ROLE_CATEGORIES.get(player.role) == "minion"
        )
        effects: list[_PendingEffect] = []
        if minions:
            for minion_id in minion_ids:
                effects.append(
                    _PendingEffect(
                        RuleEffect(
                            "emit_event",
                            {
                                "type": "evil.info_received",
                                "actor": minion_id,
                                "audience": Audience.player(minion_id),
                                "payload": {
                                    "demon_ids": demon_ids,
                                    "other_minion_ids": tuple(
                                        item for item in minion_ids if item != minion_id
                                    ),
                                },
                            },
                        ),
                        minion_id,
                    )
                )
        else:
            in_use_roles = {
                identity
                for player in seated
                for identity in (player.role, player.perceived_identity)
            }
            not_in_play_good = tuple(
                role
                for role, category in ROLE_CATEGORIES.items()
                if category in {"townsfolk", "outsider"}
                and role not in in_use_roles
            )
            for demon_id in demon_ids:
                effects.append(
                    _PendingEffect(
                        RuleEffect(
                            "select_demon_bluffs",
                            {
                                "demon_id": demon_id,
                                "minion_ids": minion_ids,
                                "options": tuple(combinations(not_in_play_good, 3)),
                            },
                        ),
                        demon_id,
                    )
                )
        return tuple(effects)

    def _night_actor(self, role: str) -> str | None:
        for player in sorted(self.state.players.values(), key=lambda item: item.seat):
            perceived = player.perceived_identity if player.role == "drunk" else player.role
            if perceived != role:
                continue
            if role == "ravenkeeper":
                if not player.alive and "killed_at_night" in player.reminders:
                    return player.player_id
                continue
            if player.alive:
                return player.player_id
        return None

    def _information_options(self, role: str, actor_id: str) -> tuple[Observation, ...]:
        context = AbilityContext.from_state(self.state, actor_id)
        handler = _PASSIVE_INFORMATION_HANDLERS[role]()
        if role == "undertaker":
            return handler.legal_observations(context, self.state.role_state.executed_today)
        return handler.legal_observations(context)

    def _apply_effects_atomically(
        self,
        effects: Sequence[RuleEffect | _PendingEffect],
        *,
        actor_id: str | None = None,
    ) -> tuple[EventRecord, ...]:
        """Resolve on isolated drafts and commit state, policy, tracker, and events once."""

        original = self.state.model_copy(deep=True)
        draft = self.state.model_copy(deep=True)
        policy = deepcopy(self.storyteller)
        transaction: dict[str, Any] = {
            "tracker": deepcopy(self._nominations),
            "deferred_end_day": None,
        }
        new_events: list[EventRecord] = []
        demon_deaths: list[_DemonDeath] = []
        pending = tuple(
            effect
            if isinstance(effect, _PendingEffect)
            else _PendingEffect(effect, actor_id)
            for effect in effects
        )

        def emit(
            event_type: str,
            *,
            payload: dict[str, Any],
            audience: Audience,
            actor: str | None = None,
            phase: str | None = None,
        ) -> None:
            new_events.append(
                EventRecord(
                    phase=draft.phase if phase is None else phase,
                    type=event_type,
                    actor=actor,
                    audience=audience,
                    payload=decision_jsonable(payload),
                )
            )

        def choose(request: DecisionRequest) -> Any:
            selected = policy.choose(request)
            new_events.append(_decision_event(policy.decisions[-1], phase=draft.phase))
            return selected

        def healthy(player_id: str | None) -> bool:
            if player_id is None or player_id not in original.players:
                return False
            player = original.players[player_id]
            return player.alive and player.role != "drunk" and "poisoned" not in player.reminders

        def clear_protection() -> None:
            target_id = draft.role_state.protected_player_id
            if target_id in draft.players:
                draft.players[target_id].reminders.discard("protected")
            draft.role_state.protected_player_id = None
            draft.role_state.protected_by_player_id = None

        def clear_poison() -> None:
            target_id = draft.role_state.poisoned_player_id
            if target_id in draft.players:
                draft.players[target_id].reminders.discard("poisoned")
            draft.role_state.poisoned_player_id = None
            draft.role_state.poisoned_by_player_id = None

        def actual_death(
            target_id: str,
            *,
            source: str,
            cause: str,
        ) -> bool:
            player = draft.players[target_id]
            if not player.alive:
                emit(
                    "death.failed",
                    payload={"player_id": target_id, "source": source, "reason": "already_dead"},
                    audience=Audience.observer(),
                )
                return False
            alive_before = draft.alive_count
            player.alive = False
            player.dead_vote_available = True
            draft.alive_count -= 1
            if draft.role_state.protected_player_id == target_id:
                clear_protection()
            if draft.role_state.poisoned_by_player_id == target_id:
                clear_poison()
            if draft.role_state.protected_by_player_id == target_id:
                clear_protection()
            if draft.phase == "night":
                player.reminders.add("killed_at_night")
                if target_id not in draft.role_state.night_deaths:
                    draft.role_state.night_deaths.append(target_id)
            emit(
                "player.died",
                payload={"player_id": target_id, "source": source, "cause": cause},
                audience=Audience.observer() if draft.phase == "night" else Audience.public(),
                actor=target_id,
            )
            if player.role == "imp":
                demon_deaths.append(_DemonDeath(target_id, alive_before))
            return True

        def resolve_kill(effect: RuleEffect, source_actor: str | None) -> None:
            target_id = effect.payload["target_id"]
            if target_id not in draft.players:
                raise ValueError(f"unknown kill target: {target_id}")
            source = str(effect.payload.get("source", "ability"))
            required_category = effect.payload.get("requires_registration_category")
            if required_category is not None:
                context_actor = source_actor or target_id
                options = registrations_for(
                    target_id,
                    RegistrationQuery.CHARACTER,
                    AbilityContext.from_state(draft, context_actor),
                )
                selected: Registration = choose(
                    DecisionRequest(
                        key=f"registration:{source}:{context_actor}:{target_id}:day-{draft.day}",
                        options=options,
                        reason_code="registration_resolution",
                    )
                )
                if selected.category != required_category:
                    emit(
                        "ability.no_effect",
                        payload={"source": source, "target_id": target_id},
                        audience=Audience.observer(),
                    )
                    return

            if source == "imp" and draft.players[target_id].alive:
                if draft.role_state.protected_player_id == target_id:
                    emit(
                        "death.prevented",
                        payload={"player_id": target_id, "source": "monk"},
                        audience=Audience.observer(),
                    )
                    return
                target = draft.players[target_id]
                if target.role == "soldier" and healthy(target_id):
                    emit(
                        "death.prevented",
                        payload={"player_id": target_id, "source": "soldier"},
                        audience=Audience.observer(),
                    )
                    return
                if target.role == "mayor" and healthy(target_id):
                    redirect_effects = Mayor().on_night_attack(
                        AbilityContext.from_state(draft, target_id)
                    )
                    if redirect_effects:
                        redirect = redirect_effects[0]
                        options = (
                            redirect.payload["normal_target_id"],
                            *redirect.payload["candidate_ids"],
                        )
                        selected_target = choose(
                            DecisionRequest(
                                key=f"mayor:redirect:day-{draft.day}:{target_id}",
                                options=tuple(options),
                                reason_code="mayor_night_death",
                            )
                        )
                        if selected_target != target_id:
                            emit(
                                "death.redirected",
                                payload={
                                    "from_player_id": target_id,
                                    "to_player_id": selected_target,
                                },
                                audience=Audience.observer(),
                            )
                            actual_death(
                                selected_target,
                                source="mayor",
                                cause="night_death_redirect",
                            )
                            return
            actual_death(target_id, source=source, cause="kill")

        def transform(effect: RuleEffect, source_actor: str | None) -> None:
            if (
                effect.payload.get("source") == "imp_self_kill"
                and source_actor in draft.players
                and draft.players[source_actor].alive
            ):
                emit(
                    "effect.suppressed",
                    payload={
                        "effect": "transform_role",
                        "actor_id": source_actor,
                        "reason": "imp_self_kill_prevented",
                    },
                    audience=Audience.observer(),
                )
                return
            target_id = effect.payload.get("player_id")
            if target_id is None:
                candidates = tuple(effect.payload.get("candidate_ids", ()))
                if not candidates:
                    raise ValueError("transform_role requires a player or candidate")
                target_id = choose(
                    DecisionRequest(
                        key=f"transform:{effect.payload.get('source', 'ability')}:day-{draft.day}",
                        options=candidates,
                        reason_code="role_successor",
                    )
                )
            if target_id not in draft.players or not draft.players[target_id].alive:
                raise ValueError("role successor must be a living player")
            role = effect.payload["role"]
            source = effect.payload.get("source")
            player = draft.players[target_id]
            if (
                draft.role_state.poisoned_by_player_id == target_id
                and role != "poisoner"
            ):
                clear_poison()
            player.role = role
            player.perceived_identity = role
            player.alignment = GameState.ROLE_ALIGNMENTS[role]
            player.known_alignment = GameState.ROLE_ALIGNMENTS[role]
            player.reminders.discard("became_imp")
            emit(
                "role.transformed",
                payload={"player_id": target_id, "role": role, "source": source},
                audience=Audience.observer(),
                actor=target_id,
            )
            defer_notification = (
                role == "imp"
                and source == "scarlet_woman"
                and draft.phase.startswith("day")
            )
            if defer_notification:
                player.reminders.add("became_imp")
            else:
                emit(
                    "role.changed_private",
                    payload={"player_id": target_id, "role": role, "source": source},
                    audience=Audience.player(target_id),
                    actor=target_id,
                )

        def finish_day(payload: dict[str, Any]) -> None:
            ended_phase = draft.phase
            draft.phase = "night"
            draft.role_state.first_night = False
            draft.role_state.night_step_index = 0
            draft.role_state.pending_night_role = None
            draft.role_state.pending_night_actor_id = None
            draft.role_state.night_deaths = []
            transaction["tracker"] = None
            emit(
                "day.ended",
                payload={"day": draft.day, "reason": payload["reason"]},
                audience=Audience.public(),
                phase=ended_phase,
            )

        def apply_one(item: _PendingEffect) -> None:
            effect = item.effect
            kind = effect.kind
            payload = effect.payload
            if payload.get("requires_healthy") and not healthy(item.actor_id):
                emit(
                    "effect.suppressed",
                    payload={"effect": kind, "actor_id": item.actor_id},
                    audience=Audience.observer(),
                )
                return
            if kind == "emit_event":
                emit(
                    payload["type"],
                    payload=payload.get("payload", {}),
                    audience=payload["audience"],
                    actor=payload.get("actor"),
                    phase=payload.get("phase"),
                )
            elif kind == "replace_nomination_tracker":
                transaction["tracker"] = payload["tracker"]
            elif kind == "set_phase":
                draft.phase = payload["phase"]
            elif kind == "consume_dead_vote":
                player = draft.players[payload["player_id"]]
                if not player.dead_vote_available:
                    raise IllegalAction("dead vote already spent")
                player.dead_vote_available = False
            elif kind == "mark_used":
                player_id = payload["player_id"]
                reminder = f"{payload['ability']}_used"
                draft.players[player_id].reminders.add(reminder)
                emit(
                    "ability.used",
                    payload={"player_id": player_id, "ability": payload["ability"]},
                    audience=(
                        Audience.public()
                        if payload.get("public", True)
                        else Audience.observer()
                    ),
                    actor=player_id,
                )
            elif kind == "protect":
                clear_protection()
                target_id = payload["target_id"]
                draft.role_state.protected_player_id = target_id
                draft.role_state.protected_by_player_id = item.actor_id
                draft.players[target_id].reminders.add("protected")
                emit(
                    "protection.applied",
                    payload={"target_id": target_id, "source": payload.get("source")},
                    audience=Audience.observer(),
                )
            elif kind == "poison":
                clear_poison()
                target_id = payload["target_id"]
                draft.role_state.poisoned_player_id = target_id
                draft.role_state.poisoned_by_player_id = item.actor_id
                draft.players[target_id].reminders.add("poisoned")
                emit(
                    "poison.applied",
                    payload={"target_id": target_id, "source": payload.get("source")},
                    audience=Audience.observer(),
                )
            elif kind == "clear_poison":
                clear_poison()
            elif kind == "set_master":
                draft.role_state.butler_masters[payload["butler_id"]] = payload["master_id"]
                emit(
                    "butler.master_set",
                    payload={
                        "butler_id": payload["butler_id"],
                        "master_id": payload["master_id"],
                    },
                    audience=Audience.observer(),
                )
            elif kind == "request_night_choice":
                draft.role_state.pending_night_role = payload["role"]
                draft.role_state.pending_night_actor_id = payload["actor_id"]
                emit(
                    "ability.choice_requested",
                    payload={
                        "role": payload["role"],
                        "actor_id": payload["actor_id"],
                        "legal_targets": payload["legal_targets"],
                    },
                    audience=Audience.player(payload["actor_id"]),
                    actor=payload["actor_id"],
                )
            elif kind == "complete_night_choice":
                draft.role_state.pending_night_role = None
                draft.role_state.pending_night_actor_id = None
                draft.role_state.night_step_index += 1
            elif kind == "advance_night_step":
                draft.role_state.night_step_index += 1
            elif kind == "select_observation":
                options = tuple(payload["options"])
                selected: Observation = choose(
                    DecisionRequest(
                        key=(
                            f"information:{payload['role']}:{payload['actor_id']}:"
                            f"day-{draft.day}:step-{draft.role_state.night_step_index}"
                        ),
                        options=options,
                        reason_code=(
                            "false_information" if any(not item.truthful for item in options) else "legal_information"
                        ),
                    )
                )
                observation_payload = decision_jsonable(selected)
                observation_payload.pop("options", None)
                observation_payload.pop("truthful", None)
                observation_payload.pop("private_to", None)
                if payload.get("target_ids"):
                    observation_payload["player_ids"] = decision_jsonable(payload["target_ids"])
                emit(
                    "information.received",
                    payload=observation_payload,
                    audience=Audience.player(payload["actor_id"]),
                    actor=payload["actor_id"],
                )
            elif kind == "select_demon_bluffs":
                selected = choose(
                    DecisionRequest(
                        key=f"demon_bluffs:{payload['demon_id']}",
                        options=tuple(payload["options"]),
                        reason_code="demon_bluffs",
                    )
                )
                emit(
                    "evil.info_received",
                    payload={"minion_ids": payload["minion_ids"], "bluffs": selected},
                    audience=Audience.player(payload["demon_id"]),
                    actor=payload["demon_id"],
                )
            elif kind == "resolve_virgin_trigger":
                options = tuple(payload["registration_options"])
                selected: Registration = choose(
                    DecisionRequest(
                        key=(
                            f"virgin:registration:{item.actor_id}:"
                            f"{payload['nominator_id']}:day-{draft.day}"
                        ),
                        options=options,
                        reason_code="virgin_registration",
                    )
                )
                if selected.category == payload["required_category"]:
                    apply_one(
                        _PendingEffect(
                            RuleEffect(
                                "execute",
                                {
                                    "target_id": payload["nominator_id"],
                                    "source": "virgin",
                                },
                            ),
                            item.actor_id,
                        )
                    )
                    apply_one(_PendingEffect(RuleEffect("end_day", {"reason": "virgin"})))
            elif kind == "kill":
                resolve_kill(effect, item.actor_id)
            elif kind == "execute":
                target_id = payload["target_id"]
                emit(
                    "execution.resolved",
                    payload={"player_id": target_id, "source": payload.get("source")},
                    audience=Audience.public(),
                    actor=target_id,
                )
                target_before = draft.model_copy(deep=True)
                died = actual_death(
                    target_id,
                    source=str(payload.get("source", "execution")),
                    cause="execution",
                )
                draft.role_state.executed_today = target_id if died else None
                if died and target_before.players[target_id].role == "saint":
                    for saint_effect in Saint().on_executed(
                        AbilityContext.from_state(target_before, target_id)
                    ):
                        apply_one(_PendingEffect(saint_effect, target_id))
            elif kind == "transform_role":
                transform(effect, item.actor_id)
            elif kind == "declare_winner":
                if draft.role_state.winner is None:
                    draft.role_state.winner = payload["winner"]
                    draft.role_state.winner_reason = payload["reason"]
            elif kind == "end_day":
                if transaction["deferred_end_day"] is not None:
                    raise ValueError("day end is already pending")
                transaction["deferred_end_day"] = payload
            elif kind == "notify_new_imp":
                player_id = payload["player_id"]
                draft.players[player_id].reminders.discard("became_imp")
                emit(
                    "role.change_notified",
                    payload={"role": "imp", "source": payload["source"]},
                    audience=Audience.player(player_id),
                    actor=player_id,
                )
            elif kind == "dawn":
                deaths = tuple(draft.role_state.night_deaths)
                emit(
                    "night.deaths_announced",
                    payload={"player_ids": deaths},
                    audience=Audience.public(),
                )
                clear_protection()
                for player in draft.players.values():
                    player.reminders.discard("killed_at_night")
                if not draft.role_state.first_night:
                    draft.day += 1
                draft.phase = "day.discussion"
                draft.role_state.first_night = False
                draft.role_state.night_step_index = 0
                draft.role_state.pending_night_role = None
                draft.role_state.pending_night_actor_id = None
                draft.role_state.night_deaths = []
                draft.role_state.executed_today = None
                transaction["tracker"] = NominationTracker(
                    alive_count=draft.alive_count,
                    nomination_id_prefix=f"nom-day-{draft.day}",
                )
            else:
                raise ValueError(f"unsupported rule effect: {kind}")

        for item in pending:
            apply_one(item)

        for demon_death in demon_deaths:
            if _living_imps(draft):
                continue
            if demon_death.alive_before >= 5:
                for player in sorted(draft.players.values(), key=lambda candidate: candidate.seat):
                    if player.role != "scarlet_woman" or not player.alive:
                        continue
                    hook_state = draft.model_copy(deep=True)
                    hook_state.alive_count = demon_death.alive_before
                    hook_effects = ScarletWoman().on_demon_death(
                        AbilityContext.from_state(hook_state, player.player_id),
                        demon_id=demon_death.demon_id,
                    )
                    for hook_effect in hook_effects:
                        apply_one(_PendingEffect(hook_effect, player.player_id))
                    if _living_imps(draft):
                        break
            continuation_available = bool(_living_imps(draft))
            demon_effects = Imp.on_demon_death(
                AbilityContext.from_state(draft, demon_death.demon_id),
                continuation_available=continuation_available,
            )
            for demon_effect in demon_effects:
                apply_one(_PendingEffect(demon_effect, demon_death.demon_id))

        if transaction["deferred_end_day"] is not None:
            finish_day(transaction["deferred_end_day"])

        winner = _winner_for(draft)
        if winner is not None and not draft.role_state.game_ended:
            reason = draft.role_state.winner_reason or (
                "two_alive" if winner == "evil" else "demon_dead"
            )
            draft.role_state.winner = winner
            draft.role_state.winner_reason = reason
            draft.role_state.game_ended = True
            emit(
                "game.ended",
                payload={"winner": winner, "reason": reason},
                audience=Audience.public(),
            )

        committed = tuple(new_events)
        self.state = draft
        self.storyteller = policy
        self._nominations = transaction["tracker"]
        self._events.extend(committed)
        return committed


def _tagged(effects: Iterable[RuleEffect], actor_id: str) -> tuple[_PendingEffect, ...]:
    return tuple(_PendingEffect(effect, actor_id) for effect in effects)


def _living_imps(state: GameState) -> tuple[str, ...]:
    return tuple(
        player.player_id
        for player in state.players.values()
        if player.alive and player.role == "imp"
    )


def _winner_for(state: GameState) -> str | None:
    if state.role_state.winner is not None:
        return state.role_state.winner
    if not _living_imps(state):
        return "good"
    if state.alive_count <= 2:
        return "evil"
    return None


def _decision_event(record: DecisionRecord, *, phase: str) -> EventRecord:
    return EventRecord(
        phase=phase,
        type="storyteller.decision",
        audience=Audience.observer(),
        payload={
            "request_key": record.request_key,
            "options": decision_jsonable(record.options),
            "selected": decision_jsonable(record.selected),
            "reason_code": record.reason_code,
        },
    )


def _select_setup_roles(player_count: int, policy: StorytellerPolicy) -> tuple[str, ...]:
    townsfolk_count, outsider_count, minion_count, demon_count = setup_counts(player_count)
    include_baron = policy.choose(
        DecisionRequest(
            key=f"setup:{player_count}:include_baron",
            options=(False, True),
            reason_code="setup_baron_presence",
        )
    )
    if include_baron:
        townsfolk_count -= 2
        outsider_count += 2

    selected: list[str] = []
    selected.extend(
        _choose_without_replacement(
            _TOWNSFOLK,
            townsfolk_count,
            policy,
            key=f"setup:{player_count}:townsfolk",
        )
    )
    selected.extend(
        _choose_without_replacement(
            _OUTSIDERS,
            outsider_count,
            policy,
            key=f"setup:{player_count}:outsider",
        )
    )
    if include_baron:
        selected.append("baron")
        remaining_minions = tuple(role for role in _MINIONS if role != "baron")
        minion_count -= 1
    else:
        remaining_minions = tuple(role for role in _MINIONS if role != "baron")
    selected.extend(
        _choose_without_replacement(
            remaining_minions,
            minion_count,
            policy,
            key=f"setup:{player_count}:minion",
        )
    )
    selected.extend(
        _choose_without_replacement(
            _DEMONS,
            demon_count,
            policy,
            key=f"setup:{player_count}:demon",
        )
    )
    return tuple(selected)


def _choose_without_replacement(
    options: Sequence[str],
    count: int,
    policy: StorytellerPolicy,
    *,
    key: str,
) -> tuple[str, ...]:
    remaining = list(options)
    selected: list[str] = []
    for index in range(count):
        choice = policy.choose(
            DecisionRequest(
                key=f"{key}:{index}",
                options=tuple(remaining),
                reason_code="setup_role_selection",
            )
        )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)
