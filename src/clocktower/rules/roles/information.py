"""Read-only Trouble Brewing information-role handlers."""

from __future__ import annotations

from itertools import combinations, product
from typing import Iterable

from clocktower.domain.state import GameState
from clocktower.rules.roles.base import AbilityChoice, AbilityContext, Observation, RuleEffect
from clocktower.rules.roles.registration import RegistrationQuery, registrations_for
from clocktower.rules.setup import ROLE_CATEGORIES


class _InformationRole:
    first_night_order: int | None = None
    other_night_order: int | None = None

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        return [RuleEffect("observation.requested", {"actor_id": ctx.actor_id, "targets": choice.targets})]


class Washerwoman(_InformationRole):
    role = "washerwoman"
    first_night_order = 2

    def legal_observations(self, ctx: AbilityContext) -> tuple[Observation, ...]:
        truthful = self._observations(ctx)
        return _with_false_information(ctx, truthful, self._all_structural_observations(ctx))

    @staticmethod
    def _observations(ctx: AbilityContext) -> tuple[Observation, ...]:
        return _pair_character_observations(ctx, "townsfolk")

    @staticmethod
    def _all_structural_observations(ctx: AbilityContext) -> tuple[Observation, ...]:
        return _pair_character_observations(ctx, "townsfolk", include_unregistered=True)


class Librarian(_InformationRole):
    role = "librarian"
    first_night_order = 3

    def legal_observations(self, ctx: AbilityContext) -> tuple[Observation, ...]:
        truthful = self._observations(ctx)
        structural = self._all_structural_observations(ctx)
        return _with_false_information(ctx, truthful, structural)

    @staticmethod
    def _observations(ctx: AbilityContext) -> tuple[Observation, ...]:
        observations = _pair_character_observations(ctx, "outsider")
        return observations or (Observation(kind="librarian", number=0),)

    @staticmethod
    def _all_structural_observations(ctx: AbilityContext) -> tuple[Observation, ...]:
        return (Observation(kind="librarian", number=0),) + _pair_character_observations(
            ctx, "outsider", include_unregistered=True
        )


class Investigator(_InformationRole):
    role = "investigator"
    first_night_order = 4

    def legal_observations(self, ctx: AbilityContext) -> tuple[Observation, ...]:
        truthful = _pair_character_observations(ctx, "minion")
        structural = _pair_character_observations(ctx, "minion", include_unregistered=True)
        return _with_false_information(ctx, truthful, structural)


class Chef(_InformationRole):
    role = "chef"
    first_night_order = 5

    def observe(self, game: GameState, *, actor_seat: int = 0) -> Observation:
        """Return the truthful adjacent-evil-pair count around the full circle."""

        _require_seat(game, actor_seat)
        players = _seated_players(game)
        count = sum(
            left.alignment == "evil" and right.alignment == "evil"
            for left, right in zip(players, players[1:] + players[:1], strict=True)
        )
        return Observation(kind="chef", number=count)

    def legal_observations(self, ctx: AbilityContext) -> tuple[Observation, ...]:
        truthful = _chef_observations(ctx)
        maximum = len(ctx.state.players)
        structural = tuple(Observation(kind="chef", number=value) for value in range(maximum + 1))
        return _with_false_information(ctx, truthful, structural)


class Empath(_InformationRole):
    role = "empath"
    first_night_order = 6
    other_night_order = 3

    def observe(self, game: GameState, *, actor_seat: int = 0) -> Observation:
        """Count only living neighbours, wrapping at the ends of the circle."""

        players = _seated_players(game)
        index = _require_seat(game, actor_seat)
        neighbours = (players[index - 1], players[(index + 1) % len(players)])
        count = sum(player.alive and player.alignment == "evil" for player in neighbours)
        return Observation(kind="empath", number=count)

    def legal_observations(self, ctx: AbilityContext) -> tuple[Observation, ...]:
        truthful = _empath_observations(ctx)
        structural = tuple(Observation(kind="empath", number=value) for value in range(3))
        return _with_false_information(ctx, truthful, structural)


class FortuneTeller(_InformationRole):
    role = "fortune_teller"
    first_night_order = 7
    other_night_order = 4

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return [
            AbilityChoice(ctx.actor_id, targets)
            for targets in combinations(ctx.state.players, 2)
        ]

    def choose(self, game: GameState, actor_id: str, targets: Iterable[str]) -> Observation:
        target_ids = _exactly_two_known_targets(game, targets)
        if actor_id not in game.players:
            raise ValueError(f"unknown ability actor: {actor_id}")
        yes = any(
            target_id == game.role_state.fortune_teller_red_herring
            or any(
                registration.category == "demon" and registration.truthful
                for registration in registrations_for(
                    target_id,
                    RegistrationQuery.CHARACTER,
                    AbilityContext.from_state(game, actor_id),
                )
            )
            for target_id in target_ids
        )
        return Observation(kind="fortune_teller", yes=yes, player_ids=target_ids)

    def legal_observations(self, ctx: AbilityContext, targets: Iterable[str]) -> tuple[Observation, ...]:
        target_ids = _exactly_two_known_targets(ctx.state, targets)
        truthful = (self.choose(ctx.state, ctx.actor_id, target_ids),)
        registered_results = {
            ctx.state.role_state.fortune_teller_red_herring in target_ids
            or any(registration.category == "demon" for registration in registrations)
            for registrations in product(
                *(_registered_character_options(ctx, target_id) for target_id in target_ids)
            )
        }
        registration_options = tuple(
            Observation(
                kind="fortune_teller",
                yes=yes,
                player_ids=target_ids,
                truthful=yes == truthful[0].yes,
            )
            for yes in registered_results
        )
        structural = tuple(
            Observation(kind="fortune_teller", yes=yes, player_ids=target_ids) for yes in (False, True)
        )
        legal = _unique_observations([*truthful, *registration_options])
        return _with_false_information(ctx, legal, structural)

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        self.legal_observations(ctx, choice.targets)
        return super().apply(ctx, choice)


class Undertaker(_InformationRole):
    role = "undertaker"
    other_night_order = 5

    def legal_observations(
        self, ctx: AbilityContext, executed_player_id: str | None
    ) -> tuple[Observation, ...]:
        if executed_player_id is None:
            return ()
        options = _registered_character_options(ctx, executed_player_id)
        observations = tuple(
            Observation(kind="undertaker", character=option.character, truthful=option.truthful)
            for option in options
        )
        true_observations = tuple(observation for observation in observations if observation.truthful)
        if not ctx.is_misinformed:
            return observations
        structural = tuple(
            Observation(kind="undertaker", character=role)
            for role in ROLE_CATEGORIES
        )
        return _with_false_information(ctx, observations, structural)


class Ravenkeeper(_InformationRole):
    role = "ravenkeeper"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        if ctx.actor.alive or "killed_at_night" not in ctx.actor.reminders:
            return []
        return [AbilityChoice(ctx.actor_id, (player_id,)) for player_id in ctx.state.players]

    def choose(self, ctx: AbilityContext, target_id: str) -> Observation:
        if AbilityChoice(ctx.actor_id, (target_id,)) not in self.legal_choices(ctx):
            raise ValueError("Ravenkeeper may choose only after being killed at night")
        options = tuple(
            Observation(kind="ravenkeeper", character=registration.character, truthful=registration.truthful)
            for registration in _registered_character_options(ctx, target_id)
        )
        structural = tuple(
            Observation(kind="ravenkeeper", character=role) for role in ROLE_CATEGORIES
        )
        return Observation(
            kind="ravenkeeper",
            player_ids=(target_id,),
            options=_with_false_information(ctx, options, structural),
        )

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        if len(choice.targets) != 1:
            raise ValueError("Ravenkeeper requires exactly one target")
        self.choose(ctx, choice.targets[0])
        return super().apply(ctx, choice)


class Spy(_InformationRole):
    role = "spy"
    first_night_order = 9

    def observe(self, ctx: AbilityContext) -> Observation:
        """Create a private copy of complete game truth for the Spy alone."""

        grimoire = {
            player_id: {
                "seat": player.seat,
                "role": player.role,
                "alignment": player.alignment,
                "alive": player.alive,
                "reminders": sorted(player.reminders),
            }
            for player_id, player in ctx.state.players.items()
        }
        grimoire["role_state"] = ctx.state.role_state.model_dump()
        return Observation(kind="spy_grimoire", private_to=ctx.actor_id, grimoire=grimoire)

    def legal_observations(self, ctx: AbilityContext) -> tuple[Observation, ...]:
        return (self.observe(ctx),)


class Recluse(_InformationRole):
    role = "recluse"


def _pair_character_observations(
    ctx: AbilityContext, category: str, *, include_unregistered: bool = False
) -> tuple[Observation, ...]:
    players = [player for player in _seated_players(ctx.state) if player.player_id != ctx.actor_id]
    if len(players) < 2:
        return ()
    observations: list[Observation] = []
    for named_player in players:
        registrations = _registered_character_options(ctx, named_player.player_id)
        if include_unregistered:
            registrations = _all_script_character_options(named_player.player_id, category)
        for registration in registrations:
            if registration.category != category:
                continue
            for other_player in players:
                if other_player.player_id == named_player.player_id:
                    continue
                pair = tuple(
                    sorted(
                        (named_player.player_id, other_player.player_id),
                        key=lambda player_id: ctx.state.players[player_id].seat,
                    )
                )
                observations.append(
                    Observation(
                        kind=ctx.actor.role,
                        character=registration.character,
                        player_ids=pair,
                        truthful=registration.truthful,
                    )
                )
    return _unique_observations(observations)


def _all_script_character_options(player_id: str, category: str):
    from clocktower.rules.roles.registration import Registration

    return tuple(
        Registration(player_id, "good", role, role_category, False)
        for role, role_category in ROLE_CATEGORIES.items()
        if role_category == category
    )


def _registered_character_options(ctx: AbilityContext, player_id: str):
    return registrations_for(player_id, RegistrationQuery.CHARACTER, ctx)


def _with_false_information(
    ctx: AbilityContext,
    truthful: tuple[Observation, ...],
    structural: tuple[Observation, ...],
) -> tuple[Observation, ...]:
    if not ctx.is_misinformed:
        return truthful
    truthful_keys = {_observation_key(observation) for observation in truthful}
    false = tuple(
        Observation(
            kind=observation.kind,
            number=observation.number,
            yes=observation.yes,
            character=observation.character,
            player_ids=observation.player_ids,
            options=observation.options,
            private_to=observation.private_to,
            grimoire=observation.grimoire,
            truthful=False,
        )
        for observation in structural
        if _observation_key(observation) not in truthful_keys
    )
    return _unique_observations([*truthful, *false])


def _unique_observations(observations: Iterable[Observation]) -> tuple[Observation, ...]:
    unique: list[Observation] = []
    for observation in observations:
        if observation not in unique:
            unique.append(observation)
    return tuple(unique)


def _observation_key(observation: Observation) -> tuple[object, ...]:
    return (
        observation.kind,
        observation.number,
        observation.yes,
        observation.character,
        observation.player_ids,
        observation.private_to,
    )


def _chef_observations(ctx: AbilityContext) -> tuple[Observation, ...]:
    players = _seated_players(ctx.state)
    registration_options = [
        registrations_for(player.player_id, RegistrationQuery.ALIGNMENT, ctx) for player in players
    ]
    observations: list[Observation] = []
    for registrations in product(*registration_options):
        count = sum(
            left.alignment == "evil" and right.alignment == "evil"
            for left, right in zip(registrations, registrations[1:] + registrations[:1], strict=True)
        )
        observations.append(
            Observation(kind="chef", number=count, truthful=all(item.truthful for item in registrations))
        )
    return _unique_observations(observations)


def _empath_observations(ctx: AbilityContext) -> tuple[Observation, ...]:
    players = _seated_players(ctx.state)
    index = _require_seat(ctx.state, ctx.actor.seat)
    neighbours = (players[index - 1], players[(index + 1) % len(players)])
    registration_options = [
        registrations_for(player.player_id, RegistrationQuery.ALIGNMENT, ctx)
        if player.alive
        else ()
        for player in neighbours
    ]
    observations: list[Observation] = []
    for registrations in product(*registration_options):
        count = sum(item.alignment == "evil" for item in registrations)
        observations.append(
            Observation(kind="empath", number=count, truthful=all(item.truthful for item in registrations))
        )
    return _unique_observations(observations)


def _seated_players(game: GameState):
    return sorted(game.players.values(), key=lambda player: player.seat)


def _require_seat(game: GameState, seat: int) -> int:
    players = _seated_players(game)
    for index, player in enumerate(players):
        if player.seat == seat:
            return index
    raise ValueError(f"unknown seat: {seat}")


def _exactly_two_known_targets(game: GameState, targets: Iterable[str]) -> tuple[str, str]:
    target_ids = tuple(targets)
    if len(target_ids) != 2 or len(set(target_ids)) != 2:
        raise ValueError("Fortune Teller requires exactly two distinct targets")
    unknown = set(target_ids).difference(game.players)
    if unknown:
        raise ValueError(f"unknown Fortune Teller target: {sorted(unknown)[0]}")
    return target_ids
