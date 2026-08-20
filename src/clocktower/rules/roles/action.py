"""Pure Trouble Brewing protection, execution, and public-action handlers."""

from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.roles.registration import RegistrationQuery, registrations_for


class _ActionRole:
    first_night_order: int | None = None
    other_night_order: int | None = None

    @staticmethod
    def _targets(ctx: AbilityContext, *, include_actor: bool) -> list[AbilityChoice]:
        """Expose every player-facing target while the actor is alive.

        Drunk and poisoned players retain their perceived ability choices; the
        engine later checks the health precondition carried by the effect.
        """

        if not ctx.actor.alive:
            return []
        return [
            AbilityChoice(ctx.actor_id, (player_id,))
            for player_id in ctx.state.players
            if include_actor or player_id != ctx.actor_id
        ]

    @staticmethod
    def _validate_choice(ctx: AbilityContext, choice: AbilityChoice, choices: list[AbilityChoice]) -> None:
        if choice.actor_id != ctx.actor_id or choice not in choices:
            raise ValueError("illegal ability choice")


class Monk(_ActionRole):
    role = "monk"
    other_night_order = 2

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return self._targets(ctx, include_actor=False)

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        self._validate_choice(ctx, choice, self.legal_choices(ctx))
        return [
            RuleEffect(
                "protect",
                {
                    "target_id": choice.targets[0],
                    "source": self.role,
                    "expires": "dawn",
                    "requires_healthy": True,
                },
            )
        ]


class Virgin(_ActionRole):
    role = "virgin"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Virgin has a nomination trigger, not an activated choice")

    def on_nominated(self, ctx: AbilityContext) -> list[RuleEffect]:
        """Return the immediate effect only for the Virgin's first legal trigger."""

        if ctx.nominator_id is None or ctx.nominee_id != ctx.actor_id:
            raise ValueError("Virgin requires a nomination context for itself")
        if "virgin_used" in ctx.actor.reminders:
            return []
        effects = [
            RuleEffect(
                "mark_used",
                {
                    "player_id": ctx.actor_id,
                    "ability": self.role,
                    "public": False,
                },
            )
        ]
        if not ctx.is_healthy:
            return effects

        registrations = registrations_for(ctx.nominator_id, RegistrationQuery.CHARACTER, ctx)
        if not any(registration.category == "townsfolk" for registration in registrations):
            return effects
        effects.append(
            RuleEffect(
                "resolve_virgin_trigger",
                {
                    "nominator_id": ctx.nominator_id,
                    "required_category": "townsfolk",
                    "registration_options": registrations,
                    "allows_no_trigger": any(
                        registration.category != "townsfolk" for registration in registrations
                    ),
                },
            )
        )
        return effects


class Slayer(_ActionRole):
    role = "slayer"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        if "slayer_used" in ctx.actor.reminders:
            return []
        return self._targets(ctx, include_actor=True)

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        self._validate_choice(ctx, choice, self.legal_choices(ctx))
        target_id = choice.targets[0]
        effects = [RuleEffect("mark_used", {"player_id": ctx.actor_id, "ability": self.role})]
        registrations = registrations_for(target_id, RegistrationQuery.CHARACTER, ctx)
        if any(registration.category == "demon" for registration in registrations):
            effects.append(
                RuleEffect(
                    "kill",
                    {
                        "target_id": target_id,
                        "source": self.role,
                        "requires_registration_category": "demon",
                        "requires_healthy": True,
                    },
                )
            )
        return effects


class Soldier(_ActionRole):
    role = "soldier"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Soldier has a passive Demon-attack trigger")

    def on_demon_attack(self, ctx: AbilityContext) -> list[RuleEffect]:
        return [
            RuleEffect(
                "prevent_death",
                {"target_id": ctx.actor_id, "source": self.role, "requires_healthy": True},
            )
        ]


class Mayor(_ActionRole):
    role = "mayor"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Mayor has passive night and day-end triggers")

    def on_night_attack(self, ctx: AbilityContext) -> list[RuleEffect]:
        candidates = tuple(
            player_id
            for player_id in ctx.state.players
            if player_id != ctx.actor_id
        )
        if not candidates:
            return []
        return [
            RuleEffect(
                "redirect_death",
                {
                    "from_player_id": ctx.actor_id,
                    "candidate_ids": candidates,
                    "normal_target_id": ctx.actor_id,
                    "allow_no_redirect": True,
                    "source": self.role,
                    "requires_healthy": True,
                },
            )
        ]

    def end_of_day_effects(self, ctx: AbilityContext, *, no_execution: bool) -> list[RuleEffect]:
        if not ctx.is_healthy or not no_execution or ctx.state.alive_count != 3:
            return []
        return [RuleEffect("declare_winner", {"winner": "good", "reason": self.role})]
