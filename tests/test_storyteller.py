from __future__ import annotations

import pytest

from clocktower.storyteller import DecisionRequest, StorytellerPolicy


def test_storyteller_same_seed_and_request_is_reproducible_and_audited():
    """Using process-global randomness would make the same game non-replayable."""

    request = DecisionRequest(
        key="false-info:1",
        options=("a", "b", "c"),
        reason_code="false_information",
    )

    first = StorytellerPolicy(44)
    second = StorytellerPolicy(44)

    assert first.choose(request) == second.choose(request)
    assert first.decisions == second.decisions
    assert first.decisions[0].request_key == "false-info:1"
    assert first.decisions[0].options == ("a", "b", "c")
    assert first.decisions[0].selected in request.options
    assert first.decisions[0].reason_code == "false_information"


def test_storyteller_request_choice_does_not_depend_on_unrelated_call_order():
    """A mutable RNG cursor would let unrelated decisions change a later ruling."""

    target = DecisionRequest(key="mayor:redirect:night-2", options=("mayor", "bob", "carol"))
    direct = StorytellerPolicy(17).choose(target)

    after_unrelated = StorytellerPolicy(17)
    after_unrelated.choose(DecisionRequest(key="setup:role", options=("chef", "empath")))

    assert after_unrelated.choose(target) == direct


def test_storyteller_rejects_an_empty_legal_option_set_without_logging_a_decision():
    """Inventing a fallback result would let policy decide outside the legal set."""

    policy = StorytellerPolicy(17)

    with pytest.raises(ValueError, match="at least one legal option"):
        policy.choose(DecisionRequest(key="broken", options=()))

    assert policy.decisions == ()
