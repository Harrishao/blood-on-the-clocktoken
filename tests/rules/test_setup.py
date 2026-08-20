import pytest

from clocktower.rules.setup import build_setup, setup_counts


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (5, (3, 0, 1, 1)),
        (6, (3, 1, 1, 1)),
        (7, (5, 0, 1, 1)),
        (8, (5, 1, 1, 1)),
        (9, (5, 2, 1, 1)),
        (10, (7, 0, 2, 1)),
        (11, (7, 1, 2, 1)),
        (12, (7, 2, 2, 1)),
        (13, (9, 0, 3, 1)),
        (14, (9, 1, 3, 1)),
        (15, (9, 2, 3, 1)),
    ],
)
def test_official_setup_counts(count, expected):
    """A wrong Trouble Brewing distribution must reject valid role selections."""

    assert setup_counts(count) == expected


def test_setup_rejects_player_counts_outside_official_range():
    """A game outside 5--15 players has no official distribution."""

    with pytest.raises(ValueError, match="5 to 15"):
        build_setup(4, (), seed=17)


def test_baron_adjusts_categories_before_validating_selected_roles():
    """Treating Baron as an ordinary Minion would wrongly reject this legal setup."""

    result = build_setup(
        5,
        ("washerwoman", "recluse", "drunk", "baron", "imp"),
        seed=17,
    )

    assert result.category_counts == (1, 2, 1, 1)
    assert set(result.roles_by_seat) == {"washerwoman", "recluse", "drunk", "baron", "imp"}


def test_setup_rejects_roles_that_do_not_match_baron_adjusted_counts():
    """A Baron setup with the base zero-Outsider distribution is illegal."""

    with pytest.raises(ValueError, match="category counts"):
        build_setup(
            5,
            ("washerwoman", "chef", "investigator", "baron", "imp"),
            seed=17,
        )


def test_setup_seats_are_reproducible_from_seed_only():
    """Changing unrelated process randomness must not change the role seating."""

    roles = ("washerwoman", "chef", "investigator", "poisoner", "imp")

    first = build_setup(5, roles, seed=41)
    second = build_setup(5, roles, seed=41)

    assert first.roles_by_seat == second.roles_by_seat
    assert tuple(seat.seat for seat in first.seats) == (0, 1, 2, 3, 4)
