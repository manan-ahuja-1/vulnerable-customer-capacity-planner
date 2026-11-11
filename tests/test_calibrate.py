"""Tests 1 to 4. Every expected value is computed by hand from FIXTURE, never from the synthetic generator."""

import numpy as np
import pandas as pd
import pytest

from src import calibrate as cal

# 10 conversations; 5 warrant a specialist (rows 0, 2, 3, 5, 8). Two rows tie at 0.80.
SCORES = [0.95, 0.90, 0.80, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.10]
LABELS = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
CHARS = ["health", "", "life_events", "capability", "", "capability", "", "", "resilience", ""]
FIXTURE = pd.DataFrame({"score": SCORES, "warrants_specialist": LABELS, "characteristic": CHARS})


# Test 1: metrics at a threshold ----------------------------------------------------------------------------------
def test_metrics_at_threshold_080():
    # score >= 0.80 hands off rows 0-3: 4 handoffs, 3 of them positives.
    p = cal.metrics_at_threshold(SCORES, LABELS, 0.80)
    assert (p.handoffs, p.true_handoffs, p.positives) == (4, 3, 5)
    assert p.precision == pytest.approx(3 / 4)
    assert p.catch_rate == pytest.approx(3 / 5)
    assert p.handoffs_per_1000 == pytest.approx(400.0)


def test_metrics_at_threshold_050():
    # score >= 0.50 hands off rows 0-6: 7 handoffs, 4 positives.
    p = cal.metrics_at_threshold(SCORES, LABELS, 0.50)
    assert (p.handoffs, p.true_handoffs) == (7, 4)
    assert p.precision == pytest.approx(4 / 7)
    assert p.catch_rate == pytest.approx(4 / 5)


# Test 2: target catch rate -> fewest handoffs that reach it ----------------------------------------------------------
@pytest.mark.parametrize(
    "target, threshold, handoffs",
    [
        (0.8, 0.60, 6),  # need 4 of 5: first reached at 0.60 (rows 0-5)
        (0.6, 0.80, 4),  # need 3 of 5: reached at 0.80, the tie group included
        (1.0, 0.30, 9),  # need all 5: the lowest positive score
    ],
)
def test_target_catch_rate(target, threshold, handoffs):
    p = cal.capacity_for_target_catch_rate(SCORES, LABELS, target)
    assert p.threshold == pytest.approx(threshold)
    assert p.handoffs == handoffs
    assert p.catch_rate >= target


def test_target_is_highest_threshold_that_reaches_it():
    p = cal.capacity_for_target_catch_rate(SCORES, LABELS, 0.8)
    higher = [s for s in sorted(set(SCORES)) if s > p.threshold]
    assert all(cal.metrics_at_threshold(SCORES, LABELS, t).catch_rate < 0.8 for t in higher)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01])
def test_target_out_of_range(bad):
    with pytest.raises(ValueError):
        cal.capacity_for_target_catch_rate(SCORES, LABELS, bad)


# Test 3: capacity -> lowest threshold within capacity, and it is maximal -------------------------------------------
def test_capacity_tie_group_that_straddles_stays_out():
    # Capacity 3: 0.95 and 0.90 give 2 handoffs; the 0.80 tie group would make 4 > 3, so it stays out.
    p = cal.operating_point_for_capacity(SCORES, LABELS, 3)
    assert p.threshold == pytest.approx(0.90)
    assert p.handoffs == 2
    assert cal.metrics_at_threshold(SCORES, LABELS, 0.80).handoffs > 3  # one step lower overshoots


def test_capacity_exactly_fits_tie_group():
    p = cal.operating_point_for_capacity(SCORES, LABELS, 4)
    assert (p.threshold, p.handoffs, p.true_handoffs) == (pytest.approx(0.80), 4, 3)


def test_capacity_zero_hands_off_nothing():
    p = cal.operating_point_for_capacity(SCORES, LABELS, 0)
    assert p.threshold is None and p.handoffs == 0 and p.catch_rate == 0 and p.precision is None


@pytest.mark.parametrize("capacity", [10, 11, 1000])
def test_capacity_at_or_above_n_hands_off_everything(capacity):
    p = cal.operating_point_for_capacity(SCORES, LABELS, capacity)
    assert p.threshold == pytest.approx(0.10) and p.handoffs == 10 and p.catch_rate == 1.0


def test_capacity_point_is_maximal_on_random_ties():
    """Property check on a random fixture made here (not by the synthetic generator): within capacity, and one
    step lower would exceed it unless everything is already handed off."""
    rng = np.random.default_rng(0)
    s = rng.integers(0, 40, size=500) / 40  # coarse scores force many ties
    y = (rng.random(500) < 0.1).astype(int)
    distinct = np.sort(np.unique(s))[::-1]
    for capacity in range(0, 520, 7):
        p = cal.operating_point_for_capacity(s, y, capacity)
        assert p.handoffs <= capacity
        if p.handoffs == len(s):
            continue
        lower = distinct[0] if p.threshold is None else distinct[distinct < p.threshold].max()
        assert (s >= lower).sum() > capacity


def test_negative_capacity_rejected():
    with pytest.raises(ValueError):
        cal.operating_point_for_capacity(SCORES, LABELS, -1)


# Test 4: edge cases and the shortfall flag ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn, arg",
    [(cal.metrics_at_threshold, 0.5), (cal.operating_point_for_capacity, 3), (cal.capacity_for_target_catch_rate, 0.9)],
)
def test_no_positives_is_a_clear_error(fn, arg):
    with pytest.raises(cal.NoPositivesError):
        fn([0.9, 0.5, 0.1], [0, 0, 0], arg)


@pytest.mark.parametrize("required, staffed, expected", [(54.0, 50.0, 4.0), (50.0, 50.0, 0.0), (49.9, 50.0, 0.0)])
def test_shortfall_fires_exactly_when_need_exceeds_staffed(required, staffed, expected):
    assert cal.shortfall_per_1000(required, staffed) == pytest.approx(expected)
    assert (cal.shortfall_per_1000(required, staffed) > 0) == (required > staffed)


def test_per_1000_to_count_rounds_down():
    assert cal.per_1000_to_count(50, 20_000) == 1000
    assert cal.per_1000_to_count(50.5, 1000) == 50
    assert cal.per_1000_to_count(0, 1000) == 0


def test_misses_by_characteristic():
    # At 0.60, the only positive below threshold is row 8 (0.30, resilience).
    m = cal.misses_by_characteristic(FIXTURE, 0.60).set_index("characteristic")
    assert m.loc["resilience", "missed"] == 1 and m.loc["resilience", "positives"] == 1
    assert m.loc["capability", "missed"] == 0 and m.loc["capability", "positives"] == 2
    assert m["missed"].sum() == 1


def test_misses_with_no_handoffs_counts_every_positive():
    assert cal.misses_by_characteristic(FIXTURE, None)["missed"].sum() == 5


def test_curve_matches_hand_values():
    c = cal.curve(SCORES, LABELS).set_index("threshold")
    assert c.loc[0.80, "catch_rate"] == pytest.approx(3 / 5)
    assert c.loc[0.80, "handoffs_per_1000"] == pytest.approx(400.0)
    assert c.loc[0.60, "precision"] == pytest.approx(4 / 6)
    assert len(c) == 9  # 10 rows, one tie


# The perfect-guardrail floor and the hard ceiling (added 11 Nov 2026) ---------------------------------------------
# Hand-computed on a 10-row fixture with 4 positives.
FOUR = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]


@pytest.mark.parametrize("target, need", [(0.9, 4), (0.75, 3), (1.0, 4), (0.5, 2), (0.26, 2)])
def test_perfect_guardrail_need(target, need):
    # ceil(0.9 x 4) = ceil(3.6) = 4; 0.75 x 4 = 3 exactly; 0.26 x 4 = 1.04 -> 2
    assert cal.perfect_guardrail_need(FOUR, target) == need


@pytest.mark.parametrize("capacity, expected", [(0, 0.0), (2, 0.5), (4, 1.0), (10, 1.0)])
def test_max_catch_rate_at_capacity(capacity, expected):
    assert cal.max_catch_rate_at_capacity(FOUR, capacity) == pytest.approx(expected)


def test_ceiling_functions_need_positives():
    with pytest.raises(cal.NoPositivesError):
        cal.perfect_guardrail_need([0, 0, 0], 0.9)
    with pytest.raises(cal.NoPositivesError):
        cal.max_catch_rate_at_capacity([0, 0, 0], 5)


def test_float_error_cannot_add_one():
    # 0.56 * 100 == 56.00000000000001 in floats, whose ceiling is 57. The answer is 56.
    assert 0.56 * 100 != 56
    assert cal.positives_needed_for_target(100, 0.56) == 56
    # Same through the threshold search: top 56 scores are positives, then 44 more positives, then 100 negatives.
    scores = list(np.linspace(1.0, 0.51, 100)) + list(np.linspace(0.5, 0.01, 100))
    labels = [1] * 100 + [0] * 100
    p = cal.capacity_for_target_catch_rate(scores, labels, 0.56)
    assert (p.true_handoffs, p.handoffs) == (56, 56)


def test_target_must_be_a_whole_percent():
    with pytest.raises(ValueError):
        cal.positives_needed_for_target(100, 0.565)
    with pytest.raises(ValueError):
        cal.capacity_for_target_catch_rate(SCORES, LABELS, 0.565)


def test_real_guardrail_never_beats_the_perfect_one():
    rng = np.random.default_rng(3)
    scores, labels = rng.random(500), (rng.random(500) < 0.1).astype(int)
    for pct in range(5, 101, 5):
        p = cal.capacity_for_target_catch_rate(scores, labels, pct / 100)
        assert p.handoffs >= cal.perfect_guardrail_need(labels, pct / 100)
