"""Tests for evaluate/devig.py — bookmaker odds -> normalised probabilities.

Basic multiplicative de-vigging (§7 backlog notes: Shin/power methods are v2).
"""

import numpy as np
import pytest

from eplforecast.evaluate.devig import devig


def test_devig_no_overround_recovers_implied_probs():
    # implied = .5, .25, .25 which already sums to 1 (no vig)
    assert devig(2.0, 4.0, 4.0) == pytest.approx((0.5, 0.25, 0.25))


def test_devig_result_sums_to_one():
    assert sum(devig(2.0, 3.0, 4.0)) == pytest.approx(1.0)


def test_devig_normalises_by_overround():
    implied = np.array([1 / 2.0, 1 / 3.0, 1 / 4.0])
    expected = implied / implied.sum()
    assert devig(2.0, 3.0, 4.0) == pytest.approx(tuple(expected))


def test_devig_favourite_gets_highest_probability():
    p_home, p_draw, p_away = devig(1.5, 4.0, 7.0)
    assert p_home > p_draw > p_away
