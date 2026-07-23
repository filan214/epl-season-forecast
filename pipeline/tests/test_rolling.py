"""Tests for features/rolling.py — the shift(1)-before-rolling leakage contract.

Every rolling feature applies .shift(1) BEFORE .rolling(), so a match's rolling
value never includes that same match's result (IMPLEMENTATION_GUIDE.md §4.2).
"""

import numpy as np
import pandas as pd
import pytest

from eplforecast.features.rolling import team_rolling


def _matches():
    # Team A plays three home games scoring 1, 2, then 3.
    return pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "X",
         "home_goals": 1, "away_goals": 0},
        {"date": pd.Timestamp("2020-01-08"), "home_team": "A", "away_team": "Y",
         "home_goals": 2, "away_goals": 0},
        {"date": pd.Timestamp("2020-01-15"), "home_team": "A", "away_team": "Z",
         "home_goals": 3, "away_goals": 0},
    ])


def test_rolling_excludes_current_match_shift1():
    # A's third match: trailing-2 goals-for = mean(1, 2) = 1.5, NOT including 3.
    out = team_rolling(_matches(), window=2)
    assert out.loc[2, "home_gf_roll"] == pytest.approx(1.5)


def test_first_match_has_no_history():
    out = team_rolling(_matches(), window=2)
    assert np.isnan(out.loc[0, "home_gf_roll"])


def test_second_match_uses_only_the_first():
    out = team_rolling(_matches(), window=2)
    assert out.loc[1, "home_gf_roll"] == pytest.approx(1.0)


def test_points_rolling_counts_prior_wins():
    # A won matches 1 and 2 (3 pts each); trailing-2 points at match 3 = 3.0.
    out = team_rolling(_matches(), window=2)
    assert out.loc[2, "home_pts_roll"] == pytest.approx(3.0)


def test_output_indexed_like_matches_with_expected_columns():
    out = team_rolling(_matches(), window=3)
    assert list(out.index) == [0, 1, 2]
    for c in ("home_gf_roll", "home_ga_roll", "home_pts_roll",
              "away_gf_roll", "away_ga_roll", "away_pts_roll"):
        assert c in out.columns
