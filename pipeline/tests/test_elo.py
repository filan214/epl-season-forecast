"""Tests for features/elo.py.

compute_elo returns, per match, the PRE-MATCH Elo difference
(home_rating + home_adv - away_rating). "Pre-match" is the leakage contract: a
match's feature uses only matches strictly before it.
"""

import pandas as pd
import pytest

from eplforecast.features.elo import compute_elo


def _m(rows):
    return pd.DataFrame(rows)


def test_first_meeting_diff_is_zero_without_home_adv():
    m = _m([{"date": pd.Timestamp("2020-01-01"), "home_team": "A",
             "away_team": "B", "home_goals": 1, "away_goals": 0}])
    assert compute_elo(m, k=20.0, home_adv=0.0).iloc[0] == pytest.approx(0.0)


def test_home_advantage_is_added_to_prematch_diff():
    m = _m([{"date": pd.Timestamp("2020-01-01"), "home_team": "A",
             "away_team": "B", "home_goals": 0, "away_goals": 0}])
    assert compute_elo(m, k=20.0, home_adv=100.0).iloc[0] == pytest.approx(100.0)


def test_ratings_update_after_home_win():
    m = _m([
        {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 0},
        {"date": pd.Timestamp("2020-01-08"), "home_team": "A", "away_team": "B",
         "home_goals": 0, "away_goals": 0},
    ])
    e = compute_elo(m, k=20.0, home_adv=0.0)
    assert e.iloc[0] == pytest.approx(0.0)
    assert e.iloc[1] == pytest.approx(20.0)  # A +10, B -10 after the win


def test_prematch_diff_does_not_leak_the_result():
    # A huge first-match win must not change that same match's pre-match diff.
    m = _m([{"date": pd.Timestamp("2020-01-01"), "home_team": "A",
             "away_team": "B", "home_goals": 5, "away_goals": 0}])
    assert compute_elo(m, k=40.0, home_adv=0.0).iloc[0] == pytest.approx(0.0)


def test_returns_series_aligned_to_input_index_even_if_unsorted():
    m = _m([
        {"date": pd.Timestamp("2020-01-08"), "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 1},
        {"date": pd.Timestamp("2020-01-01"), "home_team": "C", "away_team": "D",
         "home_goals": 2, "away_goals": 0},
    ])
    e = compute_elo(m, k=20.0, home_adv=0.0)
    assert list(e.index) == list(m.index)
    assert len(e) == 2
