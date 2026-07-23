"""Tests for features/build.py — model-matrix assembly + the cutoff assertion."""

import pandas as pd
import pytest

from eplforecast.features.build import FEATURE_COLUMNS, build_features


def _three_matches():
    return pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "season": "2019-20", "home_team": "A",
         "away_team": "B", "home_goals": 2, "away_goals": 0},  # home win -> 0
        {"date": pd.Timestamp("2020-01-02"), "season": "2019-20", "home_team": "C",
         "away_team": "D", "home_goals": 1, "away_goals": 1},  # draw -> 1
        {"date": pd.Timestamp("2020-01-03"), "season": "2019-20", "home_team": "E",
         "away_team": "F", "home_goals": 0, "away_goals": 3},  # away win -> 2
    ])


def test_raises_if_any_row_is_at_or_after_cutoff():
    with pytest.raises(ValueError):
        build_features(_three_matches(), cutoff="2020-01-03")  # last row == cutoff


def test_accepts_when_all_rows_are_before_cutoff():
    assert len(build_features(_three_matches(), cutoff="2020-01-04")) == 3


def test_outcome_encoding_home_draw_away():
    out = build_features(_three_matches(), cutoff="2020-01-04")
    assert list(out["outcome"]) == [0, 1, 2]


def test_has_all_feature_columns_and_the_target():
    out = build_features(_three_matches(), cutoff="2020-01-04")
    assert "outcome" in out.columns
    for c in FEATURE_COLUMNS:
        assert c in out.columns


def test_index_is_preserved():
    m = _three_matches()
    out = build_features(m, cutoff="2020-01-04")
    assert list(out.index) == list(m.index)
