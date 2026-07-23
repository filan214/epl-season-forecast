"""Elo ratings as a feature column.

`compute_elo` returns, per match, the PRE-MATCH Elo difference
(home_rating + home_adv - away_rating). The recorded value uses ratings from
BEFORE the match is scored, so Elo for match m depends only on matches strictly
before m — the leakage contract of §4.2. The Elo rate k and home advantage are
selected offline, not learned here.
"""

from __future__ import annotations

import pandas as pd

BASE_RATING = 1500.0


def compute_elo(
    matches: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 0.0,
    base: float = BASE_RATING,
) -> pd.Series:
    """Pre-match Elo difference per match, aligned to ``matches.index``.

    Requires columns: date, home_team, away_team, home_goals, away_goals.
    Processed in date order internally; the returned Series is reindexed back to
    the input order.
    """
    ordered = matches.sort_values("date", kind="stable")
    ratings: dict[str, float] = {}
    diffs: list[float] = []

    for home, away, hg, ag in zip(
        ordered["home_team"],
        ordered["away_team"],
        ordered["home_goals"],
        ordered["away_goals"],
        strict=True,
    ):
        r_home = ratings.get(home, base)
        r_away = ratings.get(away, base)

        # Feature: what the model sees before kickoff.
        diffs.append(r_home + home_adv - r_away)

        # Update using the actual result (only affects FUTURE matches).
        if hg > ag:
            score = 1.0
        elif hg < ag:
            score = 0.0
        else:
            score = 0.5
        expected_home = 1.0 / (1.0 + 10.0 ** (-(r_home + home_adv - r_away) / 400.0))
        delta = k * (score - expected_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return pd.Series(diffs, index=ordered.index, name="elo_diff").reindex(matches.index)
