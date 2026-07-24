"""Assembles the model matrix from leak-free features.

`build_features` takes an explicit ``cutoff`` and asserts that no input row is at
or after it — the guardrail from §4.2. All features are leak-free by construction
(pre-match Elo via `compute_elo`, shift(1) rolling form via `team_rolling`), so a
row's features depend only on matches strictly before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import compute_elo
from .rolling import team_rolling

# Stable feature set the models train against.
FEATURE_COLUMNS = [
    "elo_diff",
    "home_gf_roll",
    "home_ga_roll",
    "home_pts_roll",
    "away_gf_roll",
    "away_ga_roll",
    "away_pts_roll",
]

_ID_COLUMNS = ("date", "season", "home_team", "away_team")
_ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")


def build_features(
    matches: pd.DataFrame,
    cutoff,
    window: int = 5,
    k: float = 20.0,
    home_adv: float = 60.0,
) -> pd.DataFrame:
    """Return a feature matrix (indexed like ``matches``) with FEATURE_COLUMNS,
    an integer ``outcome`` target (0=home, 1=draw, 2=away), identifying columns,
    and any closing-odds columns passed through for the market benchmark.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    at_or_after = int((matches["date"] >= cutoff_ts).sum())
    if at_or_after:
        raise ValueError(
            f"build_features received {at_or_after} row(s) at/after cutoff "
            f"{cutoff_ts.date()}; the leakage contract requires all matches "
            f"strictly before the cutoff."
        )

    elo = compute_elo(matches, k=k, home_adv=home_adv)
    roll = team_rolling(matches, window=window)

    hg = matches["home_goals"].to_numpy()
    ag = matches["away_goals"].to_numpy()
    outcome = np.where(hg > ag, 0, np.where(hg < ag, 2, 1))

    out = pd.DataFrame(index=matches.index)
    for col in _ID_COLUMNS:
        if col in matches.columns:
            out[col] = matches[col]
    out["outcome"] = outcome
    out["elo_diff"] = elo
    for col in roll.columns:
        out[col] = roll[col]
    for col in _ODDS_COLUMNS:
        if col in matches.columns:
            out[col] = matches[col]
    return out
