"""shift(1)-before-rolling form windows — the leakage contract of §4.2.

`team_rolling` returns trailing rolling averages of goals-for, goals-against, and
points for each match's home and away team. Every window applies ``.shift(1)``
BEFORE ``.rolling()`` within a team's time-ordered history, so a match's rolling
value never includes that same match. This is non-negotiable and carries from the
F1 and WC predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ROLL_STATS = ("gf", "ga", "pts")


def team_rolling(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-match trailing form for home and away teams, indexed like ``matches``.

    Requires columns: date, home_team, away_team, home_goals, away_goals.
    Returns columns: {home,away}_{gf,ga,pts}_roll.
    """
    ordered = matches.sort_values("date", kind="stable")

    # One row per team-appearance (a match contributes a home row and an away row).
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        gf = ordered[f"{side}_goals"].to_numpy()
        ga = ordered[f"{opp}_goals"].to_numpy()
        pts = np.where(gf > ga, 3, np.where(gf < ga, 0, 1))
        frames.append(
            pd.DataFrame(
                {
                    "match_index": ordered.index,
                    "team": ordered[f"{side}_team"].to_numpy(),
                    "date": ordered["date"].to_numpy(),
                    "side": side,
                    "gf": gf,
                    "ga": ga,
                    "pts": pts,
                }
            )
        )

    long = pd.concat(frames, ignore_index=True).sort_values(
        ["team", "date"], kind="stable"
    )
    grouped = long.groupby("team", sort=False)
    for stat in _ROLL_STATS:
        long[f"{stat}_roll"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        sub = long[long["side"] == side].set_index("match_index")
        for stat in _ROLL_STATS:
            out[f"{side}_{stat}_roll"] = sub[f"{stat}_roll"]
    return out
