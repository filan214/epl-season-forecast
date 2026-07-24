"""Half-life grid search on out-of-sample RPS (spec §4, guide P2).

The decay rate is NOT estimated inside the model — the weighted likelihood is
a pseudo-likelihood and sampling xi is near-degenerate. Instead: fit once per
candidate half-life per expanding-window split, score held-out seasons by RPS,
take the argmin. Run once pre-season; never re-tune weekly (leakage vector).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.bayes_goals import fit_bayes, predict_outcome_probs
from .metrics import rps
from .validation import expanding_window_splits

# Spec §4's grid; np.inf = no decay.
HALF_LIFE_GRID = (60, 90, 120, 180, 270, 365, np.inf)


@dataclass(frozen=True)
class HalfLifeResult:
    best_half_life: float
    table: pd.DataFrame  # one row per grid value: half_life, rps, n_matches


def _outcomes(df: pd.DataFrame) -> np.ndarray:
    hg = df["home_goals"].to_numpy()
    ag = df["away_goals"].to_numpy()
    return np.where(hg > ag, 0, np.where(hg < ag, 2, 1))


def select_half_life(
    matches: pd.DataFrame,
    grid: tuple = HALF_LIFE_GRID,
    min_train_seasons: int = 3,
    fit_fn=None,
    sample_kwargs: dict | None = None,
) -> HalfLifeResult:
    """Grid-search the decay half-life on held-out-season RPS.

    `fit_fn` defaults to the real `fit_bayes`; it is injectable so the
    orchestration can be tested without MCMC. Ties resolve to the first
    (shortest) grid value.
    """
    fit_fn = fit_fn or fit_bayes
    sample_kwargs = sample_kwargs or {}
    splits = list(expanding_window_splits(matches, min_train_seasons))
    if not splits:
        raise ValueError(
            f"half-life search needs at least {min_train_seasons + 1} seasons "
            f"(one held-out season); got {matches['season'].nunique()}"
        )

    records = []
    for half_life in grid:
        all_probs, all_outcomes = [], []
        for train_idx, test_idx, _season in splits:
            train, test = matches.loc[train_idx], matches.loc[test_idx]
            # Teams promoted into the test season carry prior-shrunk strengths.
            teams = sorted(
                set(train["home_team"]) | set(train["away_team"])
                | set(test["home_team"]) | set(test["away_team"])
            )
            cutoff = test["date"].min()
            idata = fit_fn(train, teams, cutoff, half_life_days=half_life, **sample_kwargs)
            all_probs.append(predict_outcome_probs(idata, test, teams))
            all_outcomes.append(_outcomes(test))
        probs = np.vstack(all_probs)
        outcomes = np.concatenate(all_outcomes)
        records.append({
            "half_life": half_life,
            "rps": rps(probs, outcomes),
            "n_matches": len(outcomes),
        })

    table = pd.DataFrame.from_records(records)
    best = table.loc[table["rps"].idxmin(), "half_life"]
    return HalfLifeResult(best_half_life=float(best), table=table)
