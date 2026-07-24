"""Tests for evaluate/halflife.py — half-life grid search on out-of-sample RPS.

Spec §4: the decay rate is selected by grid search on held-out RPS, run once
pre-season — never sampled inside the model, never re-tuned weekly. The search
reuses the expanding-window-by-season protocol.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eplforecast.evaluate.halflife import HALF_LIFE_GRID, select_half_life

TEAMS6 = [f"T{i}" for i in range(6)]


def _fake_idata(teams, att, def_, mu=0.25, home_adv=0.20, rho=0.0, n_draws=8):
    posterior = xr.Dataset(
        {
            "mu": (("chain", "draw"), np.full((1, n_draws), mu)),
            "home_adv": (("chain", "draw"), np.full((1, n_draws), home_adv)),
            "rho": (("chain", "draw"), np.full((1, n_draws), rho)),
            "att": (("chain", "draw", "team"), np.tile(np.asarray(att, float), (1, n_draws, 1))),
            "def": (("chain", "draw", "team"), np.tile(np.asarray(def_, float), (1, n_draws, 1))),
        },
        coords={"chain": [0], "draw": np.arange(n_draws), "team": list(teams)},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def _seasons(strengths_by_season, seed=0, rounds=2):
    """Poisson-goal seasons; strengths_by_season maps season -> (att, def_)."""
    rng = np.random.default_rng(seed)
    mu, gamma = 0.25, 0.20
    rows = []
    for s, (season, (att, def_)) in enumerate(strengths_by_season.items()):
        date = pd.Timestamp("2020-08-01") + pd.DateOffset(years=s)
        for _ in range(rounds):
            for i in range(len(TEAMS6)):
                for j in range(len(TEAMS6)):
                    if i == j:
                        continue
                    lam_h = np.exp(mu + gamma + att[i] - def_[j])
                    lam_a = np.exp(mu + att[j] - def_[i])
                    rows.append({
                        "season": season, "date": date,
                        "home_team": TEAMS6[i], "away_team": TEAMS6[j],
                        "home_goals": int(rng.poisson(lam_h)),
                        "away_goals": int(rng.poisson(lam_a)),
                    })
                    date += pd.Timedelta(hours=8)
    return pd.DataFrame(rows)


ATT = np.array([0.3, 0.15, 0.0, 0.0, -0.15, -0.3])
DEF = np.array([0.4, 0.2, 0.0, 0.0, -0.2, -0.4])


def test_grid_search_picks_the_half_life_whose_fit_predicts_best():
    # Injected fit returns the TRUE generating parameters only at h=90 and an
    # uninformative posterior otherwise; predictions and RPS run through the
    # real code, so 90 must win the argmin.
    matches = _seasons({
        "2020-21": (ATT, DEF), "2021-22": (ATT, DEF), "2022-23": (ATT, DEF),
    }, seed=1)
    calls = []

    def fake_fit(df, teams, cutoff, half_life_days=180, **kwargs):
        calls.append((len(df), cutoff, half_life_days))
        assert (df["date"] < cutoff).all()  # leakage contract holds per call
        if half_life_days == 90:
            return _fake_idata(teams, ATT, DEF)
        return _fake_idata(teams, np.zeros(6), np.zeros(6))

    result = select_half_life(
        matches, grid=(60, 90), min_train_seasons=1, fit_fn=fake_fit
    )

    assert result.best_half_life == 90
    assert list(result.table["half_life"]) == [60, 90]
    assert len(calls) == 4  # 2 grid values x 2 test seasons
    assert result.table["rps"].between(0, 1).all()
    good = result.table.set_index("half_life")["rps"]
    assert good[90] < good[60]
    # Both grid entries were evaluated on every test-season match.
    n_test = len(matches[matches["season"] != "2020-21"])
    assert (result.table["n_matches"] == n_test).all()


def test_refuses_to_run_without_a_held_out_season():
    matches = _seasons({"2020-21": (ATT, DEF)}, seed=2)
    with pytest.raises(ValueError, match="season"):
        select_half_life(matches, grid=(60,), min_train_seasons=3)


def test_default_grid_matches_the_spec():
    assert (60, 90, 120, 180, 270, 365, np.inf) == HALF_LIFE_GRID


def test_regime_shift_favours_a_short_half_life_end_to_end():
    # Strengths flip sign from season 3 on. Training on seasons 1-3 and
    # testing on season 4: an infinite half-life averages two regimes and
    # must lose to a 60-day half-life that effectively sees only the new one.
    # Real NUTS fits (short chains) — this is the honest reason the grid
    # search exists.
    matches = _seasons({
        "2020-21": (ATT, DEF),
        "2021-22": (ATT, DEF),
        "2022-23": (-ATT, -DEF),
        "2023-24": (-ATT, -DEF),
    }, seed=3)

    result = select_half_life(
        matches, grid=(60, np.inf), min_train_seasons=3,
        sample_kwargs={"draws": 150, "tune": 150, "chains": 2, "cores": 1,
                       "random_seed": 42, "progressbar": False},
    )

    rps = result.table.set_index("half_life")["rps"]
    assert result.best_half_life == 60
    assert rps[60] < rps[np.inf]
