"""Tests for evaluate/validation.py — expanding-window harness + leakage tripwire.

Expanding-window by season only; never random k-fold (PRD §9). The leakage test
is checkpoint P1: feed the actual result as a feature and validation RPS must
collapse to near zero — if it doesn't, the harness is broken, not the model.
"""

import numpy as np
import pandas as pd

from eplforecast.evaluate.validation import (
    expanding_window_splits,
    market_probs,
    run_baseline_validation,
    run_validation,
)
from eplforecast.features.build import FEATURE_COLUMNS, build_features


def _seasonal_df():
    rows = []
    d = pd.Timestamp("2016-08-01")
    for s in ("2016-17", "2017-18", "2018-19", "2019-20"):
        for _ in range(2):
            rows.append({"season": s, "date": d})
            d += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def _synthetic_matches(n_seasons=5, n_teams=10, seed=0):
    """Poisson goals from latent team strengths + a home-field term, plus a
    near-oracle 'market' (odds from the true generative probabilities + a vig)."""
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(n_teams)]
    strength = {t: rng.normal(0.0, 0.35) for t in names}
    home_field = 0.30
    rows = []
    for s in range(n_seasons):
        season = f"{2015 + s}-{str(2016 + s)[2:]}"
        date = pd.Timestamp("2015-08-01") + pd.DateOffset(years=s)
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                h, a = names[i], names[j]
                lam_h = np.exp(0.15 + home_field + strength[h] - strength[a])
                lam_a = np.exp(0.15 + strength[a] - strength[h])
                hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
                ph, pd_, pa = _true_probs(lam_h, lam_a)
                margin = 1.05  # 5% overround
                rows.append({
                    "season": season, "date": date, "home_team": h, "away_team": a,
                    "home_goals": hg, "away_goals": ag,
                    "odds_home": 1.0 / (ph * margin),
                    "odds_draw": 1.0 / (pd_ * margin),
                    "odds_away": 1.0 / (pa * margin),
                })
                date += pd.Timedelta(hours=6)
    return pd.DataFrame(rows)


def _true_probs(lam_h, lam_a, max_goals=10):
    from scipy.stats import poisson
    gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
    grid = np.outer(gh, ga)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# ---- expanding-window splits ----------------------------------------------
def test_expanding_window_splits_are_chronological_and_grow():
    splits = list(expanding_window_splits(_seasonal_df(), min_train_seasons=1))
    assert [s for _, _, s in splits] == ["2017-18", "2018-19", "2019-20"]
    sizes = [len(tr) for tr, _, _ in splits]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_expanding_window_respects_min_train_seasons():
    tested = [s for _, _, s in expanding_window_splits(_seasonal_df(), min_train_seasons=3)]
    assert tested == ["2019-20"]


# ---- market probabilities --------------------------------------------------
def test_market_probs_are_devigged():
    df = pd.DataFrame({"odds_home": [2.0], "odds_draw": [4.0], "odds_away": [4.0]})
    p = market_probs(df)
    assert p.shape == (1, 3)
    assert np.allclose(p[0], [0.5, 0.25, 0.25])


def test_market_probs_marks_invalid_odds_as_nan_instead_of_crashing():
    # Bad/missing odds (<= 1.0 or NaN) must not crash the weekly run; that row's
    # market probabilities are NaN so the harness can filter it out.
    df = pd.DataFrame({
        "odds_home": [1.0, 2.0, np.nan],
        "odds_draw": [4.0, 4.0, 3.5],
        "odds_away": [4.0, 4.0, 3.0],
    })
    p = market_probs(df)
    assert np.isnan(p[0]).all()   # odds_home == 1.0 -> invalid
    assert np.allclose(p[1], [0.5, 0.25, 0.25])
    assert np.isnan(p[2]).all()   # odds_home NaN -> invalid


# ---- end-to-end harness ----------------------------------------------------
def test_harness_reports_baseline_and_market_rps():
    results = run_baseline_validation(_synthetic_matches(seed=0), min_train_seasons=2)
    assert {"season", "baseline_rps", "market_rps"} <= set(results.columns)
    assert len(results) == 3  # seasons 3,4,5 tested after 2 train seasons
    assert results["baseline_rps"].between(0.0, 0.4).all()
    assert results["market_rps"].between(0.0, 0.4).all()


def test_near_oracle_market_beats_the_baseline():
    results = run_baseline_validation(_synthetic_matches(seed=0), min_train_seasons=2)
    assert results["market_rps"].mean() < results["baseline_rps"].mean()


# ---- checkpoint P1: leakage tripwire --------------------------------------
def test_leaking_the_result_collapses_validation_rps():
    m = _synthetic_matches(seed=1)
    cutoff = m["date"].max() + pd.Timedelta(days=1)
    feats = build_features(m, cutoff)

    honest = run_validation(feats, feature_columns=FEATURE_COLUMNS, min_train_seasons=2)
    for k in range(3):
        feats[f"leak_{k}"] = (feats["outcome"] == k).astype(float)
    leaked = run_validation(
        feats,
        feature_columns=[*FEATURE_COLUMNS, "leak_0", "leak_1", "leak_2"],
        min_train_seasons=2,
    )

    honest_rps = honest["baseline_rps"].mean()
    leaked_rps = leaked["baseline_rps"].mean()
    assert leaked_rps < 0.05          # collapsed to near zero
    assert leaked_rps < 0.3 * honest_rps
