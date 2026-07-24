"""Tests for evaluate/compare.py — leak-free nested five-way comparison.

Bayes is faked (no MCMC): fit_bayes is injected and predict_outcome_probs is
monkeypatched. XGBoost/baseline/market run for real on a tiny synthetic league.
"""

import numpy as np
import pandas as pd

from eplforecast.evaluate import compare as cmp
from eplforecast.evaluate.compare import MODEL_ORDER, run_comparison


def _matches(seasons=("2019-20", "2020-21", "2021-22", "2022-23"), n_teams=6, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = {t: rng.normal(0, 0.3) for t in teams}
    rows = []
    base = pd.Timestamp("2019-08-01")
    for s_i, season in enumerate(seasons):
        d = base + pd.DateOffset(years=s_i)
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                h, a = teams[i], teams[j]
                lam_h = np.exp(0.3 + strength[h] - strength[a])
                lam_a = np.exp(0.1 + strength[a] - strength[h])
                rows.append({
                    "season": season, "date": d, "home_team": h, "away_team": a,
                    "home_goals": int(rng.poisson(lam_h)),
                    "away_goals": int(rng.poisson(lam_a)),
                    "odds_home": 2.0, "odds_draw": 3.5, "odds_away": 3.5,
                })
                d += pd.Timedelta(hours=6)
    return pd.DataFrame(rows)


def _uniform_predict(idata, fixtures, teams, **kwargs):
    return np.full((len(fixtures), 3), 1 / 3)


def test_run_comparison_builds_the_five_way_table(monkeypatch):
    monkeypatch.setattr(cmp, "predict_outcome_probs", _uniform_predict)
    res = run_comparison(_matches(), fit_bayes_fn=lambda *a, **k: "IDATA")

    assert list(res.table["model"]) == list(MODEL_ORDER)
    assert res.tune_season == "2021-22" and res.test_season == "2022-23"
    assert 0.0 <= res.w_star <= 1.0
    assert (res.table["n"] > 0).all()
    assert res.table["rps"].notna().all()


def test_weight_tuning_never_sees_the_test_season(monkeypatch):
    seen_train_seasons = []

    def spy_fit_bayes(df, teams, cutoff, half_life_days=None, **kwargs):
        seen_train_seasons.append(set(df["season"].unique()))
        return "IDATA"

    captured = {}
    real_select = cmp.select_blend_weight

    def spy_select(p_bayes, p_xgb, outcomes, **kwargs):
        captured["n"] = len(outcomes)
        return real_select(p_bayes, p_xgb, outcomes, **kwargs)

    monkeypatch.setattr(cmp, "predict_outcome_probs", _uniform_predict)
    monkeypatch.setattr(cmp, "select_blend_weight", spy_select)
    run_comparison(_matches(), fit_bayes_fn=spy_fit_bayes)

    # First bayes fit is the tuning fit: its training seasons exclude BOTH the
    # tune season (2021-22) and the test season (2022-23).
    assert "2021-22" not in seen_train_seasons[0]
    assert "2022-23" not in seen_train_seasons[0]
    # w was tuned on exactly the tune season (6 teams double round robin = 30).
    assert captured["n"] == 30
