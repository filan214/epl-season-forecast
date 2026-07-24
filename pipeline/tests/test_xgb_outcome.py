"""Tests for models/xgb_outcome.py — isotonic-calibrated XGBoost outcome model."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from eplforecast.features.build import FEATURE_COLUMNS
from eplforecast.models.xgb_outcome import fit_xgb, xgb_probs


def _signal_frame(n=600, seed=0):
    """elo_diff drives the outcome so the classifier has real signal to learn."""
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 200, n)
    X = pd.DataFrame({
        "elo_diff": elo,
        "home_gf_roll": rng.normal(1.5, 0.5, n), "home_ga_roll": rng.normal(1.1, 0.4, n),
        "home_pts_roll": rng.normal(1.4, 0.6, n), "away_gf_roll": rng.normal(1.3, 0.5, n),
        "away_ga_roll": rng.normal(1.2, 0.4, n), "away_pts_roll": rng.normal(1.3, 0.6, n),
    })[FEATURE_COLUMNS]
    p_home = 1.0 / (1.0 + np.exp(-elo / 150.0))
    u = rng.random(n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.2, 1, 2))
    return X, y


def test_fit_xgb_returns_isotonic_calibrated_classifier():
    model = fit_xgb(*_signal_frame())
    assert isinstance(model, CalibratedClassifierCV)
    assert model.get_params()["method"] == "isotonic"


def test_xgb_probs_are_ordered_and_normalised():
    X, y = _signal_frame()
    p = xgb_probs(fit_xgb(X, y), X)
    assert p.shape == (len(X), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_xgb_learns_the_home_signal():
    X, y = _signal_frame()
    model = fit_xgb(X, y)
    row = {c: [X[c].mean()] for c in FEATURE_COLUMNS}
    strong_home = pd.DataFrame(row)
    strong_home["elo_diff"] = 500.0
    strong_away = pd.DataFrame(row)
    strong_away["elo_diff"] = -500.0
    p_home = xgb_probs(model, strong_home[FEATURE_COLUMNS])[0]
    p_away = xgb_probs(model, strong_away[FEATURE_COLUMNS])[0]
    assert p_home[0] > p_home[2]   # home favoured -> P(home) > P(away)
    assert p_away[2] > p_away[0]   # away favoured -> P(away) > P(home)
