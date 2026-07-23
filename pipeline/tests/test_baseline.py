"""Tests for models/baseline.py — the multinomial logistic baseline.

predict_proba returns (n, 3) columns aligned to classes [0=home, 1=draw, 2=away].
"""

import numpy as np
import pandas as pd

from eplforecast.models.baseline import fit_baseline


def test_predict_proba_is_a_valid_distribution():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
    y = np.array([0, 1, 2] * 20)
    p = fit_baseline(X, y).predict_proba(X)
    assert p.shape == (60, 3)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_learns_a_separable_signal():
    y = np.array([0, 1, 2] * 30)
    X = pd.DataFrame({"f": y.astype(float)})  # feature equals the class
    pred = fit_baseline(X, y).predict_proba(X).argmax(axis=1)
    assert (pred == y).mean() > 0.9


def test_handles_nan_features_without_erroring():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=60)})
    X.loc[:5, "f1"] = np.nan  # early-season rolling features are NaN
    y = np.array([0, 1, 2] * 20)
    p = fit_baseline(X, y).predict_proba(X)
    assert not np.isnan(p).any()
