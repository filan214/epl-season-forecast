"""Tests for evaluate/metrics.py — RPS, log-loss, Brier.

Probabilities are (n, 3) with columns ordered [P(home), P(draw), P(away)];
outcomes are integer class indices 0=home, 1=draw, 2=away.
"""

import numpy as np
import pytest

from eplforecast.evaluate.metrics import brier, log_loss, rps


# ---- RPS -------------------------------------------------------------------
def test_rps_perfect_prediction_is_zero():
    assert rps(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == pytest.approx(0.0)


def test_rps_known_value_home_win():
    # cumP=[.8,.9], cumO=[1,1]; ((-.2)^2+(-.1)^2)/(3-1) = 0.05/2 = 0.025
    assert rps(np.array([[0.8, 0.1, 0.1]]), np.array([0])) == pytest.approx(0.025)


def test_rps_uniform_prediction():
    # cumP=[.333,.667], cumO=[1,1]; (.444+.111)/2 = 0.27778
    assert rps(np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([0])) == pytest.approx(
        0.277778, abs=1e-5
    )


def test_rps_is_ordinal_aware():
    # When home wins, betting the draw is LESS wrong than betting the away win.
    o = np.array([0])
    assert rps(np.array([[0.1, 0.8, 0.1]]), o) < rps(np.array([[0.1, 0.1, 0.8]]), o)


def test_rps_averages_over_samples():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert rps(probs, np.array([0, 2])) == pytest.approx(0.0)


def test_rps_rejects_non_three_class():
    with pytest.raises(ValueError):
        rps(np.array([[0.5, 0.5]]), np.array([0]))


# ---- log-loss --------------------------------------------------------------
def test_log_loss_perfect_prediction_near_zero():
    assert log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == pytest.approx(
        0.0, abs=1e-9
    )


def test_log_loss_known_value():
    assert log_loss(np.array([[0.5, 0.25, 0.25]]), np.array([0])) == pytest.approx(
        -np.log(0.5)
    )


# ---- Brier -----------------------------------------------------------------
def test_brier_perfect_prediction_is_zero():
    assert brier(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == pytest.approx(0.0)


def test_brier_known_value():
    # (.8-1)^2 + .1^2 + .1^2 = .04+.01+.01 = 0.06
    assert brier(np.array([[0.8, 0.1, 0.1]]), np.array([0])) == pytest.approx(0.06)
