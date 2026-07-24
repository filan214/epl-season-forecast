"""Tests for models/blend.py — linear probability blend + weight selection."""

import numpy as np
import pytest

from eplforecast.models.blend import blend, select_blend_weight


def test_blend_w1_is_bayes_and_w0_is_xgb():
    pb = np.array([[0.6, 0.3, 0.1]])
    px = np.array([[0.2, 0.3, 0.5]])
    np.testing.assert_allclose(blend(pb, px, 1.0), pb)
    np.testing.assert_allclose(blend(pb, px, 0.0), px)


def test_blend_renormalises_rows_to_one():
    pb = np.array([[0.6, 0.3, 0.1]])
    px = np.array([[0.2, 0.3, 0.5]])
    out = blend(pb, px, 0.5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_blend_rejects_out_of_range_weight():
    p = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError):
        blend(p, p, 1.5)


def test_select_blend_weight_prefers_the_oracle_component():
    rng = np.random.default_rng(0)
    n = 400
    outcomes = rng.integers(0, 3, n)
    onehot = np.eye(3)[outcomes]
    p_oracle = 0.85 * onehot + 0.05      # near-perfect
    p_noise = np.full((n, 3), 1 / 3)
    # xgb is the oracle -> bayes weight w should collapse toward 0
    assert select_blend_weight(p_bayes=p_noise, p_xgb=p_oracle, outcomes=outcomes) < 0.2
    # mirror -> w toward 1
    assert select_blend_weight(p_bayes=p_oracle, p_xgb=p_noise, outcomes=outcomes) > 0.8


def test_select_blend_weight_breaks_ties_towards_half():
    p = np.tile([0.4, 0.3, 0.3], (50, 1))
    outcomes = np.zeros(50, dtype=int)
    assert select_blend_weight(p, p, outcomes) == pytest.approx(0.5)
