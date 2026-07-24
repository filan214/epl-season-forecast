"""Tests for simulate/scorelines.py — DC score matrix and W/D/A read-off.

Contracts from spec §7 / guide §4.4: score_matrix(lam_h, lam_a, rho,
max_goals=10) is a normalised joint pmf over scorelines with the four
Dixon-Coles cells corrected; outcome_probs(M) reads home/draw/away off the
lower triangle / trace / upper triangle.
"""

import pytest

from eplforecast.simulate.scorelines import outcome_probs, score_matrix


def test_score_matrix_is_a_normalised_pmf():
    m = score_matrix(1.5, 1.2, -0.13)
    assert m.shape == (11, 11)
    assert (m >= 0).all()
    assert m.sum() == pytest.approx(1.0)


def test_negative_rho_boosts_the_draw_cells():
    # The whole point of the correction: independent Poissons underpredict
    # 0-0 and 1-1; negative rho inflates them and deflates 0-1 / 1-0.
    plain = score_matrix(1.5, 1.2, 0.0)
    dc = score_matrix(1.5, 1.2, -0.13)
    assert dc[0, 0] > plain[0, 0]
    assert dc[1, 1] > plain[1, 1]
    assert dc[0, 1] < plain[0, 1]
    assert dc[1, 0] < plain[1, 0]


def test_outcome_probs_partition_the_matrix():
    m = score_matrix(1.8, 1.1, -0.1)
    p_home, p_draw, p_away = outcome_probs(m)
    assert p_home + p_draw + p_away == pytest.approx(1.0)
    # Home scores at a much higher rate here, so home win must dominate.
    assert p_home > p_away
    # Cross-check the read-off against explicit cell sums.
    assert p_draw == pytest.approx(sum(m[i, i] for i in range(11)))
    assert p_home == pytest.approx(sum(m[i, j] for i in range(11) for j in range(11) if i > j))
