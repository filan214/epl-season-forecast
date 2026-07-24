"""Dixon-Coles score matrix and W/D/A read-off, per PYMC_MODEL_SPEC.md §7.

Over/under 2.5, BTTS, and clean sheets all read off the same matrix — no
extra model. Season simulation (spec §8) samples scorelines from it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Joint scoreline pmf M[i, j] = P(home=i, away=j), DC-corrected, normalised."""
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    m = np.outer(ph, pa)
    m[0, 0] *= 1 - lam_h * lam_a * rho
    m[0, 1] *= 1 + lam_h * rho
    m[1, 0] *= 1 + lam_a * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    """(p_home, p_draw, p_away) from a score matrix."""
    return (
        float(np.tril(m, -1).sum()),  # home win (i > j)
        float(np.trace(m)),           # draw
        float(np.triu(m, 1).sum()),   # away win
    )
