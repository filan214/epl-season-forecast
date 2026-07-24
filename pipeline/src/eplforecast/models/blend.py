"""Linear probability blend of the Bayesian and XGBoost outcome models.

Guide §4.3 / PYMC_MODEL_SPEC.md §10: p_final = w*p_bayes + (1-w)*p_xgb,
renormalised. w is tuned by out-of-sample RPS; expect it near 0.5.
"""

from __future__ import annotations

import numpy as np

from ..evaluate.metrics import rps

BLEND_WEIGHT_GRID = np.linspace(0.0, 1.0, 21)


def blend(p_bayes, p_xgb, w) -> np.ndarray:
    """w*p_bayes + (1-w)*p_xgb, rows renormalised to sum to 1."""
    p_bayes = np.asarray(p_bayes, dtype=float)
    p_xgb = np.asarray(p_xgb, dtype=float)
    if p_bayes.shape != p_xgb.shape:
        raise ValueError(f"shape mismatch: {p_bayes.shape} vs {p_xgb.shape}")
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1]; got {w}")
    mixed = w * p_bayes + (1.0 - w) * p_xgb
    return mixed / mixed.sum(axis=1, keepdims=True)


def select_blend_weight(p_bayes, p_xgb, outcomes, grid=BLEND_WEIGHT_GRID) -> float:
    """Grid-search w on pooled RPS; ties resolve to the value nearest 0.5 (§10)."""
    scores = np.array([rps(blend(p_bayes, p_xgb, float(w)), outcomes) for w in grid])
    best = scores.min()
    near = [float(w) for w, s in zip(grid, scores, strict=True) if s <= best + 1e-12]
    return min(near, key=lambda w: abs(w - 0.5))
