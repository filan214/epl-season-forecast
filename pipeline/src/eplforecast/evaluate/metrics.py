"""RPS, log-loss, and Brier for three-way match outcomes.

Probabilities are (n, 3) with columns [P(home), P(draw), P(away)]; outcomes are
integer class indices 0=home, 1=draw, 2=away. RPS is the primary metric because
it is ordinal-aware — predicting a draw when the home side wins is less wrong
than predicting an away win (PRD §10).
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 3
OUTCOME_ORDER = ("H", "D", "A")  # class index 0, 1, 2


def _as_2d_probs(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ValueError(f"probs must have shape (n, {N_CLASSES}); got {p.shape}")
    return p


def _as_outcomes(outcomes, n: int) -> np.ndarray:
    o = np.atleast_1d(np.asarray(outcomes)).astype(int)
    if o.shape[0] != n:
        raise ValueError(f"outcomes length {o.shape[0]} != number of prob rows {n}")
    if o.min() < 0 or o.max() >= N_CLASSES:
        raise ValueError("outcomes must be integer class indices in {0, 1, 2}")
    return o


def _one_hot(outcomes: np.ndarray) -> np.ndarray:
    oh = np.zeros((outcomes.shape[0], N_CLASSES))
    oh[np.arange(outcomes.shape[0]), outcomes] = 1.0
    return oh


def rps(probs, outcomes) -> float:
    """Mean ranked probability score. 0 is perfect; lower is better."""
    p = _as_2d_probs(probs)
    o = _one_hot(_as_outcomes(outcomes, p.shape[0]))
    # Cumulative over the ordered categories; the last cumulative column is
    # always 1 for both, so it is dropped.
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(o, axis=1)[:, :-1]
    per_sample = np.sum((cum_p - cum_o) ** 2, axis=1) / (N_CLASSES - 1)
    return float(per_sample.mean())


def log_loss(probs, outcomes, eps: float = 1e-15) -> float:
    """Mean multiclass cross-entropy on the realised outcome."""
    p = _as_2d_probs(probs)
    idx = _as_outcomes(outcomes, p.shape[0])
    picked = np.clip(p[np.arange(p.shape[0]), idx], eps, 1.0 - eps)
    return float(-np.log(picked).mean())


def brier(probs, outcomes) -> float:
    """Mean multiclass Brier score: mean over samples of sum_k (p_k - o_k)^2."""
    p = _as_2d_probs(probs)
    o = _one_hot(_as_outcomes(outcomes, p.shape[0]))
    return float(np.sum((p - o) ** 2, axis=1).mean())
