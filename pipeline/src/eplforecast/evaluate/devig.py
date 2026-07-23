"""Bookmaker odds -> normalised outcome probabilities (basic de-vigging).

Multiplicative normalisation: implied probabilities (1/odds) divided by the
overround. Documented bias: this slightly favours the favourite versus Shin or
power methods (PRD §11); those are deferred to v2 (PRD §7 backlog).

Works on scalars (returns a tuple of floats) or equal-length numpy arrays
(returns a tuple of arrays), so the validation harness can de-vig a whole column
at once.
"""

from __future__ import annotations

import numpy as np


def devig(odds_home, odds_draw, odds_away):
    oh = np.asarray(odds_home, dtype=float)
    od = np.asarray(odds_draw, dtype=float)
    oa = np.asarray(odds_away, dtype=float)

    # Only flag genuinely invalid (<= 1.0) odds; NaN (missing quote) is allowed
    # to propagate to NaN probabilities for the harness to filter out.
    stacked = np.stack(np.broadcast_arrays(oh, od, oa))
    if np.any(stacked <= 1.0):
        raise ValueError("decimal odds must be > 1.0")

    imp_h, imp_d, imp_a = 1.0 / oh, 1.0 / od, 1.0 / oa
    overround = imp_h + imp_d + imp_a
    p_home, p_draw, p_away = imp_h / overround, imp_d / overround, imp_a / overround

    if oh.ndim == 0:
        return (float(p_home), float(p_draw), float(p_away))
    return (p_home, p_draw, p_away)
