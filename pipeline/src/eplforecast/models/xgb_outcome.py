"""XGBoost outcome classifier with isotonic-calibrated probabilities.

Guide §4.3 / PYMC_MODEL_SPEC.md §10: calibrate the probabilities (isotonic)
BEFORE blending, never after. Fixed, conservative hyperparameters — only the
blend weight is tuned downstream, not the booster.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ..evaluate.metrics import N_CLASSES

# Conservative, fixed HPs (~thousands of rows). No HP search (YAGNI).
_XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=1,
)


def fit_xgb(X_train, y_train) -> CalibratedClassifierCV:
    """Fit XGBoost wrapped in 3-fold isotonic calibration (guide §4.3)."""
    from xgboost import XGBClassifier

    base = XGBClassifier(**_XGB_PARAMS)
    model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    model.fit(X_train, np.asarray(y_train))
    return model


def xgb_probs(model, X) -> np.ndarray:
    """(n, 3) calibrated probabilities ordered [home, draw, away]."""
    proba = model.predict_proba(X)
    out = np.zeros((proba.shape[0], N_CLASSES))
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, col]
    return out
