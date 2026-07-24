"""Multinomial logistic baseline.

The honesty check of the project: on ~3,800 rows a well-tuned logistic model is
often competitive, and if boosting can't beat it that is a finding to publish
(PRD §9). Median-imputes (early-season rolling features are NaN), standardises,
then fits multinomial logistic regression. predict_proba columns align to the
sorted classes [0=home, 1=draw, 2=away] when all three are present in y_train.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_baseline(X_train, y_train) -> Pipeline:
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000)),
        ]
    )
    # Fit on X as given (do not coerce to a nameless array) so that predicting on
    # the same type — DataFrame or ndarray — never triggers a feature-name warning.
    model.fit(X_train, y_train)
    return model
