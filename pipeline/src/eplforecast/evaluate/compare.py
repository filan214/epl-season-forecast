"""Leak-free nested five-way comparison (guide P3): baseline, bayes, xgb, blend,
market. Tune the blend weight on the 2nd-to-last season, then report the table on
the last season the weight never saw — so "blend beats components" is honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT_HALF_LIFE_DAYS
from ..features.build import FEATURE_COLUMNS, build_features
from ..models.baseline import fit_baseline
from ..models.bayes_goals import fit_bayes, predict_outcome_probs
from ..models.blend import blend, select_blend_weight
from ..models.xgb_outcome import fit_xgb, xgb_probs
from .metrics import brier, log_loss, rps
from .validation import _ordered_proba, market_probs

MODEL_ORDER = ("baseline", "bayes", "xgb", "blend", "market")


@dataclass(frozen=True)
class ComparisonResult:
    w_star: float
    tune_season: str
    test_season: str
    table: pd.DataFrame  # columns: model, rps, log_loss, brier, n


def _season_order(matches: pd.DataFrame) -> list[str]:
    return matches.sort_values("date", kind="stable")["season"].drop_duplicates().tolist()


def _component_probs(matches, feats, target, half_life_days, fit_bayes_fn, fit_xgb_fn,
                     sample_kwargs):
    """baseline/bayes/xgb/market probs + outcomes for `target`, trained strictly on
    earlier seasons. All arrays are aligned on feats[target]'s row order."""
    order = _season_order(matches)
    prior = order[: order.index(target)]
    if not prior:
        raise ValueError(f"season {target!r} has no earlier season to train on")

    feats_train = feats[feats["season"].isin(prior)]
    feats_test = feats[feats["season"] == target]
    y = feats_test["outcome"].to_numpy()
    x_train, y_train = feats_train[FEATURE_COLUMNS], feats_train["outcome"].to_numpy()

    base = fit_baseline(x_train, y_train)
    p_base = _ordered_proba(base, feats_test[FEATURE_COLUMNS])

    xgb = fit_xgb_fn(x_train, y_train)
    p_xgb = xgb_probs(xgb, feats_test[FEATURE_COLUMNS])

    train_m = matches.loc[feats_train.index]
    teams = sorted(set(train_m["home_team"]) | set(train_m["away_team"])
                   | set(feats_test["home_team"]) | set(feats_test["away_team"]))
    cutoff = feats_test["date"].min()
    idata = fit_bayes_fn(train_m, teams, cutoff, half_life_days=half_life_days,
                         **(sample_kwargs or {}))
    p_bayes = predict_outcome_probs(idata, feats_test, teams)

    p_mkt = market_probs(feats_test)
    return {"baseline": p_base, "bayes": p_bayes, "xgb": p_xgb, "market": p_mkt}, y


def _scored_row(name: str, probs: np.ndarray, outcomes: np.ndarray) -> dict:
    valid = ~np.isnan(probs).any(axis=1)
    p, y = probs[valid], outcomes[valid]
    return {"model": name, "rps": rps(p, y), "log_loss": log_loss(p, y),
            "brier": brier(p, y), "n": int(valid.sum())}


def run_comparison(matches, half_life_days=DEFAULT_HALF_LIFE_DAYS, window=5,
                   fit_bayes_fn=None, fit_xgb_fn=None,
                   sample_kwargs=None) -> ComparisonResult:
    """Tune w on the 2nd-to-last season, score all five models on the last."""
    fit_bayes_fn = fit_bayes_fn or fit_bayes
    fit_xgb_fn = fit_xgb_fn or fit_xgb
    order = _season_order(matches)
    if len(order) < 3:
        raise ValueError(f"need >=3 seasons (train, tune, test); got {len(order)}")
    tune_season, test_season = order[-2], order[-1]

    cutoff = matches["date"].max() + pd.Timedelta(days=1)
    feats = build_features(matches, cutoff, window=window)

    tune_probs, tune_y = _component_probs(matches, feats, tune_season, half_life_days,
                                          fit_bayes_fn, fit_xgb_fn, sample_kwargs)
    w_star = select_blend_weight(tune_probs["bayes"], tune_probs["xgb"], tune_y)

    test_probs, test_y = _component_probs(matches, feats, test_season, half_life_days,
                                          fit_bayes_fn, fit_xgb_fn, sample_kwargs)
    test_probs["blend"] = blend(test_probs["bayes"], test_probs["xgb"], w_star)

    table = pd.DataFrame([_scored_row(name, test_probs[name], test_y) for name in MODEL_ORDER])
    return ComparisonResult(w_star=float(w_star), tune_season=tune_season,
                            test_season=test_season, table=table)
