"""Expanding-window validation harness (by season only; never random k-fold).

For each held-out season it trains the baseline on all strictly-earlier seasons,
predicts the held-out season, and scores the baseline against the de-vigged
bookmaker closing odds on the same matches (PRD §9, IMPLEMENTATION_GUIDE.md §7).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..features.build import FEATURE_COLUMNS, build_features
from ..models.baseline import fit_baseline
from .devig import devig
from .metrics import N_CLASSES, brier, log_loss, rps

_ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")


def expanding_window_splits(
    df: pd.DataFrame, min_train_seasons: int = 3
) -> Iterator[tuple[pd.Index, pd.Index, str]]:
    """Yield (train_index, test_index, test_season), seasons ordered by date."""
    ordered_seasons = (
        df.sort_values("date", kind="stable")["season"].drop_duplicates().tolist()
    )
    for i in range(min_train_seasons, len(ordered_seasons)):
        test_season = ordered_seasons[i]
        train_mask = df["season"].isin(ordered_seasons[:i]).to_numpy()
        test_mask = (df["season"] == test_season).to_numpy()
        yield df.index[train_mask], df.index[test_mask], test_season


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """(n, 3) de-vigged [home, draw, away] probabilities from closing odds.

    Invalid or missing quotes (odds <= 1.0 or NaN) become NaN rows rather than
    crashing the run, so the caller can filter them out.
    """
    def _clean(col: str) -> np.ndarray:
        x = df[col].to_numpy(dtype=float)
        return np.where(x > 1.0, x, np.nan)

    p_home, p_draw, p_away = devig(
        _clean("odds_home"), _clean("odds_draw"), _clean("odds_away")
    )
    return np.column_stack([p_home, p_draw, p_away])


def _ordered_proba(model, X) -> np.ndarray:
    """predict_proba reordered to fixed columns [0=home, 1=draw, 2=away],
    robust to a training fold that happens to miss a class."""
    proba = model.predict_proba(X)
    out = np.zeros((proba.shape[0], N_CLASSES))
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, col]
    return out


def _metrics(probs: np.ndarray, outcomes: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_rps": rps(probs, outcomes),
        f"{prefix}_log_loss": log_loss(probs, outcomes),
        f"{prefix}_brier": brier(probs, outcomes),
    }


def run_validation(
    feats: pd.DataFrame,
    feature_columns: list[str] | None = None,
    min_train_seasons: int = 3,
) -> pd.DataFrame:
    """Run the expanding-window baseline-vs-market comparison on a feature frame.

    `feats` must carry: season, date, outcome, the feature columns, and
    (optionally) closing-odds columns for the market benchmark.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    has_odds = all(c in feats.columns for c in _ODDS_COLUMNS)
    records = []

    for train_idx, test_idx, season in expanding_window_splits(feats, min_train_seasons):
        train, test = feats.loc[train_idx], feats.loc[test_idx]
        model = fit_baseline(train[feature_columns], train["outcome"].to_numpy())

        y_test = test["outcome"].to_numpy()
        p_base = _ordered_proba(model, test[feature_columns])
        row = {"season": season, "n_test": len(test)}
        row.update(_metrics(p_base, y_test, "baseline"))

        if has_odds:
            p_mkt = market_probs(test)
            valid = ~np.isnan(p_mkt).any(axis=1)
            row["n_market"] = int(valid.sum())
            if valid.any():
                # Score both models on the same odds-valid subset for a fair gap.
                row.update(_metrics(p_mkt[valid], y_test[valid], "market"))
                row.update(_metrics(p_base[valid], y_test[valid], "baseline"))
            else:
                row.update(_metrics(np.empty((0, 3)), np.empty(0), "market"))
        else:
            row["n_market"] = 0
            row["market_rps"] = row["market_log_loss"] = row["market_brier"] = np.nan
        records.append(row)

    return pd.DataFrame.from_records(records)


def run_baseline_validation(
    matches: pd.DataFrame,
    min_train_seasons: int = 3,
    window: int = 5,
) -> pd.DataFrame:
    """Build leak-free features for all matches, then run the harness."""
    cutoff = matches["date"].max() + pd.Timedelta(days=1)
    feats = build_features(matches, cutoff, window=window)
    return run_validation(feats, min_train_seasons=min_train_seasons)


def print_validation_report(results: pd.DataFrame) -> None:
    """Print the per-season table plus match-weighted pooled RPS."""
    cols = ["season", "n_test", "n_market", "baseline_rps", "market_rps"]
    show = results[cols].copy()
    for c in ("baseline_rps", "market_rps"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    w = results["n_market"].to_numpy(dtype=float)
    if w.sum() > 0:
        base = np.average(results["baseline_rps"], weights=results["n_test"])
        mkt = np.average(results["market_rps"], weights=w)
        print(f"\nPooled baseline RPS: {base:.4f}   |   market RPS: {mkt:.4f}   "
              f"|   gap: {base - mkt:+.4f}")
