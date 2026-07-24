# Phase 3 — XGBoost + Calibration + Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isotonic-calibrated XGBoost outcome classifier, a tuned linear blend with the Phase 2 Bayesian model, and a leak-free five-way RPS comparison (baseline, bayes, xgb, blend, market).

**Architecture:** Two model modules (`xgb_outcome.py`, `blend.py`) emitting the project's standard `(n,3)` `[home,draw,away]` probabilities, and one evaluation driver (`compare.py`) that runs a nested protocol: tune the blend weight on the 2nd-to-last season, then report the five-way table on the last season the weight never saw. Fit functions are injectable so orchestration tests run without MCMC.

**Tech Stack:** Python 3.11+, xgboost 3.x (`XGBClassifier`), scikit-learn 1.9 (`CalibratedClassifierCV`), numpy, pandas. Reuses Phase 1 features/metrics/validation and Phase 2 `fit_bayes`/`predict_outcome_probs`.

## Global Constraints

- Python `>=3.11` (running 3.13); no ORM on the Python side.
- Outcome encoding is fixed: `0=home, 1=draw, 2=away`. All probability arrays are `(n, 3)` ordered `[P(home), P(draw), P(away)]`.
- Reuse `eplforecast.features.build.FEATURE_COLUMNS`; do not invent features.
- TDD: write the failing test first, watch it fail, minimal code, watch it pass, commit. Pristine output (no warnings).
- ruff config: line-length 100, `select = E,F,I,UP,B,SIM`.
- pytest resolves imports via `pythonpath = ["src"]`; `tests/conftest.py` sets `PYTENSOR_FLAGS=cxx=` when g++ is absent — do not remove.
- Blend contract (PYMC_MODEL_SPEC.md §10): `p_final = w·p_bayes + (1−w)·p_xgb`, renormalised; calibrate XGBoost isotonic **before** blending; expect `w ≈ 0.5`.
- Selected half-life default is `config.DEFAULT_HALF_LIFE_DAYS = 365.0`.

---

### Task 1: `models/xgb_outcome.py` — calibrated XGBoost classifier

**Files:**
- Create: `pipeline/src/eplforecast/models/xgb_outcome.py`
- Test: `pipeline/tests/test_xgb_outcome.py`

**Interfaces:**
- Consumes: `eplforecast.evaluate.metrics.N_CLASSES` (== 3).
- Produces:
  - `fit_xgb(X_train, y_train) -> sklearn.calibration.CalibratedClassifierCV`
  - `xgb_probs(model, X) -> np.ndarray` shape `(n, 3)`, ordered `[home,draw,away]`.

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_xgb_outcome.py
"""Tests for models/xgb_outcome.py — isotonic-calibrated XGBoost outcome model."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from eplforecast.features.build import FEATURE_COLUMNS
from eplforecast.models.xgb_outcome import fit_xgb, xgb_probs


def _signal_frame(n=600, seed=0):
    """elo_diff drives the outcome so the classifier has real signal to learn."""
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 200, n)
    X = pd.DataFrame({
        "elo_diff": elo,
        "home_gf_roll": rng.normal(1.5, 0.5, n), "home_ga_roll": rng.normal(1.1, 0.4, n),
        "home_pts_roll": rng.normal(1.4, 0.6, n), "away_gf_roll": rng.normal(1.3, 0.5, n),
        "away_ga_roll": rng.normal(1.2, 0.4, n), "away_pts_roll": rng.normal(1.3, 0.6, n),
    })[FEATURE_COLUMNS]
    p_home = 1.0 / (1.0 + np.exp(-elo / 150.0))
    u = rng.random(n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.2, 1, 2))
    return X, y


def test_fit_xgb_returns_isotonic_calibrated_classifier():
    model = fit_xgb(*_signal_frame())
    assert isinstance(model, CalibratedClassifierCV)
    assert model.get_params()["method"] == "isotonic"


def test_xgb_probs_are_ordered_and_normalised():
    X, y = _signal_frame()
    p = xgb_probs(fit_xgb(X, y), X)
    assert p.shape == (len(X), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_xgb_learns_the_home_signal():
    X, y = _signal_frame()
    model = fit_xgb(X, y)
    row = {c: [X[c].mean()] for c in FEATURE_COLUMNS}
    strong_home = pd.DataFrame(row); strong_home["elo_diff"] = 500.0
    strong_away = pd.DataFrame(row); strong_away["elo_diff"] = -500.0
    p_home = xgb_probs(model, strong_home[FEATURE_COLUMNS])[0]
    p_away = xgb_probs(model, strong_away[FEATURE_COLUMNS])[0]
    assert p_home[0] > p_home[2]   # home favoured -> P(home) > P(away)
    assert p_away[2] > p_away[0]   # away favoured -> P(away) > P(home)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pipeline && python -m pytest tests/test_xgb_outcome.py -q`
Expected: FAIL — `ImportError: cannot import name 'fit_xgb'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/src/eplforecast/models/xgb_outcome.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pipeline && python -m pytest tests/test_xgb_outcome.py -q`
Expected: PASS (3 passed). Then `python -m ruff check src/eplforecast/models/xgb_outcome.py tests/test_xgb_outcome.py` → "All checks passed!"

- [ ] **Step 5: Commit**

```bash
git add pipeline/src/eplforecast/models/xgb_outcome.py pipeline/tests/test_xgb_outcome.py
git commit -m "feat: isotonic-calibrated XGBoost outcome model (Phase 3)"
```

---

### Task 2: `models/blend.py` — linear blend + weight selection

**Files:**
- Create: `pipeline/src/eplforecast/models/blend.py`
- Test: `pipeline/tests/test_blend.py`

**Interfaces:**
- Consumes: `eplforecast.evaluate.metrics.rps`.
- Produces:
  - `blend(p_bayes, p_xgb, w) -> np.ndarray` `(n,3)`, renormalised.
  - `BLEND_WEIGHT_GRID` (np.ndarray, `linspace(0,1,21)`).
  - `select_blend_weight(p_bayes, p_xgb, outcomes, grid=BLEND_WEIGHT_GRID) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_blend.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pipeline && python -m pytest tests/test_blend.py -q`
Expected: FAIL — `ImportError: cannot import name 'blend'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/src/eplforecast/models/blend.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pipeline && python -m pytest tests/test_blend.py -q`
Expected: PASS (5 passed). Then ruff on both files → "All checks passed!".

- [ ] **Step 5: Commit**

```bash
git add pipeline/src/eplforecast/models/blend.py pipeline/tests/test_blend.py
git commit -m "feat: linear Bayes/XGB blend + RPS weight selection (Phase 3)"
```

---

### Task 3: `evaluate/compare.py` — leak-free five-way comparison driver

**Files:**
- Create: `pipeline/src/eplforecast/evaluate/compare.py`
- Test: `pipeline/tests/test_compare.py`

**Interfaces:**
- Consumes: `features.build.{FEATURE_COLUMNS, build_features}`, `models.baseline.fit_baseline`, `models.bayes_goals.{fit_bayes, predict_outcome_probs}`, `models.xgb_outcome.{fit_xgb, xgb_probs}`, `models.blend.{blend, select_blend_weight}`, `evaluate.metrics.{rps, log_loss, brier}`, `evaluate.validation.{_ordered_proba, market_probs}`, `config.DEFAULT_HALF_LIFE_DAYS`.
- Produces:
  - `MODEL_ORDER = ("baseline", "bayes", "xgb", "blend", "market")`
  - `ComparisonResult` dataclass: `w_star: float`, `tune_season: str`, `test_season: str`, `table: pd.DataFrame` (columns `model, rps, log_loss, brier, n`).
  - `run_comparison(matches, half_life_days=DEFAULT_HALF_LIFE_DAYS, window=5, fit_bayes_fn=None, fit_xgb_fn=None, sample_kwargs=None) -> ComparisonResult`.

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_compare.py
"""Tests for evaluate/compare.py — leak-free nested five-way comparison.

Bayes is faked (no MCMC): fit_bayes is injected and predict_outcome_probs is
monkeypatched. XGBoost/baseline/market run for real on a tiny synthetic league.
"""

import numpy as np
import pandas as pd

from eplforecast.evaluate import compare as cmp
from eplforecast.evaluate.compare import MODEL_ORDER, run_comparison


def _matches(seasons=("2019-20", "2020-21", "2021-22", "2022-23"), n_teams=6, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = {t: rng.normal(0, 0.3) for t in teams}
    rows = []
    base = pd.Timestamp("2019-08-01")
    for s_i, season in enumerate(seasons):
        d = base + pd.DateOffset(years=s_i)
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                h, a = teams[i], teams[j]
                lam_h = np.exp(0.3 + strength[h] - strength[a])
                lam_a = np.exp(0.1 + strength[a] - strength[h])
                rows.append({
                    "season": season, "date": d, "home_team": h, "away_team": a,
                    "home_goals": int(rng.poisson(lam_h)),
                    "away_goals": int(rng.poisson(lam_a)),
                    "odds_home": 2.0, "odds_draw": 3.5, "odds_away": 3.5,
                })
                d += pd.Timedelta(hours=6)
    return pd.DataFrame(rows)


def _uniform_predict(idata, fixtures, teams, **kwargs):
    return np.full((len(fixtures), 3), 1 / 3)


def test_run_comparison_builds_the_five_way_table(monkeypatch):
    monkeypatch.setattr(cmp, "predict_outcome_probs", _uniform_predict)
    res = run_comparison(_matches(), fit_bayes_fn=lambda *a, **k: "IDATA")

    assert list(res.table["model"]) == list(MODEL_ORDER)
    assert res.tune_season == "2021-22" and res.test_season == "2022-23"
    assert 0.0 <= res.w_star <= 1.0
    assert (res.table["n"] > 0).all()
    assert res.table["rps"].notna().all()


def test_weight_tuning_never_sees_the_test_season(monkeypatch):
    seen_train_seasons = []

    def spy_fit_bayes(df, teams, cutoff, half_life_days=None, **kwargs):
        seen_train_seasons.append(set(df["season"].unique()))
        return "IDATA"

    captured = {}
    real_select = cmp.select_blend_weight

    def spy_select(p_bayes, p_xgb, outcomes, **kwargs):
        captured["n"] = len(outcomes)
        return real_select(p_bayes, p_xgb, outcomes, **kwargs)

    monkeypatch.setattr(cmp, "predict_outcome_probs", _uniform_predict)
    monkeypatch.setattr(cmp, "select_blend_weight", spy_select)
    run_comparison(_matches(), fit_bayes_fn=spy_fit_bayes)

    # First bayes fit is the tuning fit: its training seasons exclude BOTH the
    # tune season (2021-22) and the test season (2022-23).
    assert "2021-22" not in seen_train_seasons[0]
    assert "2022-23" not in seen_train_seasons[0]
    # w was tuned on exactly the tune season (6 teams double round robin = 30).
    assert captured["n"] == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pipeline && python -m pytest tests/test_compare.py -q`
Expected: FAIL — `ImportError: cannot import name 'MODEL_ORDER'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/src/eplforecast/evaluate/compare.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pipeline && python -m pytest tests/test_compare.py -q`
Expected: PASS (2 passed). Then ruff on both files → "All checks passed!".

- [ ] **Step 5: Commit**

```bash
git add pipeline/src/eplforecast/evaluate/compare.py pipeline/tests/test_compare.py
git commit -m "feat: leak-free five-way model comparison driver (Phase 3)"
```

---

### Task 4: `scripts/phase3_compare.py` — run the real five-way table

**Files:**
- Create: `pipeline/scripts/phase3_compare.py`

**Interfaces:**
- Consumes: `ingest.football_data_csv.load_historical`, `evaluate.compare.run_comparison`, `config.DEFAULT_HALF_LIFE_DAYS`.
- Produces: printed five-way RPS table + `w*`; no importable API (a runner, mirrors `scripts/phase1_demo.py`).

- [ ] **Step 1: Write the runner**

```python
# pipeline/scripts/phase3_compare.py
"""Phase 3: the real five-way RPS table on football-data.co.uk seasons.

Tunes the blend weight on the 2nd-to-last season and reports baseline / bayes /
xgb / blend / market on the last (unseen-by-w) season. Bayes is fit only twice
here, so a decent sampling budget is affordable.

    python scripts/phase3_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eplforecast.config import DEFAULT_HALF_LIFE_DAYS  # noqa: E402
from eplforecast.evaluate.compare import run_comparison  # noqa: E402
from eplforecast.ingest.football_data_csv import load_historical  # noqa: E402

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SAMPLE = dict(draws=1500, tune=1500, chains=4, cores=1,
              target_accept=0.9, random_seed=42, progressbar=False)


def main() -> None:
    matches = load_historical(SEASONS)
    print(f"{len(matches)} matches across {matches['season'].nunique()} seasons; "
          f"half-life={DEFAULT_HALF_LIFE_DAYS:g}d\n")
    res = run_comparison(matches, half_life_days=DEFAULT_HALF_LIFE_DAYS, sample_kwargs=SAMPLE)
    print(f"tune season: {res.tune_season}   test season: {res.test_season}   "
          f"w* (bayes weight) = {res.w_star:.2f}\n")
    show = res.table.copy()
    for c in ("rps", "log_loss", "brier"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    ranked = res.table.sort_values("rps")
    best = ranked.iloc[0]["model"]
    blend_rps = float(res.table.set_index("model").loc["blend", "rps"])
    bayes_rps = float(res.table.set_index("model").loc["bayes", "rps"])
    xgb_rps = float(res.table.set_index("model").loc["xgb", "rps"])
    beats_both = blend_rps < bayes_rps and blend_rps < xgb_rps
    print(f"\nlowest RPS: {best}. blend beats both components out-of-sample: {beats_both} "
          f"(blend {blend_rps:.4f} vs bayes {bayes_rps:.4f}, xgb {xgb_rps:.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the whole suite + lint (regression gate)**

Run: `cd pipeline && python -m pytest -q && python -m ruff check src scripts tests`
Expected: all tests pass (84 prior + 10 new = 94), ruff "All checks passed!".

- [ ] **Step 3: Run the real comparison (background — ~4–6 min, 2 Bayesian fits)**

Run: `cd pipeline && PYTENSOR_FLAGS="cxx=" python scripts/phase3_compare.py`
Expected: a five-row table with finite RPS for baseline/bayes/xgb/blend/market, `w*` printed near 0.5, and a line stating whether the blend beat both components out-of-sample. Record the numbers (do not fake them — report whatever appears).

- [ ] **Step 4: Commit**

```bash
git add pipeline/scripts/phase3_compare.py
git commit -m "feat: Phase 3 five-way comparison runner"
```

---

## Notes for the implementer

- Do not weaken TLS or change ingest; the DoH fallback in `football_data_csv.py` already handles the ISP block, so `load_historical` works locally.
- The blend "winning" is not guaranteed — the done-when is that the honest out-of-sample table exists and is reported, whichever model leads.
- After Task 4, update the project memory note (Phase 3 result: `w*`, the five RPS figures, whether blend beat both) and set the Phase 2/3 status accordingly.
