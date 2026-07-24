# Phase 3 — XGBoost, isotonic calibration, and the Bayes×XGB blend

**Status:** approved 2026-07-24
**Scope:** IMPLEMENTATION_GUIDE.md Phase 3 / P3; blend contract from PYMC_MODEL_SPEC.md §10.
**Done when:** all five RPS figures (baseline, bayes, xgb, blend, market) sit in one
table, and whether the blend beats its components is reported **out of sample**.

## Goal

Add a gradient-boosted outcome classifier with isotonic-calibrated probabilities,
combine it with the Phase 2 Bayesian goal model via a tuned linear blend, and
produce a single honest five-way RPS comparison. This is the calibration/UQ
payoff of the project, not an accuracy grab — expected gains over a good
frequentist model are modest and will be reported as such.

## Non-goals (YAGNI)

- No XGBoost hyperparameter search. Fixed, conservative HPs; only the **blend
  weight** is tuned. (HP tuning would need another nested split for little gain.)
- No new features. Reuse Phase 1 `FEATURE_COLUMNS`
  (`elo_diff`, `{home,away}_{gf,ga,pts}_roll`).
- No dashboard/DB work (Phase 4+).
- Season simulation still consumes the **Bayesian** scoreline matrix, not the
  blend (§10) — untouched here.

## Components

### 1. `models/xgb_outcome.py`
- `fit_xgb(X_train, y_train) -> CalibratedClassifierCV` (guide §4.3).
  - Base: `xgboost.XGBClassifier` — multiclass (`objective="multi:softprob"`,
    `num_class=3`), conservative HPs (`max_depth=3`, `n_estimators=200`,
    `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`,
    `reg_lambda=1.0`, fixed `random_state`).
  - Wrapped in `CalibratedClassifierCV(base, method="isotonic", cv=3)` (3-fold:
    trains the booster on 2 folds, fits isotonic on the held fold, ensembles) so
    the returned probabilities are isotonic-calibrated **before** any blend (§10).
    Isotonic is applied one-vs-rest per class by sklearn and renormalized.
  - XGBoost handles NaN natively → no imputer (unlike the logistic baseline);
    early-season rolling-feature NaNs are fine.
- `xgb_probs(model, X) -> np.ndarray` — `(n, 3)` ordered `[home, draw, away]`
  (0=home, 1=draw, 2=away), reindexing `model.classes_` to fixed columns exactly
  as `evaluate/validation.py::_ordered_proba` does.

### 2. `models/blend.py`
- `blend(p_bayes, p_xgb, w) -> np.ndarray` — `w*p_bayes + (1-w)*p_xgb`, then
  renormalize rows to sum to 1 (§10). Validates matching shapes and `0<=w<=1`.
- `BLEND_WEIGHT_GRID` — e.g. `np.linspace(0, 1, 21)`.
- `select_blend_weight(p_bayes, p_xgb, outcomes, grid=BLEND_WEIGHT_GRID) -> float`
  — argmin pooled RPS over the grid; ties resolve to the value nearest 0.5
  (the §10 prior). Returns `w*`.

### 3. `evaluate/compare.py` (+ `scripts/phase3_compare.py`)
Nested, leak-free protocol over the tail of the season list:
- Pick `tune_season` = 2nd-to-last, `test_season` = last.
- **Tune:** fit bayes + xgb on all seasons strictly before `tune_season`; predict
  `tune_season`; `w* = select_blend_weight(p_bayes, p_xgb, outcomes_tune)`.
- **Test:** fit bayes + xgb on all seasons strictly before `test_season` (which
  now includes `tune_season`); predict `test_season`; assemble probabilities for
  **baseline, bayes, xgb, blend(w\*), market** and score each with `rps`
  (plus `log_loss`, `brier`).
- Return a `ComparisonResult`: `w_star`, `tune_season`, `test_season`, and a
  table (one row per model: name, rps, log_loss, brier, n).
- **Injectable fits:** `fit_bayes_fn` / `fit_xgb_fn` parameters (default to the
  real ones) so orchestration tests run with fast fakes — no MCMC/boosting.
- Bayes appears in exactly **two** fits (tune + test), so the runner can afford a
  decent sampling budget; the exact budget is a script-level constant, documented.

`scripts/phase3_compare.py` loads real seasons via `load_historical` (DoH-enabled
ingest) and prints the table; mirrors `scripts/phase1_demo.py`.

## Data flow

```
features (build_features, Phase 1) ─┬─► baseline (logistic)      ─► (n,3)
                                    └─► xgb (calibrated)          ─► (n,3) ─┐
matches ─► fit_bayes (Phase 2) ─► predict_outcome_probs          ─► (n,3) ─┼─► blend(w*) ─► (n,3)
closing odds ─► devig                                            ─► (n,3)  │
                                                                            │
all (n,3) [home,draw,away] ─► rps / log_loss / brier ─► five-way table ◄────┘
```

## Leakage guards

- `w*` is selected **only** on `tune_season`; the test-season table uses that
  fixed `w*`. `test_compare.py` asserts the weight-selection call never receives
  test-season rows.
- Each fit trains strictly on seasons before its prediction target (expanding
  window; same discipline as Phases 1–2). No random k-fold.
- XGBoost's internal calibration CV is confined to the training seasons.

## Testing (TDD)

- `test_xgb_outcome.py`: returns a `CalibratedClassifierCV`; `xgb_probs` shape
  `(n,3)` summing to 1 with correct column order; learns signal (home-favouring
  features → higher P(home)); calibrated probs differ from raw (calibration ran).
- `test_blend.py`: `w=1`→bayes, `w=0`→xgb, intermediate renormalized; shape/`w`
  validation; `select_blend_weight` recovers `w→0` when xgb is oracle and bayes
  noise, `w→1` in the mirror case; tie-break nearest 0.5.
- `test_compare.py`: orchestration with fake fits — table has exactly the five
  named rows; `w*` from the tune season is the one used at test; **leakage
  assertion** that tuning never sees `test_season`.

## Risks / honest notes

- The blend may **not** beat both components out of sample — that's an empirical
  result to report, not to engineer. The §10 expectation is `w≈0.5` and small
  gains; we state the real number.
- Single tune/test pair gives one out-of-sample estimate; a fuller rolling
  evaluation is possible later but is out of scope for the P3 done-when.
- README limitation note (calibration/UQ over accuracy) carries from
  PYMC_MODEL_SPEC.md §11.
