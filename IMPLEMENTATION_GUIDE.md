# IMPLEMENTATION_GUIDE — EPL Season Forecast (2026-27)

> Companion to `PRD.md`. This is the build spec: architecture, dependencies, repo layout, database schema, module contracts, automation, and a phase-by-phase plan with paste-ready Claude Code prompts. Read `PRD.md` first for scope and rationale. Model math lives in `PYMC_MODEL_SPEC.md`.

**Claude Code prompt prefix (use every session):**
> Read PRD.md and IMPLEMENTATION_GUIDE.md first, then complete the following task:

---

## 0. Architecture at a glance

- **Two halves, one repo.** A Python pipeline that trains and simulates offline, and a Next.js dashboard that only ever reads. They meet at Neon Postgres and nowhere else.
- **The pipeline is the product; the dashboard is a window onto it.** The dashboard never computes anything — no model calls, no simulation, no aggregation beyond what SQL does. If it can't be read straight from a table, the pipeline should have written it.
- **Nothing runs on a server.** GitHub Actions executes the weekly job; Vercel serves static-ish pages; Neon holds state. No always-on compute anywhere.
- **Data flow:** sources → parquet cache → feature build → {PyMC, XGBoost} → blend → season simulation → Neon → dashboard.
- **Two storage tiers by design.** Training data (30 seasons of CSVs, parquet) lives in the repo and on disk; it is never imported to Postgres. Serving data (predictions, snapshots, metrics) lives in Neon. The posterior trace lives in neither — it's a GitHub Actions artifact.

### Why the split matters

Putting the training corpus in Neon would consume the free tier for no benefit — the dashboard never queries it. Putting the trace in Neon would be worse: a multi-megabyte binary per gameweek would exceed 0.5 GB by midseason. Keep each artifact in the cheapest place that can serve its actual reader.

---

## 1. Stack & dependencies

### Pipeline (Python 3.11+)

**Core modelling**
- `pymc` (≥5), `arviz`, `pytensor`
- `xgboost`, `scikit-learn` (isotonic calibration, logistic baseline, `TimeSeriesSplit`)
- `numpy`, `pandas`, `scipy`, `pyarrow`

**Ingest**
- `httpx` or `requests`, `beautifulsoup4`
- `understatapi` (or ScraperFC) for xG

**Database**
- `psycopg[binary]` — plain SQL writes, no ORM on the Python side

**Tooling**
- `pytest`, `ruff`, `python-dotenv`

### Dashboard (Next.js 15, App Router)

- TypeScript, Tailwind, shadcn/ui, Recharts
- Drizzle ORM + `@neondatabase/serverless`
- Vercel hosting

**Do not put an ORM on the Python side.** Drizzle owns the schema and the migrations; Python writes through raw parameterised SQL against tables Drizzle created. One schema authority, two consumers.

### Environment note

Development is on **Windows + PowerShell**. No backslash line continuations in any documented command — single-line commands or PowerShell-native syntax only. Any shell snippet added to the README must be given in a PowerShell-safe form.

---

## 2. Repo structure

Monorepo. A reviewer should see the whole system in one place.

```
epl-season-forecast/
  PRD.md
  IMPLEMENTATION_GUIDE.md
  PYMC_MODEL_SPEC.md
  CLAUDE_CODE_PROMPTS.md
  README.md

  pipeline/
    pyproject.toml
    src/eplforecast/
      config.py                  # paths, constants, season config, decay default
      ingest/
        football_data_csv.py     # historical results + closing odds
        understat.py             # xG, throttled
        transfermarkt.py         # squad values (adapted from Transfer Globe)
        live_results.py          # football-data.org API
        teams.py                 # name reconciliation across all four sources
      features/
        elo.py                   # Elo ratings as a feature column
        rolling.py               # shift(1) rolling form windows
        build.py                 # assembles the model matrix
      models/
        baseline.py              # multinomial logistic
        bayes_goals.py           # PyMC hierarchical Poisson (see PYMC_MODEL_SPEC.md)
        xgb_outcome.py           # XGBoost + isotonic calibration
        blend.py                 # weight tuning + probability pooling
      simulate/
        scorelines.py            # Dixon-Coles score matrix, outcome probabilities
        season.py                # posterior predictive season simulation
      evaluate/
        metrics.py               # RPS, log-loss, Brier
        devig.py                 # bookmaker odds → probabilities
        calibration.py           # reliability bins, rolling metrics
      db/
        writer.py                # parameterised writes to Neon
        gate.py                  # convergence gating logic
      cli.py                     # typer/argparse entrypoints
    data/
      raw/                       # gitignored
      processed/                 # parquet, committed if small
    tests/

  web/
    app/
      page.tsx                   # table + probabilities (primary view)
      calibration/page.tsx       # accuracy record
      team/[slug]/page.tsx       # per-team strengths + points distribution
      methodology/page.tsx       # static, links PYMC_MODEL_SPEC.md
    src/
      db/schema.ts               # Drizzle — schema authority
      db/client.ts
      components/
    drizzle.config.ts

  .github/workflows/
    weekly_forecast.yml
    heartbeat.yml
```

---

## 3. Database schema (Neon Postgres, Drizzle-owned)

Sizing: roughly 380 matches, ~1,500 prediction rows, 38 gameweeks × 20 teams across two snapshot tables. **Under 10 MB with indexes** — about 2% of the 0.5 GB free-tier cap.

### `teams`
```
id (pk), name, short_name, slug,
fd_name          -- football-data.co.uk spelling
understat_name,
transfermarkt_id,
is_promoted (bool)   -- for the 2026-27 season
```
Name reconciliation across four sources is a real and underestimated cost. Build this table first, by hand if necessary, and make every ingest module resolve through it. A silent join failure here corrupts everything downstream.

### `matches`
```
id (pk), season, gameweek, kickoff_utc (timestamptz),
home_team_id (fk), away_team_id (fk),
home_goals (int, null), away_goals (int, null),
status ('scheduled' | 'finished'),
source_ref
```

### `model_runs`
```
id (pk), run_started_at, cutoff_date, git_sha,
half_life_days, n_draws, blend_weight,
r_hat_max, divergences, ess_bulk_min,
status ('ok' | 'gated' | 'failed'), notes
```
Every other table's rows point back here. If a gameweek's numbers look wrong, this is how you find out which commit and which cutoff produced them.

### `predictions`
```
id (pk), model_run_id (fk), match_id (fk),
model_variant ('bayes' | 'xgb' | 'blend' | 'market'),
p_home, p_draw, p_away,
lambda_home, lambda_away,
p_over_2_5, p_btts, p_home_cs, p_away_cs,
created_at (timestamptz default now())
```

**Add a database-level leakage guard.** A `CHECK` or trigger enforcing that `created_at < matches.kickoff_utc` makes the core integrity claim of the project unfalsifiable by accident. A `CHECK` can't reference another table, so implement it as a `BEFORE INSERT` trigger that raises on violation. This is cheap and it is the single most valuable constraint in the schema.

Store the `market` variant alongside the model variants — same table, same shape. It makes every comparison a `GROUP BY` instead of a join across systems.

### `season_sim_snapshots`
```
id (pk), model_run_id (fk), gameweek, team_id (fk),
p_title, p_top4, p_top6, p_relegation,
exp_points, points_p10, points_p90, exp_position,
with_param_uncertainty (bool)
```
That last column is what powers the side-by-side exhibit from PRD §6.3 — write both variants each run.

### `team_strengths`
```
id (pk), model_run_id (fk), team_id (fk),
att_mean, att_hdi_low, att_hdi_high,
def_mean, def_hdi_low, def_hdi_high
```

### `calibration_metrics`
```
id (pk), model_run_id (fk), through_gameweek, model_variant,
n_matches, rps, log_loss, brier
```
Market rows use `model_variant = 'market'`, so the comparison chart is one query.

### Indexes
`predictions(match_id, model_variant)`, `season_sim_snapshots(gameweek, team_id)`, `calibration_metrics(through_gameweek, model_variant)`, `matches(gameweek)`.

---

## 4. Module contracts

Keep these signatures stable; they are what the phases build against.

### 4.1 Ingest
```python
load_historical(seasons: list[str]) -> pd.DataFrame
    # date, home, away, home_goals, away_goals, + closing odds columns

fetch_understat(season: str) -> pd.DataFrame
    # match-level xG / xGA, resolved to team ids

fetch_squad_values(as_of: date) -> pd.DataFrame

fetch_live_results(since: date) -> pd.DataFrame
    # football-data.org; also returns scheduled fixtures
```
Every ingest function **caches to parquet before returning**. A run must be repeatable offline. An Understat outage should degrade the feature set, never fail the pipeline.

### 4.2 Features
```python
compute_elo(matches: pd.DataFrame, k: float, home_adv: float) -> pd.Series
build_features(matches: pd.DataFrame, cutoff: date) -> pd.DataFrame
```

**The leakage contract, stated once and enforced everywhere:** every rolling feature applies `.shift(1)` *before* `.rolling()`. Elo for match *m* uses only matches strictly before *m*. `build_features` takes an explicit `cutoff` and asserts no row at or after it. This is non-negotiable and carries directly from the F1 and WC predictors.

### 4.3 Models
```python
fit_bayes(df, teams, cutoff, half_life_days) -> az.InferenceData
fit_xgb(X_train, y_train) -> CalibratedClassifierCV
fit_baseline(X_train, y_train) -> LogisticRegression
blend(p_bayes, p_xgb, w) -> np.ndarray
```
See `PYMC_MODEL_SPEC.md` for priors, the Dixon-Coles τ correction, the time-decay parameterisation, and the full reference implementation. Do not re-derive it here.

### 4.4 Simulation
```python
score_matrix(lam_h, lam_a, rho, max_goals=10) -> np.ndarray
outcome_probs(M) -> tuple[float, float, float]
simulate_season(idata, fixtures, current_table, n_draws, with_uncertainty=True) -> pd.DataFrame
```

### 4.5 Evaluation
```python
rps(probs, outcomes) -> float
devig(odds_home, odds_draw, odds_away) -> tuple[float, float, float]
reliability_bins(probs, outcomes, n_bins=10) -> pd.DataFrame
```

### 4.6 Gating
```python
check_convergence(idata) -> GateResult   # ok | gated, with reasons
```
Thresholds: R̂ < 1.01 on all parameters, bulk ESS > 400, zero divergences. A gated run writes a `model_runs` row with `status='gated'` and the reason — then writes **nothing else**. The dashboard falls back to the last `status='ok'` run and labels it stale.

---

## 5. Automation

### `weekly_forecast.yml`

Runs Monday mornings during the season.

```
1. Checkout, install, restore parquet cache
2. Fetch new results (football-data.org) and xG (Understat)
3. Rebuild features with cutoff = now
4. Fit PyMC; run convergence gate
   → gated: write model_runs row, exit 0 with a warning, stop
5. Fit XGBoost, calibrate, blend
6. Simulate season, both with and without parameter uncertainty
7. Write predictions for all remaining fixtures — BEFORE any kickoff
8. Compute and write calibration metrics for the gameweek that just finished
9. Upload the InferenceData trace as an Actions artifact
```

**Timing is a correctness property, not a convenience.** Step 7 must complete before the earliest kickoff of the coming gameweek. Schedule with real margin — a Monday run for a Saturday gameweek — and let the DB trigger catch it if that ever slips.

Secrets: `NEON_DATABASE_URL`, `FOOTBALL_DATA_API_KEY`.

Cache the parquet directory between runs with `actions/cache`; re-downloading thirty seasons weekly is wasteful and rude.

### `heartbeat.yml`

Monthly `SELECT 1` against Neon. Neon's free tier suspends *compute* after five minutes of idle, not the project — so this is not currently required. It exists because free-tier policies change, the cost is zero, and this exact failure has already cost a project once.

---

## 6. Dashboard notes

Four routes, no more. The dashboard's job is to be legible in ninety seconds.

| Route | Content |
|---|---|
| `/` | Current table with title / top-4 / relegation probabilities, points range bars, "as of gameweek N" |
| `/calibration` | RPS / log-loss / Brier over time versus market, reliability diagram, run diagnostics |
| `/team/[slug]` | Attack and defence posteriors with HDI, final-position distribution |
| `/methodology` | Static prose, honest limitations, links to the spec |

**Stale-state handling is a v1 requirement.** If the latest `ok` run predates the current gameweek, the UI says so plainly with the `computed_at` date. Silently serving old numbers as current is the one failure mode that would actively damage the project's credibility.

Use `@neondatabase/serverless` and let connections close — a pooled connection held open defeats scale-to-zero and burns compute hours around the clock. It fails silently, so it's worth checking once explicitly.

---

## 7. Phase plan

Ordered against the GW1 gate from PRD §12. Phases 0–4 are mandatory before the season starts; 5–7 may land after.

### Phase 0 — Scaffold, schema, historical ingest (2 days)
**Build:** monorepo structure; Drizzle schema for all seven tables plus the leakage trigger; Neon project and connection; `teams` reconciliation table; `football_data_csv.py` ingesting 10+ seasons to parquet.
**Done when:** `SELECT * FROM teams` returns 20 correctly-mapped teams, and a parquet file holds a decade of matches with odds columns intact.

### Phase 1 — Features, validation harness, baseline (4 days)
**Build:** `elo.py`, `rolling.py`, `build_features` with cutoff assertions; `metrics.py` (RPS, log-loss, Brier); `devig.py`; expanding-window validation harness; multinomial logistic baseline scored against the market.
**Done when:** the baseline's RPS and the market's RPS are both printed for a held-out season, and a deliberate leakage test (feed a future column) makes the harness fail loudly.

### Phase 2 — Bayesian goal model (4 days)
**Build:** `bayes_goals.py` per `PYMC_MODEL_SPEC.md`; prior predictive check; convergence gate; half-life grid search on out-of-sample RPS; posterior predictive scoreline check.
**Done when:** the model samples cleanly (R̂ < 1.01, no divergences), the selected half-life is recorded, and simulated 0-0 and 1-1 frequencies match observed — the direct test that ρ is working.

### Phase 3 — XGBoost, calibration, blend (2 days)
**Build:** `xgb_outcome.py` with isotonic calibration; blend weight tuned on the validation season; comparison table of baseline / Bayes / XGBoost / blend / market.
**Done when:** all five RPS figures sit in one table, and the blend beats both components.

### Phase 4 — Season simulation and the pre-season forecast (3 days) — **THE GATE**
**Build:** `scorelines.py`, `season.py`; both uncertainty variants; `db/writer.py`; full pre-season run written to Neon.
**Done when:** `season_sim_snapshots` holds gameweek-0 rows for 20 teams in both variants, `predictions` holds every GW1 fixture with `created_at` before kickoff, and the two uncertainty variants show visibly different interval widths.

> **Do not start Phase 5 until Phase 4 has actually written a pre-season forecast to the database.** This is the one irreversible deadline in the project.

### Phase 5 — Dashboard (3 days)
**Build:** the four routes; Recharts visualisations; stale-state banner.
**Done when:** the dashboard renders live Neon data and correctly shows a stale warning when fed an artificially old run.

### Phase 6 — Automation (2 days)
**Build:** `weekly_forecast.yml`, `heartbeat.yml`, parquet caching, gated-run handling.
**Done when:** a manual workflow dispatch completes end to end and updates the dashboard, and a deliberately failed fit gates cleanly without corrupting the live forecast.

### Phase 7 — Documentation and case study (2 days)
**Build:** README with architecture diagram, honest limitations from PRD §11, the resume bullets; 60–90 second walkthrough video.
**Done when:** a reader who has never seen the project understands what it claims and what it doesn't.

---

## 8. Verification checkpoints

Confirm with evidence. Do not assume.

- **P0:** trigger the leakage trigger deliberately — insert a prediction dated after kickoff and confirm the database rejects it.
- **P1:** add a column containing the actual result to the feature matrix; validation RPS should collapse to near zero. If it doesn't, the harness is broken, not the model.
- **P2:** inspect the ArviZ summary directly. R̂, ESS, divergences — read the numbers, don't trust a print statement that says "converged".
- **P3:** confirm calibrated XGBoost probabilities are better calibrated than raw ones on the reliability diagram. If isotonic made things worse, the validation split is wrong.
- **P4:** compare the two uncertainty variants numerically. The with-uncertainty title probabilities must be *less* extreme. If they're identical, the posterior isn't propagating.
- **P5:** set the latest run's `computed_at` back three weeks in the database; the stale banner must appear.
- **P6:** delete a required secret and confirm the workflow fails loudly rather than writing partial data.

---

## 9. Paste-ready Claude Code prompts

Prefix each with: *"Read PRD.md and IMPLEMENTATION_GUIDE.md first, then complete the following task:"*
Expanded versions with acceptance criteria are in `CLAUDE_CODE_PROMPTS.md`.

- **P0:** "Scaffold the monorepo per section 2. Define the Drizzle schema for all seven tables in section 3, including the BEFORE INSERT trigger enforcing that predictions predate kickoff. Implement `ingest/football_data_csv.py` and `ingest/teams.py` with parquet caching. Do not build any model yet."
- **P1:** "Implement `features/elo.py`, `features/rolling.py`, and `features/build.py` with the shift(1) contract and cutoff assertions from section 4.2. Implement `evaluate/metrics.py` and `evaluate/devig.py`. Build an expanding-window validation harness and a multinomial logistic baseline. Print baseline RPS alongside market RPS."
- **P2:** "Implement `models/bayes_goals.py` following PYMC_MODEL_SPEC.md exactly — priors, ZeroSumNormal identifiability, Dixon-Coles tau via Potential, weighted likelihood. Add `db/gate.py` convergence checks and a half-life grid search over out-of-sample RPS. Include prior and posterior predictive checks."
- **P3:** "Implement `models/xgb_outcome.py` with isotonic calibration and `models/blend.py`. Tune the blend weight on the validation season. Output one comparison table: baseline, bayes, xgb, blend, market."
- **P4:** "Implement `simulate/scorelines.py` and `simulate/season.py` per PYMC_MODEL_SPEC.md section 8, producing both uncertainty variants. Implement `db/writer.py`. Wire `cli.py` to run a full pre-season forecast and write it to Neon."
- **P5:** "Build the four dashboard routes from section 6 with Drizzle reads and Recharts. Include the stale-state banner driven by computed_at."
- **P6:** "Write the two GitHub Actions workflows from section 5, including parquet caching and gated-run handling. A gated run must leave the previous forecast live and untouched."

---

## 10. Guardrails carried from the PRD

- Predictions are written before kickoff and are never updated afterwards — enforced in the database, not just in code.
- Expanding-window validation only. Never random k-fold.
- Time-decay half-life is selected offline by out-of-sample RPS, never estimated inside the model.
- A run that fails convergence writes a `model_runs` row and nothing else.
- Stale forecasts are labelled as stale. Always.
- Market odds are a benchmark, never a target. No staking logic.
- The honest ceiling (~53–55% accuracy) stays in the README. Results above it are treated as leakage until proven otherwise.
- `def` is signed so that higher means better defence — assert it in the plotting code.
- Scrapers are Python, run offline, and are never deployed.
- Zero monthly cost. Any change that introduces a paid dependency needs an explicit decision recorded here.
