# PRD — EPL Season Forecast (2026-27)

> A live-updating Bayesian forecasting system for the 2026-27 English Premier League season. Re-simulates the full league table every gameweek, publishes title / top-four / relegation probabilities, and tracks its own calibration against bookmaker closing odds for the entire nine-month season.
>
> Companion doc: `IMPLEMENTATION_GUIDE.md` (the *how*). Model math: `PYMC_MODEL_SPEC.md`.

---

## 1. One-liner

Every Monday, an automated pipeline ingests the weekend's results, refits a Bayesian hierarchical Poisson goal model, simulates the remaining season 2,000 times from the posterior, and publishes updated table probabilities plus a running scorecard of how well-calibrated its own past predictions were.

---

## 2. Why this project exists

This is a portfolio piece with a specific job: demonstrate **production ML judgment**, not just model-fitting ability. The differentiators, in order of interview value:

1. **It runs for nine months and monitors itself.** Most candidate ML projects are notebooks that produced one number once. This one has a cron, a convergence gate, a fallback path, and a public accuracy record it cannot retroactively edit.
2. **It benchmarks against a real adversary.** Bookmaker closing odds are a hard, honest baseline. "My model's RPS versus the market's" is a quantifiable claim; "my model got 58% accuracy" is not.
3. **It uses Bayesian uncertainty for something load-bearing.** The posterior propagates into the season simulation, so the intervals mean something. A point-estimate model would produce narrower, wrong intervals — and the project shows both, side by side.

### Deliberate separation from WC 2026 Predictor

That project used Elo + Poisson + Monte Carlo as the prediction engine, ran once, and produced a CLI output. This project:

- Lives in a **separate repository** with its own README and resume line
- **Demotes Elo to a feature column** inside a supervised model rather than using it as the engine
- Replaces point-estimate Poisson with a **hierarchical Bayesian model with partial pooling**
- Replaces the standalone Monte Carlo with **posterior predictive sampling**
- Is a **monitored live system**, not a one-shot script

If a reviewer reads both READMEs back to back, the second must not read as a re-run of the first. That's a hard requirement, not a nice-to-have.

---

## 3. Target roles

| Role | Fit | What it evidences |
|---|---|---|
| **Data Scientist** | Primary | Bayesian modelling, partial pooling, calibration, proper validation protocol, honest metric selection |
| **AI / ML Engineer** | Strong secondary | Scheduled pipeline, convergence gating, graceful degradation, model-run provenance, cost-constrained infra |
| **Data Analyst** | Partial | The dashboard and the calibration reporting; weaker fit than the two above |
| **Freelance dev** | Low | Not a client-facing niche. Don't pitch this one commercially. |

### Resume bullets this produces

> Built a Bayesian hierarchical Poisson goal model (PyMC) with partial pooling and time decay, blended with a calibrated XGBoost outcome classifier, propagating full posterior uncertainty into season-long title and relegation probabilities.

> Deployed the forecast as a self-monitoring weekly pipeline (GitHub Actions → Neon Postgres → Next.js), tracking ranked probability score against de-vigged bookmaker closing odds across a full Premier League season.

---

## 4. Goals

**G1.** Publish a locked, timestamped forecast for every remaining fixture **before** each gameweek kicks off.
**G2.** Report table-level probabilities (title, top four, top six, relegation) with credible intervals that reflect parameter uncertainty, not just match randomness.
**G3.** Maintain a public, append-only calibration record: RPS, log-loss, and Brier score versus the market baseline, updated weekly.
**G4.** Run the whole thing at **zero monthly cost** on free tiers.
**G5.** Never publish a forecast from a fit that failed convergence diagnostics.

### Non-goals (v1)

- Player-level projections (goals, assists, minutes) — overlaps FormWatch
- In-play or live-minute prediction
- Any league other than the EPL
- Betting recommendations, staking, or bankroll logic — the market is a *benchmark*, not a target
- Injury, suspension, or lineup modelling
- User accounts, auth, or personalisation
- A public API

---

## 5. Users

| User | Need |
|---|---|
| **Recruiter / hiring manager** | Lands on the dashboard from the CV, sees a live system with a visible accuracy record, spends 90 seconds, leaves convinced it's real |
| **Technical interviewer** | Reads the README and `PYMC_MODEL_SPEC.md`, probes the leakage guards and the calibration protocol |
| **Me** | Needs the weekly run to be genuinely unattended. If it requires babysitting, it dies by October and the portfolio value inverts. |

The third user is the one that determines the architecture. Unattended operation is a product requirement.

---

## 6. Scope — v1

### 6.1 Match-level outputs (per fixture, per gameweek)

- P(home win), P(draw), P(away win)
- Expected goals for each side (λ)
- Full scoreline distribution over an 11×11 grid
- P(over 2.5 goals), P(both teams to score), P(clean sheet) — all derived from the same grid, no extra models

### 6.2 Season-level outputs (per team, per gameweek)

- Title probability
- Top-four (UCL) probability
- Top-six probability
- Relegation probability
- Expected final points, plus 10th/90th percentile points range
- Expected final position

### 6.3 Model transparency outputs

- Per-team attack and defence posterior means with 94% HDI
- The same season simulation run **with and without** parameter uncertainty, shown side by side — this is the core pedagogical exhibit

### 6.4 Calibration outputs

- Cumulative and rolling RPS, log-loss, Brier score
- The same three metrics for the de-vigged market baseline
- Reliability diagram (predicted vs. observed, 10 bins)
- Per-run diagnostics: max R̂, divergence count, minimum bulk ESS

### 6.5 Operational requirements

- Weekly automated run, no manual step
- A run that fails diagnostics is **gated**: last good forecast stays live, marked stale in the UI with its `computed_at` date
- Every prediction row carries a `model_run_id` traceable to a git SHA and a cutoff date

---

## 7. Backlog (explicitly not v1)

Documented so scope doesn't drift mid-season. Each of these is a defensible v2, and saying so in the README is better than pretending v1 is complete.

| Item | Why deferred |
|---|---|
| Dynamic team strengths (Gaussian random walk on att/def across gameweeks) | The statistically correct replacement for the weighted-likelihood time decay. Heavier to sample. Real gain — top of the backlog. |
| Championship data with a league-strength offset, replacing the promoted-team prior | Correct fix for the promoted-team cold start; ~1 day of work, not affordable pre-GW1 |
| Squad-value delta and fixture congestion as covariates inside the Bayesian model | Currently only reachable through the XGBoost side of the blend |
| Multi-league expansion | Scope trap. The whole project's credibility rests on doing one league properly. |
| Public API | No consumer |
| Shin / power method de-vigging | Basic normalisation is fine for v1; document the bias it introduces |

---

## 8. Data sources

| Source | Data | Access | Cadence |
|---|---|---|---|
| **football-data.co.uk** | Match results, shots, corners, cards, **and closing odds from 15+ bookmakers**, back to 1993/94 | Free CSV, no key | Bulk once pre-season; the odds columns are the benchmark |
| **Understat** | xG, xA, npxG per match and team | Scrape via `understatapi` / ScraperFC | Bulk pre-season, then weekly |
| **Transfermarkt** | Squad market values | Reuse the Transfer Globe scraper | Once pre-season, once in January |
| **football-data.org** | Live fixtures, results, standings | Free API key, 10 req/min, EPL included | Weekly |

### Notes and constraints

- **FBref is dead as an advanced-stats source.** Opta stopped supplying it in January 2026; only basic tables remain. Understat is the replacement and is the single point of failure for xG — cache every pull to parquet so a Understat outage doesn't block a run.
- **Scraping etiquette:** conservative throttle on Understat, one pull per gameweek. Never parallelise it.
- **Scrapers are Python and run offline.** They are never deployed. This follows the established pattern — scraper language does not need to match app language.
- Rate limits are a non-issue at one league on a weekly cadence.

---

## 9. Modelling approach

Full math in `PYMC_MODEL_SPEC.md`. Summary of what's used and why:

| Component | Choice | Rationale |
|---|---|---|
| Goal model | Bayesian hierarchical Poisson, PyMC, with Dixon-Coles low-score correction and exponential time decay | Partial pooling is the correct fix for early-season sparsity, when the forecast is most interesting and most data-starved. Posterior predictive sampling *is* the season simulation. |
| Outcome model | XGBoost multiclass + isotonic calibration | Absorbs covariates (Elo, rest days, squad value delta, congestion) that are awkward to attach to the hierarchical model |
| Baseline | Multinomial logistic regression | Honesty check. On ~3,800 rows it is often competitive. If boosting can't beat it, that is a finding to publish, not hide. |
| Blend | `w·p_bayes + (1−w)·p_xgb`, `w` tuned on a held-out season | Usually beats either alone; expect `w ≈ 0.5` |
| Season simulation | Posterior predictive draws | Carries parameter uncertainty, which point-estimate Monte Carlo does not |

### Explicitly rejected

- **LSTM or any neural network.** ~3,800 tabular rows is precisely where gradient boosting wins. It would also duplicate the thesis work and add zero portfolio variety.
- **Random k-fold cross-validation.** Guaranteed leakage. Expanding-window by season only.
- **Estimating the time-decay rate inside the model.** Degenerate; select it by out-of-sample RPS.

---

## 10. Success metrics

### Primary

**Ranked probability score (RPS)**, because it is ordinal-aware — predicting a draw when the home side wins is less wrong than predicting an away win, and accuracy cannot express that.

| Metric | Target | Note |
|---|---|---|
| Season RPS vs. de-vigged market closing odds | **Within 0.005** of market | Matching the market is a strong result |
| Reliability diagram | Within ±5pp of the diagonal in the 20–80% bins | Calibration is the actual claim of this project |
| R̂ on every published run | < 1.01 | Hard gate, not a target |
| Weekly run success rate | > 90% of gameweeks unattended | Product requirement |
| Monthly infra cost | $0.00 | |

### Honest ceiling — state this in the README

Three-way match accuracy tops out around **53–55%**, which is roughly where bookmakers sit. If the model reports 65%, there is leakage — that is a bug report, not a result. The target is framed against market RPS, never as a raw accuracy figure.

This is a direct carry-over from the F1 predictor, where the PRD target (≤1.5 MAE) was miscalibrated against the real baseline (~2.2 MAE). Set the ceiling honestly at the start.

**Expected predictive gain over a well-tuned frequentist Dixon-Coles is small.** The wins claimed here are calibration and uncertainty quantification. Do not oversell accuracy.

---

## 11. Known limitations — publish these

A README that names its own ceiling is stronger evidence of judgment than one claiming an undemonstrable win.

- Weighted likelihood is a pseudo-likelihood; credible intervals are approximate
- Team strengths are static within a fit — no in-season dynamics until the random-walk upgrade
- No modelling of injuries, European fixture congestion, or managerial changes
- Promoted teams handled by a prior offset, not a proper cross-league model
- Basic odds de-vigging slightly favours the favourite versus Shin or power methods
- Understat xG is a single vendor's model, not ground truth
- One league, one season of live evidence — the calibration record is thin until roughly gameweek 15

---

## 12. Timeline

**Today: 23 July 2026. GW1 is mid-August — confirm the exact date from the fixture list before planning around it.**

The deadline is real and it shapes phase ordering. There is exactly one unmissable gate:

> **A locked, timestamped pre-season forecast must be written to the database before the first ball of GW1.**

Everything else can land late. The dashboard can ship in September; the forecast cannot. A season predictor that starts at gameweek 6 has lost the most valuable part of its own evidence base.

| Block | Days | Must finish before GW1? |
|---|---|---|
| Scaffold, schema, historical ingest | 2 | Yes |
| Feature pipeline + validation harness + baseline | 4 | Yes |
| PyMC model + convergence gate + decay tuning | 4 | Yes |
| XGBoost + calibration + blend | 2 | Yes |
| Season simulation + DB writes + pre-season forecast | 3 | **Yes — this is the gate** |
| Next.js dashboard | 3 | No |
| Weekly automation + fallback + heartbeat | 2 | Soon after |
| Calibration views + README + case study | 2 | No |

~15 days to the gate, ~22 days total. Tight but feasible against a three-week runway, provided the dashboard is allowed to slip.

---

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Project stalls by October; dashboard shows three stale gameweeks** | Medium | **Severe** — worse than never building it | Unattended operation is a product requirement. Fallback path, gating, and stale labelling are all v1 scope, not polish. |
| GW1 deadline missed | Medium | High | Phase order above; dashboard is the designated slip item |
| Understat blocks or breaks | Low-Medium | Medium | Cache every pull to parquet; model degrades to non-xG features rather than failing |
| MCMC fails to converge on a given week | Low | Medium | Convergence gate + serve last good run marked stale |
| Neon free tier policy change | Low | Medium | Monthly heartbeat cron as belt-and-braces; the cost is zero |
| Leakage produces implausibly good results | Medium | High | DB-level `CHECK` that predictions predate kickoff; expanding-window validation only; the 55% accuracy ceiling as a tripwire |
| Model underperforms the market | **High — expect it** | Low | This is the honest expected outcome. Report it plainly. A calibrated model that slightly trails the market is a credible result; a model claiming to beat it is a red flag. |

The last row matters most. The project's value does not depend on winning. It depends on measuring properly and reporting honestly — which is exactly the skill the target roles are hiring for.
