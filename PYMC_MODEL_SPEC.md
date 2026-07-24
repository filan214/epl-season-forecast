# Bayesian hierarchical Poisson goal model — spec

> Drop-in section for `IMPLEMENTATION_GUIDE.md` (EPL 2026-27 season predictor). Defines the generative model, priors, time decay, sampling config, and the posterior-predictive season simulation. The XGBoost outcome classifier is specified separately; this doc ends with the blend contract between them.

---

## 1. Generative model

For match `m` between home team `h(m)` and away team `a(m)`:

```
log λ_home,m = μ + γ + att[h] − def[a]
log λ_away,m =     μ + att[a] − def[h]

y_home,m , y_away,m  ~  Poisson(λ), coupled by the Dixon-Coles τ correction
```

| Parameter | Meaning |
|---|---|
| `μ` | League baseline log scoring rate |
| `γ` | Home advantage (log multiplier) |
| `att[j]` | Team `j` attacking strength (higher = scores more) |
| `def[j]` | Team `j` defensive strength (**higher = concedes fewer** — note the minus sign in the linear predictor) |
| `ρ` | Dixon-Coles low-score dependence |

**Sign convention matters and is a frequent bug source.** `def` enters negatively, so a *large positive* `def` means a *good* defence. Assert this in the plotting code or you will publish an inverted table.

### Identifiability

`att` and `def` are only identified up to an additive constant (shift all `att` up by 0.1, shift `μ` down by 0.1 — same likelihood). Enforce sum-to-zero with `pm.ZeroSumNormal`. Do **not** use a plain `pm.Normal` with a soft centering prior; it samples poorly and R̂ will show it.

---

## 2. Priors

| Parameter | Prior | Reasoning |
|---|---|---|
| `μ` | `Normal(0.3, 0.25)` | EPL averages ~1.4 goals per team per match; `log(1.4) ≈ 0.34`. Weakly informative. |
| `γ` (home adv) | `Normal(0.20, 0.15)` | Historic home advantage ≈ 0.25 in log terms, but it has **declined post-2020** to roughly 0.15–0.20. Don't use a legacy 0.3 prior. |
| `σ_att` | `HalfNormal(0.5)` | Team attack strengths realistically span roughly ±0.5 in log space (a factor of ~1.6 between best and average). |
| `σ_def` | `HalfNormal(0.5)` | Same reasoning. |
| `att` | `ZeroSumNormal(σ_att, dims="team")` | Partial pooling — this is the whole point of the model. |
| `def` | `ZeroSumNormal(σ_def, dims="team")` | |
| `ρ` | `Normal(0, 0.1)` | Published estimates for English football land near −0.13. A zero-centred prior lets the data decide the sign. |

**Run a prior predictive check before fitting.** Simulate 200 draws and confirm total goals per match sits in roughly 2–4 and that 6-0 scorelines are rare but not impossible. If the prior predictive produces 12-goal games, `σ_att` is too wide.

---

## 3. Dixon-Coles correction

Plain independent Poissons underpredict 0-0, 1-1, and (to a lesser extent) 1-0 / 0-1. The τ correction fixes exactly those four cells:

```
τ(0,0) = 1 − λ_h · λ_a · ρ
τ(0,1) = 1 + λ_h · ρ
τ(1,0) = 1 + λ_a · ρ
τ(1,1) = 1 − ρ
τ(x,y) = 1        otherwise
```

Because this isn't a standard distribution, add it to the log-likelihood via `pm.Potential` rather than trying to express it as a `pm.Distribution`.

τ can go negative for extreme ρ/λ combinations, which makes `log(τ)` undefined. Clip at a small positive floor. If clipping fires on more than a handful of rows, tighten the ρ prior instead of raising the floor.

---

## 4. Time decay

Recent matches should count more. Weight each match by:

```
w_m = exp(−ξ · Δt_m)          Δt_m = days between match m and the cutoff date
ξ   = ln(2) / half_life_days
```

Applied as a **weighted likelihood** inside the `Potential`.

### Two things to be honest about

1. **Do not estimate ξ inside the model.** The weighted likelihood is a pseudo-likelihood, and letting the sampler choose ξ is close to degenerate — it can inflate in-sample fit without improving prediction. Select ξ by grid search on *out-of-sample* RPS instead (§7).
2. **A weighted likelihood is not a strict Bayesian posterior.** Credible intervals are approximate. The rigorous alternative is a dynamic model (Gaussian random walk on `att`/`def` across gameweeks), which is a documented v2 upgrade — heavier to sample, meaningfully better mid-season. Note the limitation in the README rather than hiding it.

**Grid to search:** half-life ∈ {60, 90, 120, 180, 270, 365, ∞} days. Run once pre-season; don't re-tune weekly (that's a leakage vector).

---

## 5. Reference implementation

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def dixon_coles_tau(y_h, y_a, lam_h, lam_a, rho):
    """Masks are fixed numpy arrays; lam_h/lam_a/rho are pytensor vars."""
    m00 = (y_h == 0) & (y_a == 0)
    m01 = (y_h == 0) & (y_a == 1)
    m10 = (y_h == 1) & (y_a == 0)
    m11 = (y_h == 1) & (y_a == 1)

    tau = pt.ones_like(lam_h)
    tau = pt.switch(m00, 1.0 - lam_h * lam_a * rho, tau)
    tau = pt.switch(m01, 1.0 + lam_h * rho, tau)
    tau = pt.switch(m10, 1.0 + lam_a * rho, tau)
    tau = pt.switch(m11, 1.0 - rho, tau)
    return pt.clip(tau, 1e-6, np.inf)


def build_model(df, teams, cutoff, half_life_days=180, promoted_mask=None):
    """
    df columns: date, home_team, away_team, home_goals, away_goals
    Only rows with date < cutoff should be present. Assert it.
    """
    assert (df["date"] < cutoff).all(), "leakage: matches at or after cutoff in training set"

    t_index = {t: i for i, t in enumerate(teams)}
    home_idx = df["home_team"].map(t_index).to_numpy()
    away_idx = df["away_team"].map(t_index).to_numpy()
    y_h = df["home_goals"].to_numpy()
    y_a = df["away_goals"].to_numpy()

    xi = np.log(2) / half_life_days
    delta_days = (cutoff - df["date"]).dt.days.to_numpy()
    w = np.exp(-xi * delta_days)

    with pm.Model(coords={"team": teams}) as model:
        mu = pm.Normal("mu", 0.30, 0.25)
        gamma = pm.Normal("home_adv", 0.20, 0.15)

        sigma_att = pm.HalfNormal("sigma_att", 0.5)
        sigma_def = pm.HalfNormal("sigma_def", 0.5)

        att = pm.ZeroSumNormal("att", sigma=sigma_att, dims="team")
        dfn = pm.ZeroSumNormal("def", sigma=sigma_def, dims="team")

        rho = pm.Normal("rho", 0.0, 0.1)

        log_lh = mu + gamma + att[home_idx] - dfn[away_idx]
        log_la = mu + att[away_idx] - dfn[home_idx]

        if promoted_mask is not None:
            prom_pen = pm.Normal("promoted_penalty", -0.20, 0.15)
            ph = promoted_mask[home_idx]
            pa = promoted_mask[away_idx]
            log_lh = log_lh + prom_pen * ph
            log_la = log_la + prom_pen * pa

        lam_h = pt.exp(log_lh)
        lam_a = pt.exp(log_la)

        tau = dixon_coles_tau(y_h, y_a, lam_h, lam_a, rho)

        ll = (
            pm.logp(pm.Poisson.dist(lam_h), y_h)
            + pm.logp(pm.Poisson.dist(lam_a), y_a)
            + pt.log(tau)
        )
        pm.Potential("weighted_dc_loglik", (w * ll).sum())

    return model


with build_model(train_df, teams, cutoff, half_life_days=180):
    idata = pm.sample(
        draws=2000, tune=2000, chains=4,
        target_accept=0.9, random_seed=42,
    )
```

**Runtime:** roughly 45 parameters over ~3,800 rows. Expect single-digit minutes on a laptop, comfortably inside a GitHub Actions runner. Persist the `InferenceData` to NetCDF so a failed weekly run can fall back to last week's trace rather than serving nothing.

---

## 6. Promoted teams

Three teams each season have zero EPL history in the training window. Partial pooling shrinks them to the league mean — but the league mean is *too generous*: promoted sides historically underperform it.

| Option | Effort | Verdict |
|---|---|---|
| Let pooling handle it | none | Systematically over-rates promoted teams, especially gameweeks 1–8 |
| `promoted_penalty` offset (shown in the code above) | ~1 hour | **v1 choice.** Informative prior centred at −0.20, applied to teams with no EPL matches in the decay window |
| Include Championship seasons with a league-strength offset parameter | ~1 day | Correct approach, real gain. Backlog it. |

---

## 7. Match probabilities & validation

### Scoreline grid → W/D/A

```python
from scipy.stats import poisson

def score_matrix(lam_h, lam_a, rho, max_goals=10):
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    M = np.outer(ph, pa)                       # M[i, j] = P(home=i, away=j)
    M[0, 0] *= 1 - lam_h * lam_a * rho
    M[0, 1] *= 1 + lam_h * rho
    M[1, 0] *= 1 + lam_a * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()

def outcome_probs(M):
    return (
        np.tril(M, -1).sum(),   # home win  (i > j)
        np.trace(M),            # draw
        np.triu(M, 1).sum(),    # away win
    )
```

Over/under 2.5, BTTS, and clean sheets all read off the same matrix — no extra model.

### Validation protocol

- **Split:** expanding window by season. Train on seasons 1..k, evaluate on season k+1. Never random k-fold.
- **Primary metric:** ranked probability score (RPS). Ordinal-aware, which accuracy is not.
- **Secondary:** log-loss, Brier score, and a reliability diagram (predicted vs. observed, 10 bins).
- **Benchmark:** de-vigged bookmaker closing odds from the football-data.co.uk columns. Basic de-vig is normalising `1/odds` to sum to 1; note in the README that this slightly favours the favourite versus a Shin or power de-vig. Matching closing-odds RPS is a strong result — beating it consistently is a red flag for leakage, not a triumph.

---

## 8. Season simulation (replaces the standalone Monte Carlo)

For each posterior draw `s` (thin to ~1,000–2,000):

1. Compute `λ_h`, `λ_a` for every **remaining** fixture using that draw's parameters.
2. Build the score matrix, sample one `(i, j)` from the flattened, normalised pmf.
3. Apply results to the current real table; sort by points, then goal difference, then goals scored.
4. Record final position for all 20 teams.

Across draws this yields title / top-4 / relegation probabilities that carry **both** match randomness and parameter uncertainty. Simulating from posterior-mean parameters instead would understate interval width — that difference is the main correctness argument for this model over point-estimate Dixon-Coles, so make it visible: plot one chart with parameter uncertainty and one without.

Cost: 1,000 draws × ~380 fixtures × an 11×11 grid. Vectorise the grid construction across fixtures; the whole simulation should run in seconds.

---

## 9. Convergence checklist (gate before publishing any prediction)

Fail the cron job loudly rather than serving numbers from a bad fit.

- [ ] R̂ < 1.01 on every parameter
- [ ] Bulk and tail ESS > 400
- [ ] Zero divergences (raise `target_accept` to 0.95 before touching the model)
- [ ] Energy plot (BFMI) shows no pathology
- [ ] Prior predictive: goals per match in ~2–4
- [ ] Posterior predictive: simulated scoreline frequencies match observed, especially 0-0 and 1-1 — this is the direct test of whether ρ is doing its job
- [ ] `def` sign convention verified against the actual table (best defence should have the highest `def`)

---

## 10. Blend contract with XGBoost

Both models emit a `(p_home, p_draw, p_away)` vector per fixture.

```
p_final = w · p_bayes + (1 − w) · p_xgb
```

- Tune `w` on the validation season by RPS; expect it to land near 0.5.
- Calibrate the XGBoost probabilities (isotonic) **before** blending, not after.
- Re-normalise after blending.
- Season simulation uses the Bayesian scoreline matrix, not the blend, because it needs goal counts for goal difference. Log the divergence between blended W/D/A and Bayesian-only W/D/A as a monitoring signal — a widening gap means one model is drifting.

---

## 11. Known v1 limitations (state these in the README)

- Weighted likelihood is a pseudo-likelihood; credible intervals are approximate.
- Team strengths are static within a fit — no in-season dynamics until the random-walk upgrade.
- No explicit modelling of injuries, European fixture congestion, or managerial changes.
- Promoted-team handling is a prior offset, not a proper cross-league model.
- Expected predictive gain over a well-tuned frequentist Dixon-Coles is small. The gains claimed here are calibration and uncertainty quantification — don't oversell accuracy.
