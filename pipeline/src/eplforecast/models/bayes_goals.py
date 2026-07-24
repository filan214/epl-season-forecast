"""PyMC hierarchical Poisson goal model, implemented per PYMC_MODEL_SPEC.md.

Generative model (spec §1):

    log lam_home = mu + gamma + att[h] - def[a]
    log lam_away =        mu + att[a] - def[h]

Sign convention: ``def`` enters NEGATIVELY, so a large positive ``def`` is a
GOOD defence (concedes fewer). `assert_def_sign_convention` verifies this
against an actual table before anything is published (spec §9).

Identifiability: att/def use `pm.ZeroSumNormal` (spec §1) — a plain Normal
with soft centering samples poorly and R-hat shows it.

Time decay (spec §4): exponential match weights with `half_life_days` as a
function argument, NEVER a sampled parameter — the weighted likelihood is a
pseudo-likelihood and letting the sampler pick the decay is near-degenerate.
The half-life is chosen by grid search on out-of-sample RPS instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

# tau(0,0) can go negative for extreme rho/lambda combinations; log(tau) must
# stay defined. Spec §3: clip at a small positive floor — and if clipping fires
# on more than a handful of rows, tighten the rho prior, don't raise the floor.
TAU_FLOOR = 1e-6


def dixon_coles_tau(y_h, y_a, lam_h, lam_a, rho):
    """Dixon-Coles low-score correction factor per match (spec §3).

    `y_h`/`y_a` are fixed numpy arrays (masks are computed eagerly);
    `lam_h`/`lam_a`/`rho` may be pytensor variables. Only the four cells
    0-0, 0-1, 1-0, 1-1 are corrected; every other scoreline gets tau = 1.
    """
    m00 = (y_h == 0) & (y_a == 0)
    m01 = (y_h == 0) & (y_a == 1)
    m10 = (y_h == 1) & (y_a == 0)
    m11 = (y_h == 1) & (y_a == 1)

    tau = pt.ones_like(lam_h)
    tau = pt.switch(m00, 1.0 - lam_h * lam_a * rho, tau)
    tau = pt.switch(m01, 1.0 + lam_h * rho, tau)
    tau = pt.switch(m10, 1.0 + lam_a * rho, tau)
    tau = pt.switch(m11, 1.0 - rho, tau)
    return pt.clip(tau, TAU_FLOOR, np.inf)


def _team_indices(df: pd.DataFrame, teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t_index = {t: i for i, t in enumerate(teams)}
    home_idx = df["home_team"].map(t_index).to_numpy()
    away_idx = df["away_team"].map(t_index).to_numpy()
    return home_idx, away_idx


def build_model(
    df: pd.DataFrame,
    teams: list[str],
    cutoff: pd.Timestamp,
    half_life_days: float = 180,
    promoted_mask: np.ndarray | None = None,
) -> pm.Model:
    """Build the hierarchical Poisson model exactly per PYMC_MODEL_SPEC.md §5.

    `df` columns: date, home_team, away_team, home_goals, away_goals.
    Only rows with date < cutoff may be present — asserted, this is a leakage
    guard. `promoted_mask` is a per-team (aligned with `teams`) 0/1 array
    marking sides with no EPL history in the decay window (spec §6).
    """
    assert (df["date"] < cutoff).all(), "leakage: matches at or after cutoff in training set"

    home_idx, away_idx = _team_indices(df, teams)
    y_h = df["home_goals"].to_numpy()
    y_a = df["away_goals"].to_numpy()

    # Exponential time decay (spec §4). half_life_days is a plain argument:
    # sampling it would let a pseudo-likelihood inflate in-sample fit.
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

        # def enters NEGATIVELY: higher def = better defence (spec §1).
        log_lh = mu + gamma + att[home_idx] - dfn[away_idx]
        log_la = mu + att[away_idx] - dfn[home_idx]

        if promoted_mask is not None:
            prom_pen = pm.Normal("promoted_penalty", -0.20, 0.15)
            log_lh = log_lh + prom_pen * promoted_mask[home_idx]
            log_la = log_la + prom_pen * promoted_mask[away_idx]

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


def fit_bayes(
    df: pd.DataFrame,
    teams: list[str],
    cutoff: pd.Timestamp,
    half_life_days: float = 180,
    promoted_mask: np.ndarray | None = None,
    **sample_kwargs,
) -> xr.DataTree:
    """Build the spec model and sample it (guide §4.3 entry point).

    Returns the InferenceData — an `xarray.DataTree` under ArviZ >= 1.0.
    Sampling defaults follow spec §5; keyword overrides exist for tests and
    the half-life grid search, not for weakening the production run.
    """
    kwargs = {
        "draws": 2000,
        "tune": 2000,
        "chains": 4,
        "target_accept": 0.9,
        "random_seed": 42,
    } | sample_kwargs
    model = build_model(df, teams, cutoff, half_life_days, promoted_mask)
    with model:
        return pm.sample(**kwargs)


def assert_def_sign_convention(idata, matches: pd.DataFrame) -> float:
    """Verify higher `def` = better defence against an actual table (spec §9).

    Computes the Spearman correlation between posterior-mean `def` and goals
    conceded per match; it must be negative or the table would be published
    inverted. Returns the correlation for logging.
    """
    from scipy.stats import spearmanr

    post = idata.posterior
    ds = post.to_dataset() if hasattr(post, "to_dataset") else post
    def_mean = ds["def"].mean(("chain", "draw"))

    home = matches.groupby("home_team")["away_goals"].agg(["sum", "count"])
    away = matches.groupby("away_team")["home_goals"].agg(["sum", "count"])
    totals = home.add(away, fill_value=0)
    conceded_per_match = totals["sum"] / totals["count"]

    # Only teams that actually appear in `matches` can be checked.
    teams = [t for t in def_mean["team"].to_numpy() if t in conceded_per_match.index]
    corr = float(
        spearmanr(
            def_mean.sel(team=teams).to_numpy(),
            conceded_per_match.loc[teams].to_numpy(),
        ).statistic
    )
    assert corr < 0, (
        f"def sign convention violated: Spearman(def, conceded/match) = {corr:.3f}, "
        "expected negative. Higher def must mean FEWER goals conceded — check the "
        "minus sign on def in the linear predictor before publishing any table."
    )
    return corr


def _stacked_params(group) -> dict[str, np.ndarray]:
    """Flatten an InferenceData group to {var: (S, ...) array}, samples first.

    ArviZ 1.x groups are DataTree nodes rather than Datasets; normalise first.
    """
    ds = group.to_dataset() if hasattr(group, "to_dataset") else group
    stacked = ds.stack(sample=("chain", "draw"))
    return {
        name: stacked[name].transpose("sample", ...).to_numpy()
        for name in stacked.data_vars
    }


def _lambdas_from_draws(
    params: dict[str, np.ndarray],
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    promoted_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(S, M) home/away Poisson rates from S parameter draws, in numpy.

    The likelihood lives in a Potential, so PyMC cannot simulate goals for us;
    both predictive checks rebuild the linear predictor from parameter draws.
    """
    mu = params["mu"][:, None]
    gamma = params["home_adv"][:, None]
    att, dfn = params["att"], params["def"]
    log_lh = mu + gamma + att[:, home_idx] - dfn[:, away_idx]
    log_la = mu + att[:, away_idx] - dfn[:, home_idx]
    if promoted_mask is not None and "promoted_penalty" in params:
        pen = params["promoted_penalty"][:, None]
        log_lh = log_lh + pen * promoted_mask[home_idx]
        log_la = log_la + pen * promoted_mask[away_idx]
    return np.exp(log_lh), np.exp(log_la)


def prior_predictive_check(
    df: pd.DataFrame,
    teams: list[str],
    cutoff: pd.Timestamp,
    half_life_days: float = 180,
    promoted_mask: np.ndarray | None = None,
    draws: int = 200,
    random_seed: int | None = None,
) -> dict:
    """Spec §2 sanity check, run BEFORE fitting: simulate `draws` prior draws
    and report goals per match. Expect the mean in roughly 2-4 and 6+ goal
    scorelines rare but not impossible; 12-goal games mean sigma_att is too wide.
    """
    model = build_model(df, teams, cutoff, half_life_days, promoted_mask)
    with warnings.catch_warnings():
        # PyMC warns that Potentials are ignored in predictive sampling. Here
        # that is exactly right: we want parameter draws from the untouched
        # priors and simulate the goals ourselves below.
        warnings.filterwarnings("ignore", message=".*Potentials.*", category=UserWarning)
        prior = pm.sample_prior_predictive(draws=draws, model=model, random_seed=random_seed)

    params = _stacked_params(prior.prior)
    home_idx, away_idx = _team_indices(df, teams)
    lam_h, lam_a = _lambdas_from_draws(params, home_idx, away_idx, promoted_mask)

    rng = np.random.default_rng(random_seed)
    y_h = rng.poisson(lam_h)
    y_a = rng.poisson(lam_a)
    return {
        "n_draws": int(draws),
        "mean_goals_per_match": float((y_h + y_a).mean()),
        "p_six_plus": float((np.maximum(y_h, y_a) >= 6).mean()),
    }


def _thinned_posterior_params(idata, thin_to: int) -> dict[str, np.ndarray]:
    params = _stacked_params(idata.posterior)
    n_samples = params["mu"].shape[0]
    take = np.linspace(0, n_samples - 1, min(thin_to, n_samples)).astype(int)
    return {k: v[take] for k, v in params.items()}


def predict_outcome_probs(
    idata,
    fixtures: pd.DataFrame,
    teams: list[str],
    promoted_mask: np.ndarray | None = None,
    max_goals: int = 10,
    thin_to: int = 500,
) -> np.ndarray:
    """(n, 3) [P(home), P(draw), P(away)] per fixture, averaged over draws.

    Outcome order matches evaluate.metrics (0=home, 1=draw, 2=away).
    `fixtures` only needs home_team/away_team — this is the pre-kickoff path.
    Cell probabilities follow spec §7's score_matrix, vectorised over draws
    and fixtures, so uncertainty is integrated rather than plugged in at the
    posterior mean.
    """
    from scipy.stats import poisson

    params = _thinned_posterior_params(idata, thin_to)
    home_idx, away_idx = _team_indices(fixtures, teams)
    lam_h, lam_a = _lambdas_from_draws(params, home_idx, away_idx, promoted_mask)
    rho = params["rho"][:, None]

    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[:, None, None], lam_h[None, :, :])  # (G+1, S, M)
    pa = poisson.pmf(goals[:, None, None], lam_a[None, :, :])
    tau = {
        (0, 0): np.maximum(1 - lam_h * lam_a * rho, TAU_FLOOR),
        (0, 1): np.maximum(1 + lam_h * rho, TAU_FLOOR),
        (1, 0): np.maximum(1 + lam_a * rho, TAU_FLOOR),
        (1, 1): np.maximum(1 - rho, TAU_FLOOR),
    }

    home = np.zeros_like(lam_h)
    draw = np.zeros_like(lam_h)
    away = np.zeros_like(lam_h)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            cell = ph[i] * pa[j]
            if (i, j) in tau:
                cell = cell * tau[(i, j)]
            if i > j:
                home += cell
            elif i == j:
                draw += cell
            else:
                away += cell

    probs = np.stack([home, draw, away], axis=-1)  # (S, M, 3)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs.mean(axis=0)


# The four scorelines the Dixon-Coles correction exists for (spec §9): if the
# simulated frequencies of these don't track observed, rho isn't doing its job.
PPC_SCORELINES = ((0, 0), (1, 0), (1, 1), (2, 1))


def posterior_predictive_check(
    idata,
    df: pd.DataFrame,
    teams: list[str],
    promoted_mask: np.ndarray | None = None,
    max_goals: int = 10,
    thin_to: int = 500,
) -> pd.DataFrame:
    """Spec §9: compare simulated vs observed frequencies of 0-0, 1-0, 1-1, 2-1.

    Simulated frequencies are exact cell probabilities of the DC-corrected,
    truncated scoreline grid, averaged over (thinned) posterior draws and the
    matches in `df` — lower variance than re-sampling scorelines.
    """
    from scipy.stats import poisson

    params = _thinned_posterior_params(idata, thin_to)
    home_idx, away_idx = _team_indices(df, teams)
    lam_h, lam_a = _lambdas_from_draws(params, home_idx, away_idx, promoted_mask)
    rho = params["rho"][:, None]

    p_h = {g: poisson.pmf(g, lam_h) for g in (0, 1, 2)}
    p_a = {g: poisson.pmf(g, lam_a) for g in (0, 1)}
    tau = {
        (0, 0): np.maximum(1 - lam_h * lam_a * rho, TAU_FLOOR),
        (0, 1): np.maximum(1 + lam_h * rho, TAU_FLOOR),
        (1, 0): np.maximum(1 + lam_a * rho, TAU_FLOOR),
        (1, 1): np.maximum(1 - rho, TAU_FLOOR),
    }

    # Normaliser of the truncated grid: the independent mass plus what the
    # four corrected cells add or remove.
    z = poisson.cdf(max_goals, lam_h) * poisson.cdf(max_goals, lam_a)
    for (i, j), t in tau.items():
        z = z + p_h[i] * p_a[j] * (t - 1)

    y_h = df["home_goals"].to_numpy()
    y_a = df["away_goals"].to_numpy()
    rows = {}
    for i, j in PPC_SCORELINES:
        cell = p_h[i] * p_a[j] * tau.get((i, j), 1.0) / z
        rows[f"{i}-{j}"] = {
            "simulated": float(cell.mean()),
            "observed": float(((y_h == i) & (y_a == j)).mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")[["simulated", "observed"]]
