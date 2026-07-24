"""Tests for models/bayes_goals.py — Phase 2 PyMC hierarchical Poisson model.

Everything here pins PYMC_MODEL_SPEC.md contracts: the four Dixon-Coles tau
cells and their clip floor (§3), the ZeroSumNormal identifiability constraint
(§1), the negative-def sign convention (§1/§9), exponential time decay with
half_life_days as a function argument — never a sampled parameter (§4) — the
promoted_penalty prior (§6), and the prior / posterior predictive checks
(§2/§9).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from eplforecast.models.bayes_goals import (
    assert_def_sign_convention,
    build_model,
    dixon_coles_tau,
    fit_bayes,
    posterior_predictive_check,
    predict_outcome_probs,
    prior_predictive_check,
)

TEAMS = ["Arsenal", "Brentford", "Chelsea", "Derby"]
CUTOFF = pd.Timestamp("2026-06-01")


def _toy_df(dates=None):
    rows = [
        ("Arsenal", "Brentford", 2, 0),
        ("Chelsea", "Derby", 1, 1),
        ("Brentford", "Chelsea", 0, 3),
        ("Derby", "Arsenal", 0, 2),
    ]
    dates = dates or ["2026-05-01", "2026-05-08", "2026-05-15", "2026-05-22"]
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "home_team": h, "away_team": a,
             "home_goals": hg, "away_goals": ag}
            for d, (h, a, hg, ag) in zip(dates, rows, strict=True)
        ]
    )


# ---- Dixon-Coles tau (spec §3) ---------------------------------------------
def test_tau_matches_the_four_spec_cells():
    y_h = np.array([0, 0, 1, 1, 2])
    y_a = np.array([0, 1, 0, 1, 2])
    lam_h = np.full(5, 1.5)
    lam_a = np.full(5, 1.2)
    rho = -0.13
    tau = dixon_coles_tau(y_h, y_a, lam_h, lam_a, rho).eval()
    expected = np.array([
        1 - 1.5 * 1.2 * rho,  # 0-0
        1 + 1.5 * rho,        # 0-1
        1 + 1.2 * rho,        # 1-0
        1 - rho,              # 1-1
        1.0,                  # every other scoreline is untouched
    ])
    np.testing.assert_allclose(tau, expected, rtol=1e-9)


def test_tau_is_clipped_to_a_small_positive_floor():
    # tau(0,0) = 1 - lam_h*lam_a*rho goes negative for extreme rho/lambda;
    # log(tau) must stay defined (spec §3: clip at a small positive floor).
    tau = dixon_coles_tau(
        np.array([0]), np.array([0]), np.array([4.0]), np.array([4.0]), 0.9
    ).eval()
    assert tau[0] == pytest.approx(1e-6)


# ---- build_model (spec §5) --------------------------------------------------
def test_build_model_rejects_matches_at_or_after_cutoff():
    df = _toy_df(dates=["2026-05-01", "2026-05-08", "2026-05-15", "2026-06-01"])
    with pytest.raises(AssertionError, match="leakage"):
        build_model(df, TEAMS, CUTOFF)


def test_att_and_def_are_zero_sum():
    # ZeroSumNormal is the identifiability constraint (spec §1); a plain Normal
    # with soft centering would fail this — its draws don't sum to zero.
    model = build_model(_toy_df(), TEAMS, CUTOFF)
    att_draws = pm.draw(model["att"], draws=50, random_seed=1)
    def_draws = pm.draw(model["def"], draws=50, random_seed=2)
    np.testing.assert_allclose(att_draws.sum(axis=1), 0.0, atol=1e-8)
    np.testing.assert_allclose(def_draws.sum(axis=1), 0.0, atol=1e-8)


def test_free_parameters_are_exactly_the_spec_set():
    # half_life/xi must NOT appear: the decay rate is a function argument,
    # never a sampled parameter (spec §4).
    model = build_model(_toy_df(), TEAMS, CUTOFF)
    names = {rv.name for rv in model.free_RVs}
    assert names == {"mu", "home_adv", "sigma_att", "sigma_def", "att", "def", "rho"}


def test_promoted_penalty_present_only_with_mask_and_uses_spec_prior():
    mask = np.array([0.0, 0.0, 0.0, 1.0])  # Derby just came up
    model = build_model(_toy_df(), TEAMS, CUTOFF, promoted_mask=mask)
    assert "promoted_penalty" in {rv.name for rv in model.free_RVs}
    draws = pm.draw(model["promoted_penalty"], draws=4000, random_seed=3)
    assert np.mean(draws) == pytest.approx(-0.20, abs=0.02)  # spec §6 prior
    assert np.std(draws) == pytest.approx(0.15, abs=0.02)


def test_time_decay_downweights_ancient_matches():
    recent = _toy_df()
    ancient_date = CUTOFF - pd.Timedelta(days=900)  # 10 half-lives at hl=90
    with_ancient = pd.concat(
        [recent, recent.iloc[[0]].assign(date=ancient_date)], ignore_index=True
    )

    def logp_at_initial(model):
        return float(model.compile_logp()(model.initial_point()))

    # At a 90-day half-life the 900-day-old duplicate carries weight 2^-10:
    # adding it must leave the log-density essentially unchanged.
    short = logp_at_initial(build_model(recent, TEAMS, CUTOFF, half_life_days=90))
    short_plus = logp_at_initial(build_model(with_ancient, TEAMS, CUTOFF, half_life_days=90))
    assert abs(short_plus - short) < 0.02

    # With an effectively infinite half-life the same match counts fully.
    flat = logp_at_initial(build_model(recent, TEAMS, CUTOFF, half_life_days=1e9))
    flat_plus = logp_at_initial(build_model(with_ancient, TEAMS, CUTOFF, half_life_days=1e9))
    assert abs(flat_plus - flat) > 0.5


# ---- prior predictive check (spec §2) ---------------------------------------
def test_prior_predictive_reports_sane_goals_per_match():
    report = prior_predictive_check(_toy_df(), TEAMS, CUTOFF, draws=200, random_seed=42)
    assert report["n_draws"] == 200
    # Spec §2: total goals per match should sit in roughly 2-4 under the prior;
    # 12-goal games would mean sigma_att is too wide.
    assert 1.5 < report["mean_goals_per_match"] < 4.5
    # 6-0-ish scorelines rare but not impossible.
    assert 0.0 < report["p_six_plus"] < 0.2


# ---- def sign convention (spec §1 / §9) -------------------------------------
def _fake_idata(att, def_, mu=0.25, home_adv=0.20, rho=-0.10, teams=TEAMS, n_draws=8):
    """Hand-built posterior with constant draws — no sampling needed."""
    att = np.asarray(att, dtype=float)
    def_ = np.asarray(def_, dtype=float)
    posterior = xr.Dataset(
        {
            "mu": (("chain", "draw"), np.full((1, n_draws), mu)),
            "home_adv": (("chain", "draw"), np.full((1, n_draws), home_adv)),
            "rho": (("chain", "draw"), np.full((1, n_draws), rho)),
            "att": (("chain", "draw", "team"), np.tile(att, (1, n_draws, 1))),
            "def": (("chain", "draw", "team"), np.tile(def_, (1, n_draws, 1))),
        },
        coords={"chain": [0], "draw": np.arange(n_draws), "team": list(teams)},
    )
    # ArviZ >= 1.0: InferenceData IS an xarray DataTree.
    return xr.DataTree.from_dict({"posterior": posterior})


def _table_df():
    """Goals conceded per match strictly orders the teams:
    Arsenal 0.0 < Brentford 1.5 < Chelsea 2.0 < Derby 2.5."""
    rows = [
        ("Arsenal", "Derby", 1, 0),
        ("Derby", "Arsenal", 0, 4),
        ("Brentford", "Chelsea", 2, 1),
        ("Chelsea", "Brentford", 2, 2),
    ]
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-05-01"), "home_team": h, "away_team": a,
             "home_goals": hg, "away_goals": ag}
            for h, a, hg, ag in rows
        ]
    )


def test_sign_convention_passes_when_best_defence_has_highest_def():
    # def decreasing while conceded-per-match increases -> convention holds.
    idata = _fake_idata(att=[0, 0, 0, 0], def_=[1.0, 0.3, -0.3, -1.0])
    corr = assert_def_sign_convention(idata, _table_df())
    assert corr < 0


def test_sign_convention_raises_when_table_is_inverted():
    # A sign flip somewhere in the linear predictor would look exactly like
    # this: leaky defences carrying the highest def. Must blow up loudly.
    idata = _fake_idata(att=[0, 0, 0, 0], def_=[-1.0, -0.3, 0.3, 1.0])
    with pytest.raises(AssertionError, match="sign convention"):
        assert_def_sign_convention(idata, _table_df())


# ---- posterior predictive check (spec §9) -----------------------------------
def _dc_sample_scoreline(rng, lam_h, lam_a, rho, max_goals=10):
    """Independent reference sampler for the DC-corrected scoreline pmf."""
    from scipy.stats import poisson

    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    m[0, 0] *= 1 - lam_h * lam_a * rho
    m[0, 1] *= 1 + lam_h * rho
    m[1, 0] *= 1 + lam_a * rho
    m[1, 1] *= 1 - rho
    m /= m.sum()
    flat = rng.choice((max_goals + 1) ** 2, p=m.ravel())
    return divmod(flat, max_goals + 1)


def test_ppc_matches_scoreline_frequencies_of_model_generated_data():
    # Data drawn FROM the DC process with known parameters, checked against a
    # posterior fixed at those same parameters: simulated and observed
    # frequencies of the four corrected cells must agree.
    rng = np.random.default_rng(11)
    att = np.array([0.25, 0.05, -0.1, -0.2])
    def_ = np.array([0.2, 0.1, -0.1, -0.2])
    mu, gamma, rho = 0.25, 0.20, -0.12
    rows = []
    for _ in range(50):  # 50 double round robins -> 600 matches
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                lam_h = np.exp(mu + gamma + att[i] - def_[j])
                lam_a = np.exp(mu + att[j] - def_[i])
                hg, ag = _dc_sample_scoreline(rng, lam_h, lam_a, rho)
                rows.append({
                    "date": pd.Timestamp("2026-05-01"),
                    "home_team": TEAMS[i], "away_team": TEAMS[j],
                    "home_goals": hg, "away_goals": ag,
                })
    df = pd.DataFrame(rows)

    idata = _fake_idata(att=att, def_=def_, mu=mu, home_adv=gamma, rho=rho)
    ppc = posterior_predictive_check(idata, df, TEAMS)

    assert list(ppc.index) == ["0-0", "1-0", "1-1", "2-1"]
    assert set(ppc.columns) == {"simulated", "observed"}
    assert (ppc["simulated"] - ppc["observed"]).abs().max() < 0.03


# ---- predict_outcome_probs --------------------------------------------------
def test_predict_outcome_probs_matches_the_direct_score_matrix():
    # Constant posterior draws -> the draw-average must equal a single direct
    # score_matrix/outcome_probs evaluation at those parameters. Fixtures have
    # no goals columns: this is the pre-kickoff path.
    from eplforecast.simulate.scorelines import outcome_probs, score_matrix

    att = np.array([0.2, 0.0, -0.1, -0.1])
    def_ = np.array([0.1, 0.0, -0.05, -0.05])
    mu, gamma, rho = 0.25, 0.20, -0.12
    idata = _fake_idata(att=att, def_=def_, mu=mu, home_adv=gamma, rho=rho)

    pairs = [(0, 1), (2, 3), (3, 0)]
    fixtures = pd.DataFrame(
        [{"home_team": TEAMS[i], "away_team": TEAMS[j]} for i, j in pairs]
    )
    probs = predict_outcome_probs(idata, fixtures, TEAMS)

    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    for row, (i, j) in enumerate(pairs):
        lam_h = np.exp(mu + gamma + att[i] - def_[j])
        lam_a = np.exp(mu + att[j] - def_[i])
        expected = outcome_probs(score_matrix(lam_h, lam_a, rho))
        np.testing.assert_allclose(probs[row], expected, atol=1e-9)


# ---- fit_bayes end to end (guide §4.3) --------------------------------------
@pytest.fixture(scope="module")
def fitted_league():
    """One short NUTS fit on a synthetic 6-team league, shared by the
    integration tests. Ground truth has a strong defensive spread so the
    sign-convention check has real signal to find."""
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(6)]
    att = np.array([0.2, 0.1, 0.0, 0.0, -0.1, -0.2])
    def_ = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
    mu, gamma = 0.25, 0.20
    rows = []
    date = CUTOFF - pd.Timedelta(days=200)
    for _ in range(4):  # 4 double round robins = 120 matches
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                lam_h = np.exp(mu + gamma + att[i] - def_[j])
                lam_a = np.exp(mu + att[j] - def_[i])
                rows.append({
                    "date": date,
                    "home_team": teams[i], "away_team": teams[j],
                    "home_goals": int(rng.poisson(lam_h)),
                    "away_goals": int(rng.poisson(lam_a)),
                })
                date += pd.Timedelta(hours=6)
    df = pd.DataFrame(rows)
    idata = fit_bayes(
        df, teams, CUTOFF, half_life_days=365,
        draws=250, tune=250, chains=2, cores=1,
        random_seed=42, progressbar=False,
    )
    return df, teams, idata


def test_fit_bayes_returns_posterior_with_the_spec_parameters(fitted_league):
    _, teams, idata = fitted_league
    post = idata.posterior
    assert {"mu", "home_adv", "sigma_att", "sigma_def", "att", "def", "rho"} <= set(
        post.data_vars
    )
    assert list(post["att"].coords["team"].to_numpy()) == teams
    assert post["def"].sizes["team"] == len(teams)


def test_fitted_model_satisfies_the_def_sign_convention(fitted_league):
    # Checkpoint from spec §9: best defence carries the highest def. With a
    # 1.0-wide true def spread over 120 matches this must be unambiguous.
    df, _, idata = fitted_league
    corr = assert_def_sign_convention(idata, df)
    assert corr < -0.5


def test_fitted_ppc_tracks_observed_low_scoreline_frequencies(fitted_league):
    df, teams, idata = fitted_league
    ppc = posterior_predictive_check(idata, df, teams)
    assert np.isfinite(ppc.to_numpy()).all()
    assert (ppc["simulated"] - ppc["observed"]).abs().max() < 0.08
