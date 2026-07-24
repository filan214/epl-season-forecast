"""Tests for db/gate.py — the convergence gate (guide §4.6, spec §9).

The gate's job: fail the weekly run loudly rather than publish numbers from a
bad fit. Thresholds: R-hat < 1.01 on every parameter, bulk AND tail ESS > 400,
zero divergences, no BFMI pathology. A gated run writes a model_runs row with
status='gated' and the reasons — then writes nothing else.

Each test manufactures chains with exactly one known pathology and demands the
gate names it.
"""

import numpy as np
import xarray as xr

from eplforecast.db.gate import check_convergence


def _idata(mu=None, diverging=None, energy=None, n_draws=400, seed=0):
    """Two-chain idata; healthy by default (iid draws, no divergences)."""
    rng = np.random.default_rng(seed)
    if mu is None:
        mu = rng.normal(0.0, 1.0, (2, n_draws))
    n_draws = mu.shape[1]
    if diverging is None:
        diverging = np.zeros((2, n_draws), dtype=bool)
    if energy is None:
        energy = rng.normal(0.0, 1.0, (2, n_draws))
    posterior = xr.Dataset(
        {
            "mu": (("chain", "draw"), mu),
            "att": (("chain", "draw", "team"), rng.normal(0.0, 1.0, (2, n_draws, 3))),
        },
        coords={"chain": [0, 1], "draw": np.arange(n_draws), "team": ["a", "b", "c"]},
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), diverging),
            "energy": (("chain", "draw"), energy),
        },
        coords={"chain": [0, 1], "draw": np.arange(n_draws)},
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def test_healthy_chains_pass_with_no_reasons():
    result = check_convergence(_idata())
    assert result.status == "ok"
    assert result.passed
    assert result.reasons == []
    assert result.stats["r_hat_max"] < 1.01
    assert result.stats["ess_bulk_min"] > 400
    assert result.stats["ess_tail_min"] > 400
    assert result.stats["divergences"] == 0
    assert result.stats["bfmi_min"] > 0.3


def test_chains_that_disagree_gate_on_rhat():
    rng = np.random.default_rng(1)
    mu = np.stack([rng.normal(0.0, 1.0, 400), rng.normal(3.0, 1.0, 400)])
    result = check_convergence(_idata(mu=mu))
    assert result.status == "gated"
    assert not result.passed
    assert any("R-hat" in r for r in result.reasons)
    assert result.stats["r_hat_max"] > 1.01


def test_sticky_autocorrelated_chains_gate_on_ess():
    rng = np.random.default_rng(2)
    mu = np.empty((2, 400))
    for c in range(2):
        e = rng.normal(0.0, 1.0, 400)
        x = np.empty(400)
        x[0] = e[0]
        for t in range(1, 400):  # AR(1), phi = 0.95 -> ESS far below 400
            x[t] = 0.95 * x[t - 1] + np.sqrt(1 - 0.95**2) * e[t]
        mu[c] = x
    result = check_convergence(_idata(mu=mu))
    assert result.status == "gated"
    assert any("ESS" in r for r in result.reasons)
    assert result.stats["ess_bulk_min"] < 400


def test_a_single_divergence_gates():
    diverging = np.zeros((2, 400), dtype=bool)
    diverging[1, 37] = True
    result = check_convergence(_idata(diverging=diverging))
    assert result.status == "gated"
    assert any("divergen" in r.lower() for r in result.reasons)
    assert result.stats["divergences"] == 1


def test_pathological_energy_gates_on_bfmi():
    rng = np.random.default_rng(3)
    # Energy that drifts as a slow random walk: tiny transition variance
    # relative to marginal variance -> E-BFMI near zero.
    energy = np.cumsum(rng.normal(0.0, 0.1, (2, 400)), axis=1)
    result = check_convergence(_idata(energy=energy))
    assert result.status == "gated"
    assert any("BFMI" in r for r in result.reasons)
    assert result.stats["bfmi_min"] < 0.3


def test_multiple_pathologies_are_all_reported():
    rng = np.random.default_rng(4)
    mu = np.stack([rng.normal(0.0, 1.0, 400), rng.normal(3.0, 1.0, 400)])
    diverging = np.zeros((2, 400), dtype=bool)
    diverging[0, :5] = True
    result = check_convergence(_idata(mu=mu, diverging=diverging))
    assert result.status == "gated"
    assert len(result.reasons) >= 2
