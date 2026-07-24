"""Convergence gate (guide §4.6, spec §9): decide ok vs gated BEFORE any write.

Fail the cron job loudly rather than serve numbers from a bad fit. A gated run
writes a model_runs row with status='gated' and the reasons — then writes
nothing else; the dashboard falls back to the last status='ok' run.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np

# Thresholds per spec §9 / guide §4.6.
RHAT_MAX = 1.01
ESS_MIN = 400
BFMI_MIN = 0.3


@dataclass(frozen=True)
class GateResult:
    status: str  # "ok" | "gated" — mirrors the model_runs status enum
    reasons: list[str]
    stats: dict[str, float]  # r_hat_max / ess_*_min / divergences / bfmi_min

    @property
    def passed(self) -> bool:
        return self.status == "ok"


def _extreme(tree, reduce_fn) -> float:
    """Reduce every variable of a diagnostic result to one scalar.

    ArviZ 1.x returns diagnostics as a DataTree keyed by parameter; NaNs are
    deliberately propagated so an undefined diagnostic gates the run.
    """
    ds = tree.to_dataset() if hasattr(tree, "to_dataset") else tree
    values = np.concatenate([np.ravel(ds[name].to_numpy()) for name in ds.data_vars])
    return float(reduce_fn(values))


def check_convergence(idata) -> GateResult:
    """Spec §9 checklist over a fitted trace. All comparisons are written so
    that a NaN diagnostic fails the check rather than slipping through."""
    stats = {
        "r_hat_max": _extreme(az.rhat(idata), np.max),
        "ess_bulk_min": _extreme(az.ess(idata, method="bulk"), np.min),
        "ess_tail_min": _extreme(az.ess(idata, method="tail"), np.min),
        "divergences": float(idata.sample_stats["diverging"].sum()),
        "bfmi_min": float(np.min(az.bfmi(idata)["energy"].to_numpy())),
    }

    reasons = []
    if not (stats["r_hat_max"] < RHAT_MAX):
        reasons.append(f"R-hat {stats['r_hat_max']:.4f} >= {RHAT_MAX} on the worst parameter")
    if not (stats["ess_bulk_min"] > ESS_MIN):
        reasons.append(f"bulk ESS {stats['ess_bulk_min']:.0f} <= {ESS_MIN}")
    if not (stats["ess_tail_min"] > ESS_MIN):
        reasons.append(f"tail ESS {stats['ess_tail_min']:.0f} <= {ESS_MIN}")
    if stats["divergences"] != 0:
        reasons.append(f"{int(stats['divergences'])} divergences (must be zero)")
    if not (stats["bfmi_min"] > BFMI_MIN):
        reasons.append(f"BFMI {stats['bfmi_min']:.3f} < {BFMI_MIN} (energy pathology)")

    return GateResult(status="gated" if reasons else "ok", reasons=reasons, stats=stats)
