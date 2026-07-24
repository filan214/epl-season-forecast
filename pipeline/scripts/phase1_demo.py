"""Phase 1 demo: baseline RPS alongside market RPS on a SYNTHETIC season set.

Real held-out numbers need football-data.co.uk (blocked by TLS interception on
this machine). This script instead generates a synthetic multi-season dataset —
Poisson goals from latent team strengths + home advantage, with a near-oracle
'market' priced from the true generative probabilities plus a 5% vig — and runs
the real expanding-window harness end to end, exactly as `eplforecast validate`
would on real data.

    python scripts/phase1_demo.py

For the real thing once you have network/CSVs:
    python -m eplforecast validate --seasons 2014-15 2015-16 ... 2024-25
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eplforecast.evaluate.validation import (  # noqa: E402
    print_validation_report,
    run_baseline_validation,
)


def _true_probs(lam_h: float, lam_a: float, max_goals: int = 10):
    gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
    grid = np.outer(gh, ga)
    p_home = np.tril(grid, -1).sum()
    p_draw = float(np.trace(grid))
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def synthetic_matches(n_seasons=10, n_teams=20, seed=7, vig=1.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"T{i:02d}" for i in range(n_teams)]
    strength = {t: rng.normal(0.0, 0.35) for t in names}
    home_field = 0.30
    rows = []
    for s in range(n_seasons):
        season = f"{2014 + s}-{str(2015 + s)[2:]}"
        date = pd.Timestamp("2014-08-01") + pd.DateOffset(years=s)
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                h, a = names[i], names[j]
                lam_h = np.exp(0.15 + home_field + strength[h] - strength[a])
                lam_a = np.exp(0.15 + strength[a] - strength[h])
                ph, pd_, pa = _true_probs(lam_h, lam_a)
                rows.append({
                    "season": season, "date": date, "home_team": h, "away_team": a,
                    "home_goals": int(rng.poisson(lam_h)),
                    "away_goals": int(rng.poisson(lam_a)),
                    "odds_home": 1.0 / (ph * vig),
                    "odds_draw": 1.0 / (pd_ * vig),
                    "odds_away": 1.0 / (pa * vig),
                })
                date += pd.Timedelta(hours=6)
    return pd.DataFrame(rows)


def main() -> None:
    matches = synthetic_matches()
    print(f"Synthetic dataset: {len(matches)} matches across "
          f"{matches['season'].nunique()} seasons.\n")
    results = run_baseline_validation(matches, min_train_seasons=3, window=6)
    print_validation_report(results)
    print("\n(Synthetic data. On real football-data.co.uk odds the gap is the "
          "honest claim; expect the model to slightly TRAIL the market - PRD s10/s13.)")


if __name__ == "__main__":
    main()
