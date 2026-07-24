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

    by_model = res.table.set_index("model")
    blend_rps = float(by_model.loc["blend", "rps"])
    bayes_rps = float(by_model.loc["bayes", "rps"])
    xgb_rps = float(by_model.loc["xgb", "rps"])
    best = res.table.sort_values("rps").iloc[0]["model"]
    beats_both = blend_rps < bayes_rps and blend_rps < xgb_rps
    print(f"\nlowest RPS: {best}. blend beats both components out-of-sample: {beats_both} "
          f"(blend {blend_rps:.4f} vs bayes {bayes_rps:.4f}, xgb {xgb_rps:.4f})")


if __name__ == "__main__":
    main()
