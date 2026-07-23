# EPL Season Forecast (2026-27)

A live-updating Bayesian forecasting system for the 2026-27 English Premier
League season. Every gameweek it re-fits a hierarchical Poisson goal model,
re-simulates the full league table from the posterior, publishes title /
top-four / relegation probabilities, and tracks its own calibration against
bookmaker closing odds across the whole season.

> **Status: Phase 0 — scaffolding.** The repo structure, database schema, and
> the leakage guard are in place. The ingest, model, simulation, dashboard, and
> automation code land in later phases (see `IMPLEMENTATION_GUIDE.md` §7).

- **Product spec:** [`PRD.md`](./PRD.md)
- **Build spec:** [`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md)
- **Model math:** `PYMC_MODEL_SPEC.md` (companion doc)

## Architecture

Two halves, one repo, meeting only at Neon Postgres:

- **`pipeline/`** — a Python package that ingests data, fits the model, and
  simulates the season **offline** (GitHub Actions), then writes results to Neon.
- **`web/`** — a Next.js dashboard that **only ever reads** from Neon. It
  computes nothing; if a number can't be read from a table, the pipeline should
  have written it.

Drizzle (in `web/`) is the single schema authority. Python writes through raw
parameterised SQL against the tables Drizzle creates — no ORM on the Python side.

```
epl-season-forecast/
  pipeline/                 # Python 3.11+ modelling + simulation (offline)
    pyproject.toml
    src/eplforecast/        # config, ingest/, features/, models/, simulate/, evaluate/, db/, cli
    data/                   # raw/ (gitignored) + processed/ parquet
    tests/
  web/                      # Next.js 15 dashboard (read-only)
    app/                    # routes: /, /calibration, /team/[slug], /methodology
    src/db/                 # Drizzle schema authority + Neon client
    drizzle/                # SQL migrations (incl. the leakage-guard trigger)
  .github/workflows/        # weekly_forecast.yml, heartbeat.yml
```

## The leakage guard

The project's core integrity claim — that no prediction ever saw information
from at/after kickoff — is enforced in the database, not just in code. A
`BEFORE INSERT` trigger on `predictions` rejects any row whose `created_at` is
not strictly before the referenced match's `kickoff_utc`
(`web/drizzle/0001_leakage_guard.sql`). Verify it with
`web/scripts/verify_leakage_guard.sql`.

## Setup (Windows / PowerShell)

### Web (dashboard + schema)

```powershell
cd web
npm install
Copy-Item .env.example .env.local   # then set DATABASE_URL to your Neon string
npm run db:generate                 # regenerate migrations after a schema change
npm run db:migrate                  # apply migrations to Neon
npm run dev                         # http://localhost:3000
```

### Pipeline (Python)

```powershell
cd pipeline
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Cost

Runs at **$0/month** on free tiers: GitHub Actions (compute), Neon (Postgres),
Vercel (hosting). No always-on server anywhere.
