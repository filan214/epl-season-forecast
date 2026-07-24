"""Central paths, constants, and season configuration.

Deliberately free of ingest/model logic (IMPLEMENTATION_GUIDE.md §2). Filled out
further in later phases; for now it fixes the storage-tier paths and a couple of
season-level constants the rest of the package will import.
"""

from __future__ import annotations

from pathlib import Path

# --- Storage tiers (IMPLEMENTATION_GUIDE.md §0) ------------------------------
# Training data (decades of CSVs, parquet) lives on disk / in the repo and is
# NEVER imported to Postgres. Serving data (predictions, snapshots, metrics)
# lives in Neon. The posterior trace lives in neither — it is a CI artifact.
PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PIPELINE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # gitignored
PROCESSED_DIR = DATA_DIR / "processed"  # parquet, committed if small

# --- Season under forecast --------------------------------------------------
SEASON = "2026-27"

# --- Modelling defaults -----------------------------------------------------
# Exponential time-decay half-life (days). Selected offline by out-of-sample RPS
# in Phase 2 — never estimated inside the model (PRD §9, IMPLEMENTATION_GUIDE §10).
# Grid search over {60,90,120,180,270,365,inf} on seasons 2015-16..2024-25
# (7 expanding-window held-out seasons): 365d won at pooled RPS 0.2082, a shallow
# plateau with 270d (0.2092) and inf (0.2099) close behind. 2026-07-24.
DEFAULT_HALF_LIFE_DAYS = 365.0
