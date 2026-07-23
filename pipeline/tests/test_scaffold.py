"""Smoke test: the package imports and its config exposes the expected constants.

Real tests (leakage assertions, RPS, convergence gate) arrive with their phases.
"""

import eplforecast
from eplforecast import config


def test_version() -> None:
    assert eplforecast.__version__


def test_season_config() -> None:
    assert config.SEASON == "2026-27"
    assert config.DEFAULT_HALF_LIFE_DAYS > 0
