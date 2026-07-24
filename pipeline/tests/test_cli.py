"""Tests for the argument parser wiring in cli.py."""

from eplforecast.cli import build_parser


def test_validate_parses_seasons_and_options():
    ns = build_parser().parse_args(
        ["validate", "--seasons", "2015-16", "2016-17", "--min-train-seasons", "4"]
    )
    assert ns.command == "validate"
    assert ns.seasons == ["2015-16", "2016-17"]
    assert ns.min_train_seasons == 4


def test_validate_defaults():
    ns = build_parser().parse_args(["validate", "--seasons", "2015-16"])
    assert ns.min_train_seasons == 3
    assert ns.window == 5
    assert ns.refresh is False
