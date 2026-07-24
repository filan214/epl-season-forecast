"""Tests for ingest/football_data_csv.py (offline paths only — no network).

The network download is a thin wrapper; here we test the season-code mapping, the
CSV -> canonical-schema normalisation, and loading from a local CSV with parquet
caching (the offline fallback used when a download isn't available).

The DoH-fallback tests cover the download orchestration without touching the
network: when the direct fetch is DNS-hijacked/TLS-intercepted (e.g. Indonesia's
Kominfo Trust+ Positif rewriting football-data.co.uk), the downloader resolves
the real IP out of band via DNS-over-HTTPS and pins it, verification still on.
"""

import socket

import httpx
import pandas as pd
import pytest

from eplforecast.ingest import football_data_csv as fd
from eplforecast.ingest.football_data_csv import (
    _download_fd_csv,
    _first_a_record,
    _normalize_fd_frame,
    _pin_getaddrinfo,
    load_historical,
    season_to_code,
)


def test_season_to_code():
    assert season_to_code("2023-24") == "2324"
    assert season_to_code("2009-10") == "0910"
    assert season_to_code("1999-00") == "9900"


def _raw_fd_frame():
    return pd.DataFrame({
        "Div": ["E0", "E0"],
        "Date": ["10/08/2023", "17/08/2023"],
        "HomeTeam": ["Arsenal", "Chelsea"],
        "AwayTeam": ["Forest", "Liverpool"],
        "FTHG": [2, 1],
        "FTAG": [1, 1],
        "FTR": ["H", "D"],
        "AvgCH": [1.50, 2.50],
        "AvgCD": [4.20, 3.40],
        "AvgCA": [6.00, 2.80],
    })


def test_normalize_maps_to_canonical_schema():
    out = _normalize_fd_frame(_raw_fd_frame(), "2023-24")
    for c in ("season", "date", "home_team", "away_team", "home_goals",
              "away_goals", "odds_home", "odds_draw", "odds_away"):
        assert c in out.columns
    assert out["home_team"].tolist() == ["Arsenal", "Chelsea"]
    assert out["home_goals"].tolist() == [2, 1]
    assert out["season"].tolist() == ["2023-24", "2023-24"]
    assert out["date"].iloc[0] == pd.Timestamp("2023-08-10")  # dayfirst


def test_normalize_prefers_average_closing_odds():
    out = _normalize_fd_frame(_raw_fd_frame(), "2023-24")
    assert out["odds_home"].tolist() == [1.50, 2.50]
    assert out["odds_away"].tolist() == [6.00, 2.80]


def test_normalize_falls_back_to_bet365_when_no_average_closing():
    raw = _raw_fd_frame().drop(columns=["AvgCH", "AvgCD", "AvgCA"])
    raw["B365H"], raw["B365D"], raw["B365A"] = [1.6, 2.4], [4.0, 3.3], [5.5, 2.9]
    out = _normalize_fd_frame(raw, "2023-24")
    assert out["odds_home"].tolist() == [1.6, 2.4]


def test_normalize_drops_unplayed_rows():
    raw = _raw_fd_frame()
    raw.loc[1, "FTHG"] = None  # a scheduled-but-unplayed fixture
    out = _normalize_fd_frame(raw, "2023-24")
    assert len(out) == 1


def test_load_historical_reads_local_csv_and_caches_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "football_data").mkdir(parents=True)
    cache_dir = tmp_path / "processed"
    _raw_fd_frame().to_csv(raw_dir / "football_data" / "2324_E0.csv", index=False)

    df = load_historical(["2023-24"], raw_dir=raw_dir, cache_dir=cache_dir)

    assert len(df) == 2
    assert df["season"].unique().tolist() == ["2023-24"]
    assert (cache_dir / "fd_2324.parquet").exists()  # cached before returning


# ---- DoH fallback (ISP DNS hijack / TLS interception) ----------------------
def test_first_a_record_picks_the_ipv4_answer():
    payload = {"Answer": [
        {"name": "www.football-data.co.uk", "type": 5, "data": "cdn.example."},  # CNAME
        {"name": "www.football-data.co.uk", "type": 1, "data": "217.160.0.246"},  # A
    ]}
    assert _first_a_record(payload, "www.football-data.co.uk") == "217.160.0.246"


def test_first_a_record_raises_when_no_ipv4_answer():
    payload = {"Answer": [{"name": "x", "type": 28, "data": "::1"}]}  # AAAA only
    with pytest.raises(RuntimeError, match="no A record"):
        _first_a_record(payload, "x")


def test_pin_getaddrinfo_overrides_only_the_target_host_and_restores():
    original = socket.getaddrinfo
    host, ip = "www.football-data.co.uk", "217.160.0.246"
    with _pin_getaddrinfo(host, ip):
        # A literal IP resolves without network, so this is offline-safe.
        pinned = socket.getaddrinfo(host, 443)
        assert any(info[4][0] == ip for info in pinned)
    assert socket.getaddrinfo is original  # restored on exit


def test_download_uses_direct_fetch_and_skips_doh_when_reachable(monkeypatch):
    monkeypatch.setattr(fd, "_http_get_bytes", lambda url, timeout=30.0: b"Div,Date,ok\n1,2")

    def _no_doh(*a, **k):
        raise AssertionError("DoH must not run when the direct fetch works")

    monkeypatch.setattr(fd, "_doh_resolve", _no_doh)
    assert _download_fd_csv("2324").startswith(b"Div,")


def test_download_falls_back_to_doh_on_transport_error(monkeypatch):
    seen = {}

    def _hijacked(url, timeout=30.0):
        raise httpx.ConnectError("certificate verify failed: Hostname mismatch")

    def _pinned(url, host, ip, timeout=30.0):
        seen.update(url=url, host=host, ip=ip)
        return b"Div,Date,via-doh\n1,2"

    monkeypatch.setattr(fd, "_http_get_bytes", _hijacked)
    monkeypatch.setattr(fd, "_doh_resolve", lambda host, timeout=30.0: "217.160.0.246")
    monkeypatch.setattr(fd, "_pinned_get_bytes", _pinned)

    out = _download_fd_csv("2324")
    assert out.startswith(b"Div,")
    assert seen["host"] == "www.football-data.co.uk"
    assert seen["ip"] == "217.160.0.246"
    assert seen["url"].endswith("/2324/E0.csv")


def test_download_falls_back_when_direct_returns_a_block_page(monkeypatch):
    # DNS hijack to an HTTP block server returns 200 + HTML, not a CSV.
    monkeypatch.setattr(fd, "_http_get_bytes", lambda url, timeout=30.0: b"<html>Diblokir</html>")
    monkeypatch.setattr(fd, "_doh_resolve", lambda host, timeout=30.0: "217.160.0.246")
    monkeypatch.setattr(
        fd, "_pinned_get_bytes", lambda url, host, ip, timeout=30.0: b"Div,Date,real\n1,2"
    )
    assert _download_fd_csv("2324").startswith(b"Div,")


def test_download_raises_when_doh_fallback_also_returns_junk(monkeypatch):
    monkeypatch.setattr(fd, "_http_get_bytes", lambda url, timeout=30.0: b"<html>block</html>")
    monkeypatch.setattr(fd, "_doh_resolve", lambda host, timeout=30.0: "1.2.3.4")
    monkeypatch.setattr(
        fd, "_pinned_get_bytes", lambda url, host, ip, timeout=30.0: b"<html>still blocked</html>"
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        _download_fd_csv("2324")
