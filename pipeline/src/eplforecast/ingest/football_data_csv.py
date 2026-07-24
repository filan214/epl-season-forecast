"""football-data.co.uk historical results + closing odds -> parquet.

`load_historical` returns one canonical frame (date, season, home/away team,
goals, result, and the best-available CLOSING odds as odds_home/draw/away) and
caches each season to parquet before returning, so a run is repeatable offline
(§4.1). Resolution order per season: parquet cache -> local raw CSV -> download.

The odds columns are the market benchmark: average CLOSING odds are preferred
(PRD §8, §10). This module never touches Neon — training data stays on disk (§0).
"""

from __future__ import annotations

import io
import socket
import ssl
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RAW_DIR

FD_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# DNS-over-HTTPS endpoint used to route around ISP DNS hijacking of the source
# (e.g. Indonesia's Kominfo Trust+ Positif rewrites football-data.co.uk, since
# it carries betting odds). The DoH query itself travels inside HTTPS, so it
# can't be spoofed on port 53 the way plain DNS is.
DOH_ENDPOINT = "https://cloudflare-dns.com/dns-query"

# Priority for the benchmark odds triple: closing before pre-close, market
# average before a single book.
_ODDS_PRIORITY = [
    ("AvgCH", "AvgCD", "AvgCA"),    # average closing across bookmakers (best)
    ("B365CH", "B365CD", "B365CA"),  # Bet365 closing
    ("MaxCH", "MaxCD", "MaxCA"),     # market max closing
    ("AvgH", "AvgD", "AvgA"),        # pre-close average (fallback)
    ("B365H", "B365D", "B365A"),     # pre-close Bet365 (fallback)
    ("BbAvH", "BbAvD", "BbAvA"),     # legacy average (pre-2019 files)
]


def season_to_code(season: str) -> str:
    """'2023-24' -> '2324' (football-data.co.uk mmz4281 season code)."""
    parts = season.split("-")
    if len(parts) != 2 or not parts[0][-2:].isdigit() or not parts[1][-2:].isdigit():
        raise ValueError(f"season must look like '2023-24'; got {season!r}")
    return parts[0][-2:] + parts[1][-2:]


def _select_odds(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    for home, draw, away in _ODDS_PRIORITY:
        if {home, draw, away} <= set(df.columns):
            return df[home], df[draw], df[away]
    nan = pd.Series(np.nan, index=df.index)
    return nan, nan, nan


def _normalize_fd_frame(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalise a football-data.co.uk E0 frame to the canonical schema."""
    df = raw[raw["FTHG"].notna() & raw["FTAG"].notna()].reset_index(drop=True)
    odds_h, odds_d, odds_a = _select_odds(df)

    if "FTR" in df.columns:
        result = df["FTR"].astype(str)
    else:
        hg, ag = df["FTHG"], df["FTAG"]
        result = pd.Series(np.where(hg > ag, "H", np.where(hg < ag, "A", "D")))

    return pd.DataFrame(
        {
            "season": season,
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home_team": df["HomeTeam"].astype(str),
            "away_team": df["AwayTeam"].astype(str),
            "home_goals": df["FTHG"].astype(int),
            "away_goals": df["FTAG"].astype(int),
            "result": result,
            "odds_home": pd.to_numeric(odds_h, errors="coerce"),
            "odds_draw": pd.to_numeric(odds_d, errors="coerce"),
            "odds_away": pd.to_numeric(odds_a, errors="coerce"),
        }
    )


def _read_fd_csv(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(data), encoding="latin-1", on_bad_lines="skip")


def _is_fd_csv(data: bytes) -> bool:
    """True if `data` looks like a football-data.co.uk E0 file (header 'Div,...').

    A DNS hijack that redirects to an HTTP block page returns 200 + HTML, not a
    CSV; this sniff catches that so we can fall back rather than cache garbage.
    """
    head = data[:64].lstrip(b"\xef\xbb\xbf").lstrip()  # tolerate BOM/whitespace
    return head.startswith(b"Div,")


def _http_get_bytes(url: str, timeout: float = 30.0) -> bytes:
    """Plain verified GET. Raises httpx.TransportError on a hijacked/intercepted
    connection (cert hostname mismatch, refused port, reset)."""
    import httpx  # local import so offline use never needs the dependency loaded

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _first_a_record(payload: dict, host: str) -> str:
    """Pull the first A record (type 1) out of a Cloudflare DoH JSON response."""
    for answer in payload.get("Answer", []):
        if answer.get("type") == 1:
            return answer["data"]
    raise RuntimeError(f"DoH returned no A record for {host!r}")


def _doh_resolve(host: str, timeout: float = 30.0) -> str:
    """Resolve `host` to an IPv4 address via DNS-over-HTTPS (bypasses port-53
    DNS hijacking)."""
    import httpx

    resp = httpx.get(
        DOH_ENDPOINT,
        params={"name": host, "type": "A"},
        headers={"accept": "application/dns-json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return _first_a_record(resp.json(), host)


@contextmanager
def _pin_getaddrinfo(host: str, ip: str):
    """Force resolution of `host` to `ip` for the duration, leaving every other
    hostname untouched. SNI, the Host header, and cert verification still use
    the real hostname, so TLS validates normally against CN=*.football-data.co.uk.
    """
    real_getaddrinfo = socket.getaddrinfo

    def _patched(h, *args, **kwargs):
        target = ip if h == host else h
        return real_getaddrinfo(target, *args, **kwargs)

    socket.getaddrinfo = _patched
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


def _pinned_get_bytes(url: str, host: str, ip: str, timeout: float = 30.0) -> bytes:
    """Verified GET of `url` with `host` pinned to `ip` (DoH-resolved)."""
    with _pin_getaddrinfo(host, ip):
        return _http_get_bytes(url, timeout=timeout)


def _download_fd_csv(code: str, timeout: float = 30.0) -> bytes:
    """Download one season's E0.csv, falling back to DoH + IP pin when the
    direct fetch is DNS-hijacked or TLS-intercepted by the local ISP.

    The fallback keeps TLS verification fully on: only name resolution is routed
    around the block, so the bytes are the genuine football-data.co.uk file.
    """
    import httpx

    url = f"{FD_BASE_URL}/{code}/E0.csv"
    host = urlsplit(url).hostname

    try:
        data = _http_get_bytes(url, timeout=timeout)
        if _is_fd_csv(data):
            return data
        reason = "the direct response was not a CSV (looks like an ISP block page)"
    except (httpx.TransportError, ssl.SSLError) as exc:
        # Cert hostname mismatch / refused port / reset — the ISP-hijack signature.
        reason = f"the direct fetch failed ({type(exc).__name__}: {exc})"

    ip = _doh_resolve(host, timeout=timeout)
    data = _pinned_get_bytes(url, host, ip, timeout=timeout)
    if not _is_fd_csv(data):
        raise RuntimeError(
            f"football-data.co.uk unreachable for {code}: {reason}; the DoH "
            f"fallback via {ip} also did not return a valid CSV."
        )
    return data


def load_historical(
    seasons: list[str],
    cache_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load and normalise EPL seasons, caching each to parquet before returning.

    Per season the source is resolved as: parquet cache -> local raw CSV at
    ``{raw_dir}/football_data/{code}_E0.csv`` -> download from football-data.co.uk.
    """
    cache_dir = Path(cache_dir) if cache_dir else PROCESSED_DIR
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in seasons:
        code = season_to_code(season)
        parquet = cache_dir / f"fd_{code}.parquet"
        if parquet.exists() and not refresh:
            frames.append(pd.read_parquet(parquet))
            continue

        local = raw_dir / "football_data" / f"{code}_E0.csv"
        data = local.read_bytes() if local.exists() else _download_fd_csv(code)
        normalized = _normalize_fd_frame(_read_fd_csv(data), season)
        normalized.to_parquet(parquet, index=False)  # cache before returning (§4.1)
        frames.append(normalized)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
