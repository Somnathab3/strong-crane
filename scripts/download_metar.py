# SPDX-License-Identifier: GPL-3.0-only
"""Fetch METAR surface observations for every airport the movement data touches.

The ten challenge airports are only one side of each flight. A departure has a
destination and an arrival has an origin, and conditions at that far end plausibly
reach back into taxi-out through flow restrictions and slot delays: a destination
under thunderstorms or low visibility gets a regulated arrival rate, the departure
receives a calculated take-off time, and the aircraft can end up holding on the
ground. Whether that shows up in *taxi* rather than in the pre-pushback wait is an
empirical question, and it cannot be asked without the observations.

Source: Iowa Environmental Mesonet ASOS archive, an open, documented, public-domain
service requiring no credentials. See the attribution table in README.md for the licence note.

The condition flags are computed to the same definitions as the pre-existing
station files, so old and new downloads share one schema.

Resumable: a station already present and non-trivial is skipped, so an interrupted
run can simply be restarted.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from taxiout.config import AIRPORT_TZ, paths  # noqa: E402

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

#: Fields requested, matching the schema of the existing station files.
FIELDS = ("tmpc", "dwpc", "relh", "drct", "sknt", "gust", "alti", "vsby",
          "wxcodes", "ice_accretion_1hr")

#: Be a considerate client of a free public archive.
MAX_WORKERS = 6
REQUEST_TIMEOUT = 120
RETRIES = 3
BACKOFF_SECONDS = 4

#: The archive answers 429 under load; wait long enough to actually clear it.
THROTTLE_BACKOFF_SECONDS = 30

#: Below this a file is treated as a failed or empty fetch and retried.
MINIMUM_USEFUL_BYTES = 4096


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_url(icao: str, start: str, end: str) -> str:
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    data = "&".join(f"data={field}" for field in FIELDS)
    return (
        f"{ENDPOINT}?station={icao}&{data}"
        f"&year1={int(y1)}&month1={int(m1)}&day1={int(d1)}"
        f"&year2={int(y2)}&month2={int(m2)}&day2={int(d2)}"
        "&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T"
        "&direct=no&report_type=3"
    )


def condition_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Operational flags, to the same definitions as the existing station files."""
    codes = frame["wxcodes"].fillna("").astype(str).str.upper()
    frame["is_fog_mist"] = codes.str.contains("FG|BR|HZ", regex=True)
    frame["is_snow_ice"] = codes.str.contains("SN|SG|IC|PL|FZRA|FZDZ|GS", regex=True)
    frame["is_rain"] = codes.str.contains("RA|DZ|SHRA", regex=True)
    frame["is_thunderstorm"] = codes.str.contains("TS", regex=True)
    # Low-visibility procedures reduce movement rates well before they close a field.
    frame["is_low_visibility"] = (frame["vsby"].fillna(10.0) < 3.0) | frame["is_fog_mist"]
    # De-icing demand needs cold *and* moisture, not cold alone.
    frame["is_deicing_risk"] = (frame["tmpc"].fillna(15.0) <= 3.0) & (
        frame["is_snow_ice"] | frame["is_rain"] | frame["is_fog_mist"]
    )
    return frame


def covers(target: Path, end: str, start: str | None = None, slack_days: int = 7, max_gap_days: int = 14) -> bool:
    """A cached file counts only if its observations start by the requested start, reach the requested
    end (each within slack) and have no gap longer than ``max_gap_days``."""
    try:
        t = pd.to_datetime(pd.read_parquet(target, columns=["valid_utc"])["valid_utc"], utc=True).dropna().sort_values()
    except Exception:
        return False
    if t.empty or t.iloc[-1] < pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=slack_days):
        return False
    if start is not None and t.iloc[0] > pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=slack_days):
        return False
    return bool(t.diff().max() <= pd.Timedelta(days=max_gap_days))


def fetch_station(icao: str, start: str, end: str, destination: Path) -> dict:
    target = destination / f"{icao}_metar.parquet"
    if target.is_file() and target.stat().st_size >= MINIMUM_USEFUL_BYTES and covers(target, end, start):
        return {"station": icao, "status": "cached", "rows": None, "path": str(target)}

    url = build_url(icao, start, end)
    request = urllib.request.Request(url, headers={"User-Agent": "prc2026-nm-reconstruction/0.1"})
    last_error = ""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = response.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt == RETRIES - 1:
                return {"station": icao, "status": "error", "detail": last_error}
            # A free archive rate-limits rather than refuses; back off much
            # harder for 429 than for a transient network fault.
            throttled = isinstance(error, urllib.error.HTTPError) and error.code == 429
            delay = THROTTLE_BACKOFF_SECONDS if throttled else BACKOFF_SECONDS
            time.sleep(delay * (attempt + 1))

    frame = pd.read_csv(io.StringIO(payload), na_values=["M", "T", "None", ""], low_memory=False)
    if "valid" not in frame.columns or frame.empty:
        # A station with no ASOS archive answers with a header and nothing else.
        return {"station": icao, "status": "empty", "rows": 0}

    frame["valid_utc"] = pd.to_datetime(frame["valid"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["valid_utc"]).sort_values("valid_utc").reset_index(drop=True)
    for field in ("tmpc", "dwpc", "relh", "drct", "sknt", "gust", "alti", "vsby",
                  "ice_accretion_1hr"):
        if field in frame.columns:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "gust" in frame.columns and "sknt" in frame.columns:
        # No gust reported means no gust above the sustained wind.
        frame["gust"] = frame["gust"].fillna(frame["sknt"])
    frame = condition_flags(frame)
    frame["station"] = icao

    if frame.empty:
        return {"station": icao, "status": "empty", "rows": 0}
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return {"station": icao, "status": "downloaded", "rows": int(len(frame)),
            "path": str(target), "first": str(frame.valid_utc.min()),
            "last": str(frame.valid_utc.max())}


def other_side_airports() -> pd.Series:
    """The ten challenge airports and every airport at the far end of a movement (a departure's destination, an
    arrival's origin), ordered by how often they appear in the organiser files."""
    org = paths()["organiser"]
    files = sorted(org.glob("training_2025-*.parquet")) + [org / "ranking.parquet"]
    movements = pd.concat([pd.read_parquet(f, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt"]) for f in files], ignore_index=True)
    far = pd.concat([movements.loc[movements.PHASE_mvt.eq("DEP"), "ADES_mvt"], movements.loc[movements.PHASE_mvt.eq("ARR"), "ADEP_mvt"],
                     pd.Series(list(AIRPORT_TZ) * 10 ** 6)])        # the ten airports first, whatever their far-end count
    far = far.dropna().astype(str).str.strip().str.upper()
    far = far[far.str.fullmatch(r"[A-Z]{4}")]
    return far.value_counts()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=None,
                        help="Defaults to the metar entry of configs/paths.json.")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the N most frequent far-end airports.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--retry-only", action="store_true",
                        help="Only attempt stations with no file yet.")
    args = parser.parse_args()

    destination = args.destination or paths()["metar"]
    if destination is None:
        raise SystemExit("No destination: set metar in configs/paths.json")
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)

    log("counting far-end airports across training and ranking movements")
    counts = other_side_airports()
    stations = list(counts.index)
    if args.limit:
        stations = stations[: args.limit]
    covered = counts.loc[stations].sum() / counts.sum()
    log(f"{len(stations)} stations requested, covering {covered:.2%} of far-end movements")
    log(f"window {args.start} to {args.end}; destination {destination}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_station, s, args.start, args.end, destination): s
                   for s in stations}
        done = 0
        for future in as_completed(futures):
            outcome = future.result()
            results.append(outcome)
            done += 1
            if outcome["status"] in ("downloaded", "error") or done % 50 == 0:
                log(f"{done}/{len(stations)} {outcome['station']}: {outcome['status']}"
                    + (f" ({outcome.get('rows'):,} rows)" if outcome.get("rows") else ""))

    summary: dict[str, int] = {}
    for outcome in results:
        summary[outcome["status"]] = summary.get(outcome["status"], 0) + 1
    usable = {r["station"] for r in results if r["status"] in ("downloaded", "cached")}
    log(f"summary {summary}")
    on_disk = counts.loc[list(usable & set(counts.index))].sum() / counts.sum()
    log(f"movement coverage by stations now on disk: {on_disk:.2%}")

    manifest = destination / "download_manifest.json"
    manifest.write_text(json.dumps({
        "endpoint": ENDPOINT,
        "licence": "Iowa Environmental Mesonet ASOS archive; public domain / CC BY 4.0",
        "window": {"start": args.start, "end": args.end},
        "requested": len(stations),
        "summary": summary,
        "results": sorted(results, key=lambda r: r["station"]),
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {manifest}")


if __name__ == "__main__":
    main()
