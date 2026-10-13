# SPDX-License-Identifier: GPL-3.0-only
"""readsb trace parsing, geometry, archive addressing and callsign spellings."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8
_TRACE_MIN_LEN = 5      # readsb trace rows are positional: [dt, lat, lon, alt, gs, track, flags, vrate, aircraft, source, ...]


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; array-safe."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _as_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def parse_trace(document: dict) -> pd.DataFrame:
    """Flatten one trace_full_<icao>.json into a point table. `alt` is the string "ground" when readsb believes the
    aircraft is on the surface; that flag, not an altitude threshold, defines on_ground."""
    base = float(document.get("timestamp", 0.0))
    rows: list[dict[str, Any]] = []
    for entry in document.get("trace", []) or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < _TRACE_MIN_LEN:
            continue
        alt = entry[3]; aircraft = entry[8] if len(entry) > 8 else None; callsign = None
        if isinstance(aircraft, dict) and isinstance(aircraft.get("flight"), str):
            callsign = aircraft["flight"].strip() or None
        rows.append({"t": base + float(entry[0]), "lat": _as_float(entry[1]), "lon": _as_float(entry[2]),
                     "on_ground": isinstance(alt, str) and alt == "ground", "alt_ft": None if isinstance(alt, str) else _as_float(alt),
                     "gs_kt": _as_float(entry[4]), "callsign": callsign})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["t"], unit="s", utc=True)
    return frame.sort_values("t").reset_index(drop=True)


def release_tag(date, pod: str = "planes-readsb-prod-0") -> str:
    """adsb.lol tags a day as v2025.01.15-planes-readsb-prod-0 (repository globe_history_<year>)."""
    stamp = pd.Timestamp(date)
    return f"v{stamp.year:04d}.{stamp.month:02d}.{stamp.day:02d}-{pod}"


def trace_member_name(icao24: str) -> str:
    """Path of one aircraft inside the day tar."""
    code = icao24.strip().lower()
    return f"./traces/{code[-2:]}/trace_full_{code}.json"


def wanted_members(icao24s: Iterable[str]) -> dict[str, str]:
    """Tar member name -> icao24 for a day's retrieval set."""
    return {trace_member_name(code): code.strip().lower() for code in icao24s}


def callsign_variants(callsign: str) -> list[str]:
    """Callsign spellings to try against OPDI, most literal first. Some FLIGHT_mvt values carry a doubled operator
    prefix (ITYTY680 for ITY680, RYRR90BN for RYR90BN): a three-letter ICAO designator followed by a repeat of its own
    last k letters. Only letter prefixes of callsigns that contain a digit are considered (IATA-style LH996 and
    registration callsigns such as DFEEL are left alone)."""
    text = (callsign or "").strip().upper()
    if len(text) < 4:
        return [text] if text else []
    variants = [text]; prefix = text[:3]
    if not prefix.isalpha() or not any(c.isdigit() for c in text):
        return variants
    for k in (3, 2, 1):
        repeat = text[3:3 + k]
        if len(repeat) == k and repeat.isalpha() and repeat == prefix[3 - k:]:
            candidate = prefix + text[3 + k:]
            if candidate not in variants and len(candidate) >= 3:
                variants.append(candidate)
    return variants
