# SPDX-License-Identifier: GPL-3.0-only
"""Pushback signature: the aircraft moving backwards on the surface, the last such run before take-off.

On the surface the reported track is the aircraft's heading; during a pushback the aircraft moves opposite to it. For
a departure with a trace: consecutive surface points 3-60 m apart and <= 30 s apart whose motion bearing differs from
the reported track by > 135 deg, within 90 min before take-off; runs split by > 120 s; the last run before take-off is
the pushback.
    rev_t   first point of that run       rev_end_t  its last point
    rev_n   moving-backwards steps in it  rev_gs     ground speed at rev_t
Label-free (take-off time and the trace only).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .traces import haversine_m

COLUMNS = ["MVT_ID_mvt", "rev_t", "rev_end_t", "rev_n", "rev_gs"]


def bearing(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    y = np.sin(lo2 - lo1) * np.cos(la2); x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def _surface(path: Path):
    d = json.load(gzip.open(path, "rt", encoding="utf-8")); base = float(d["timestamp"])
    tr = [e for e in d.get("trace", []) if isinstance(e, list) and len(e) > 5 and e[1] is not None and isinstance(e[3], str)]
    return (np.array([base + e[0] for e in tr]), np.array([e[1] for e in tr], float), np.array([e[2] for e in tr], float),
            np.array([e[4] if isinstance(e[4], (int, float)) else np.nan for e in tr], float),
            np.array([e[5] if isinstance(e[5], (int, float)) else np.nan for e in tr], float))


def pushbacks(clock: pd.DataFrame, raw_day: Path) -> pd.DataFrame:
    """One row per departure of a clock file (MVT_ID_mvt, icao24, mvt) whose trace shows a backward run."""
    rows, cache = [], {}
    for r in clock[clock.icao24.notna()].itertuples():
        icao = str(r.icao24).lower(); p = Path(raw_day) / f"trace_full_{icao}.json.gz"
        if not p.exists():
            continue
        if icao not in cache:
            try:
                cache[icao] = _surface(p)
            except Exception:                                       # a broken trace must not stop the day
                cache[icao] = None
        if cache[icao] is None:
            continue
        t, la, lo, gs, trk = cache[icao]; m = (t >= r.mvt - 5400) & (t <= r.mvt)
        if m.sum() < 3:
            continue
        t, la, lo, gs, trk = t[m], la[m], lo[m], gs[m], trk[m]
        dd = haversine_m(la[:-1], lo[:-1], la[1:], lo[1:]); br = bearing(la[:-1], lo[:-1], la[1:], lo[1:])
        diff = np.abs((br - trk[1:] + 180) % 360 - 180)
        rev = np.where((dd >= 3) & (dd <= 60) & (diff > 135) & (np.diff(t) <= 30))[0]
        if not len(rev):
            continue
        k = len(rev) - 1; end = rev[k]
        while k > 0 and t[rev[k]] - t[rev[k - 1]] <= 120:
            k -= 1
        rows.append({"MVT_ID_mvt": r.MVT_ID_mvt, "rev_t": float(t[rev[k]]), "rev_end_t": float(t[end + 1]),
                     "rev_n": int(len(rev) - k), "rev_gs": float(gs[rev[k]]) if gs[rev[k]] == gs[rev[k]] else np.nan})
    return pd.DataFrame(rows, columns=COLUMNS)
