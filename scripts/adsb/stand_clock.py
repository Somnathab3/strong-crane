# SPDX-License-Identifier: GPL-3.0-only
"""The stand clock of one departure from its ADS-B trace and the X-Plane stand positions.

Label-free: only MVT_TIME, STAND_mvt and the trace are read.
  1. Ground points within 3 km of the airport in [MVT - 8 h, MVT], stopped at the take-off roll.
  2. First point of the outbound leg (after the last reporting gap > 600 s): ob_first_t, its distance to the assigned
     stand (ob_first_dist_m) and speed; clock_b_ok when it is at the assigned stand (<= 100 m) and slow (<= 6 kt).
     Transponders are commonly switched on at pushback, so this point is the start of the pushback when the aircraft was
     silent while parked.
  3. The parked hold: the last run of points that stay within 15 m of their anchor for >= 120 s. Its last point is
     clock_last_in; the first point more than 15 m away is clock_first_out. status "clock" when the hold sits within
     60 m of an X-Plane stand (stand_seen).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .traces import haversine_m, parse_trace

HOLD_RADIUS_M = 15.0
HOLD_MIN_S = 120.0
MATCH_M = 60.0
AIRPORT_RADIUS_M = 3000.0
LOOKBACK_S = 8 * 3600
GAP_S = 600.0
OB_MATCH_M = 100.0
OB_MAX_GS = 6.0


def norm_stand(x) -> str:
    x = str(x).upper().strip()
    x = re.sub(r"^(STAND|GATE|RAMP|PARKING|APRON)\s*", "", x)
    return re.sub(r"\s+", "", x)


def load_stands(xplane_dir: Path) -> pd.DataFrame:
    s = pd.read_parquet(Path(xplane_dir) / "stands.parquet"); s["key"] = s["name"].map(norm_stand)
    return s


def unzero(k: str) -> str:
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", k)


def lookup(frame: pd.DataFrame) -> dict[str, dict[tuple[str, str], tuple[float, float]]]:
    """airport -> {(kind, key): (lat, lon)} for exact keys, zero-stripped keys (H01 = H1) and split-position roots
    (215 = centre of 215A / 215B)."""
    out = {}
    for ap, g in frame.groupby("airport"):
        d = {}
        for k, la, lo in zip(g.key, g.lat, g.lon):
            d.setdefault(("x", k), (la, lo)); d.setdefault(("z", unzero(k)), (la, lo))
        roots = g.assign(root=g.key.map(lambda k: unzero(k)[:-1] if re.search(r"[0-9][A-Z]$", unzero(k)) else None)).dropna(subset=["root"])
        for r, rg in roots.groupby("root"):
            d.setdefault(("r", r), (float(rg.lat.mean()), float(rg.lon.mean())))
        out[ap] = d
    return out


def resolve(ap: str, stand: str, xp: dict, osm: dict) -> tuple[float, float, str]:
    """Position of a stand the X-Plane names do not match exactly: X-Plane with normalised names, then OpenStreetMap."""
    k = norm_stand(stand); z = unzero(k); tries = [("x", k), ("z", z), ("r", z)]
    if re.search(r"[0-9][A-Z]$", z):
        tries.append(("z", z[:-1]))                       # 215A in the data, only 215 in the map
    for src, table in (("xplane", xp.get(ap, {})), ("osm", osm.get(ap, {}))):
        for i, t in enumerate(tries):
            if t in table:
                return (*table[t], src if i == 0 else src + "_norm")
    return np.nan, np.nan, "none"


def airport_centres(stands: pd.DataFrame) -> dict[str, tuple[float, float]]:
    g = stands.groupby("airport")[["lat", "lon"]].median()
    return {a: (r.lat, r.lon) for a, r in g.iterrows()}


def parked_hold(pts: pd.DataFrame) -> tuple[int, int] | None:
    """Indices [i, j] of the last stationary run of >= HOLD_MIN_S within HOLD_RADIUS_M of its anchor (point j)."""
    lat = pts["lat"].to_numpy(); lon = pts["lon"].to_numpy(); t = pts["t"].to_numpy(); j = len(pts) - 1
    while j >= 0:
        i = j
        while i - 1 >= 0 and haversine_m(lat[i - 1], lon[i - 1], lat[j], lon[j]) <= HOLD_RADIUS_M:
            i -= 1
        if t[j] - t[i] >= HOLD_MIN_S and (j - i + 1) >= 3:
            return i, j
        j = i - 1
    return None


def clock_for(dep: pd.Series, trace_path: Path, stands_ap: pd.DataFrame, centre: tuple[float, float]) -> dict:
    out = {"MVT_ID_mvt": dep["MVT_ID_mvt"], "status": "no_trace"}
    if not trace_path.exists():
        return out
    with gzip.open(trace_path, "rt", encoding="utf-8") as fh:
        pts = parse_trace(json.load(fh))
    if pts.empty:
        return out
    mvt = pd.Timestamp(dep["MVT_TIME_UTC_mvt"]).timestamp()
    pts = pts[(pts.t >= mvt - LOOKBACK_S) & (pts.t <= mvt + 120) & pts.lat.notna() & pts.lon.notna()].copy()
    pts["d_apt"] = haversine_m(pts.lat.to_numpy(), pts.lon.to_numpy(), centre[0], centre[1])
    pts = pts[pts.d_apt <= AIRPORT_RADIUS_M]
    air = pts[(~pts.on_ground) & (pts.t >= mvt - 900)]                       # take-off roll: first airborne point near MVT
    t_off = float(air.t.iloc[0]) if len(air) else mvt
    g = pts[pts.t < t_off].reset_index(drop=True)
    out["n_ground"] = int(len(g))
    assigned = norm_stand(dep.get("STAND_mvt")); a = stands_ap[stands_ap.key == assigned]
    if len(g) and len(a):
        tt = g.t.to_numpy(); gaps = np.where(np.diff(tt) > GAP_S)[0]; first = int(gaps[-1] + 1) if len(gaps) else 0; f = g.iloc[first]
        dist = float(haversine_m(a.lat.iloc[0], a.lon.iloc[0], f.lat, f.lon))
        out.update({"ob_first_t": float(f.t), "ob_first_dist_m": dist, "ob_first_gs": float(f.gs_kt) if pd.notna(f.gs_kt) else np.nan,
                    "ob_gap_before_s": float(tt[first] - tt[first - 1]) if first > 0 else np.nan, "ob_first_is_trace_start": first == 0})
        out["clock_b_ok"] = bool(dist <= OB_MATCH_M and (pd.isna(f.gs_kt) or f.gs_kt <= OB_MAX_GS))
    if len(g) < 3:
        out["status"] = "no_ground_points"; return out
    hold = parked_hold(g)
    if hold is None:
        out["status"] = "no_hold"; return out
    i, j = hold
    plat, plon = float(g.lat.iloc[i:j + 1].median()), float(g.lon.iloc[i:j + 1].median())
    d = haversine_m(stands_ap.lat.to_numpy(), stands_ap.lon.to_numpy(), plat, plon); k = int(np.argmin(d))
    out.update({"hold_start": float(g.t.iloc[i]), "hold_s": float(g.t.iloc[j] - g.t.iloc[i]), "hold_n": int(j - i + 1),
                "stand_seen": stands_ap.key.iloc[k], "stand_seen_dist_m": float(d[k]), "clock_last_in": float(g.t.iloc[j])})
    after = g.iloc[j + 1:]
    moved = after[haversine_m(after.lat.to_numpy(), after.lon.to_numpy(), plat, plon) > HOLD_RADIUS_M]
    out["clock_first_out"] = float(moved.t.iloc[0]) if len(moved) else np.nan
    out["gap_s"] = out["clock_first_out"] - out["clock_last_in"] if len(moved) else np.nan
    out["assigned_known"] = bool(len(a))
    out["assigned_dist_m"] = float(haversine_m(a.lat.iloc[0], a.lon.iloc[0], plat, plon)) if len(a) else np.nan
    out["at_assigned"] = bool(len(a)) and out["assigned_dist_m"] <= MATCH_M
    out["status"] = "clock" if d[k] <= MATCH_M else "hold_not_at_a_stand"
    return out
