# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1 family cx_ (v4): runway-relative wind, low-visibility procedures, runway-configuration changes and the
planned block time relative to its route. Label-free: METAR, the organiser's runway fields and NM planned times.

cx_headwind_kt, cx_crosswind_kt, cx_gust_crosswind_kt
    the METAR wind resolved onto the departure runway (heading = runway number x 10 degrees: "25" -> 250)
cx_lvp, cx_vsby_margin
    low-visibility procedures (visibility <= 550 m, i.e. 0.34 statute miles, or fog / mist codes) and how far
    visibility is above that threshold; departures are spaced further apart under LVP
cx_dep_runways_30, cx_config_changed_60, cx_min_since_config_change, cx_own_runway_new
    how many runways served departures in the last 30 min, whether that set differs from the 30-90 min before, how
    long since the set last changed (15-min buckets), and whether our runway had no departures in the 30-90 min before
cx_plan_rel_route
    NM planned block time (planned arrival - EOBT) minus the 2025 median for the same origin, destination and type:
    the airline's padding, part of which is planned taxi time
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKET = 900


def runway_heading(rw: pd.Series) -> np.ndarray:
    num = pd.to_numeric(rw.astype(str).str.extract(r"^(\d{1,2})", expand=False), errors="coerce")
    return (num * 10.0).to_numpy()


def wind(d: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    hdg = np.radians(runway_heading(d.RUNWAY_mvt))
    wdir = np.arctan2(feats.get("mt_wdir_sin", pd.Series(np.nan, index=d.index)).to_numpy(float),
                      feats.get("mt_wdir_cos", pd.Series(np.nan, index=d.index)).to_numpy(float))
    spd = feats.get("mt_sknt", pd.Series(np.nan, index=d.index)).to_numpy(float)
    gust = feats.get("mt_gust", pd.Series(np.nan, index=d.index)).to_numpy(float)
    rel = wdir - hdg
    out = {"cx_headwind_kt": spd * np.cos(rel), "cx_crosswind_kt": np.abs(spd * np.sin(rel)),
           "cx_gust_crosswind_kt": np.abs(np.where(np.isfinite(gust), gust, spd) * np.sin(rel))}
    vs = feats.get("mt_vsby", pd.Series(np.nan, index=d.index)).to_numpy(float)
    fog = feats.get("mt_is_fog_mist", pd.Series(0.0, index=d.index)).to_numpy(float)
    out["cx_lvp"] = ((np.nan_to_num(vs, nan=99) <= 0.34) | ((fog > 0) & (np.nan_to_num(vs, nan=99) <= 1.0))).astype(float)
    out["cx_vsby_margin"] = vs - 0.34
    return pd.DataFrame(out, index=d.index)


def configuration(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    dep = m[m.PHASE_mvt == "DEP"][["airport", "RUNWAY_mvt", "t"]].dropna()
    dep = dep[dep.RUNWAY_mvt.astype(str).str.match(r"^\d")]
    dep["b"] = (dep.t // BUCKET).astype(np.int64)
    sets = dep.groupby(["airport", "b"]).RUNWAY_mvt.agg(lambda s: "|".join(sorted(set(s.astype(str)))))
    sets = sets.reset_index().sort_values(["airport", "b"])
    sets["changed"] = sets.groupby("airport").RUNWAY_mvt.transform(lambda s: s != s.shift(1)).astype(bool)
    sets["change_b"] = sets.b.where(sets.changed); sets["change_b"] = sets.groupby("airport").change_b.ffill()
    key = lambda ap, b: ap.astype(str) + ":" + b.astype(str)
    idx = sets.set_index(key(sets.airport, sets.b))
    b = (d.t // BUCKET).astype(np.int64)
    out = pd.DataFrame(index=d.index)
    cur = key(d.airport, b); prev = key(d.airport, b - 1); prev2 = key(d.airport, b - 2); prev3 = key(d.airport, b - 3)
    rs = idx.RUNWAY_mvt
    s_now = cur.map(rs).fillna("") + "|" + prev.map(rs).fillna("")
    s_before = prev2.map(rs).fillna("") + "|" + prev3.map(rs).fillna("")
    to_set = lambda s: s.str.split("|").map(lambda x: frozenset(v for v in x if v))
    a, bset = to_set(s_now), to_set(s_before)
    out["cx_dep_runways_30"] = a.map(len).astype(float)
    out["cx_config_changed_60"] = pd.Series([float(x != y and len(x) > 0 and len(y) > 0) for x, y in zip(a, bset)], index=d.index)
    last_change = cur.map(idx.change_b).astype(float)
    out["cx_min_since_config_change"] = np.clip((d.t.to_numpy() - last_change.to_numpy() * BUCKET) / 60.0, 0, 24 * 60)
    own = d.RUNWAY_mvt.astype(str)
    out["cx_own_runway_new"] = pd.Series([float(len(y) > 0 and r not in y) for r, y in zip(own, bset)], index=d.index)
    return out


def plan_relative(d: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    plan = feats.get("flight_dur_plan", pd.Series(np.nan, index=d.index)).astype(float)
    key = d.airport.astype(str) + ">" + d.ADES_mvt.astype(str) + ":" + d.AIRCRAFT_TYPE_mvt.astype(str)
    base = pd.DataFrame({"k": key, "p": plan})[(d["set"] == "train").to_numpy()].dropna()
    med = base.groupby("k").p.median(); cnt = base.groupby("k").size()
    med = med[cnt >= 5]
    return pd.DataFrame({"cx_plan_rel_route": (plan - key.map(med)).clip(-3600, 3600)}, index=d.index)


def build(m: pd.DataFrame, d: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """d: departures (index aligned with feats); feats: the stage-1 columns built so far."""
    return pd.concat([wind(d, feats), configuration(m, d), plan_relative(d, feats)], axis=1)
