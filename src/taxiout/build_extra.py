# SPDX-License-Identifier: GPL-3.0-only
"""Builders for two extra ADS-B-derived inputs (run once; stage 1 then reads their output files).

    python -m taxiout.build_extra taxi  [--workers 8]   # taxi dynamics from the raw adsb.lol traces -> <extra>/taxi_<day>.parquet
    python -m taxiout.build_extra opdi                  # OPDI surface events, flight-level match   -> <extra>/opdi_events.parquet
    python -m taxiout.build_extra types                 # OPDI aircraft type per departure (field filling) -> <extra>/opdi_types.parquet
    python -m taxiout.build_extra stops  [--workers 8]  # ADS-B taxi-out stop episodes and runway entries -> <extra>/stops/

Taxi dynamics (td_): for every departure the ADS-B pipeline linked to an aircraft (`icao24` in clock_<day>.parquet;
the same link that the stand clock and the pushback signature use, so nothing is re-matched), the on-ground trace
points between the start of the departure (the pushback if seen, else the first point after the last > 600 s gap,
else the first ground point within 90 min) and the take-off. From them: points, span, largest gap, path length,
time stationary (< 2 kt) and moving, stops (stationary >= 20 s after first motion), the longest and the last stop,
the time from the end of the last stop to the take-off, mean and 90th-percentile moving speed, turns (heading changes
accumulated past 45 deg while moving) and the total heading change. These describe how
the aircraft moved, not where it was heard (which drifted with the receiver network between 2025 and 2026).
Label-free: the block time is never read.

OPDI surface events (op_), OpenSky's network, independent of adsb.lol. Matched per FLIGHT, not per aircraft: the
OPDI flight must have the same aircraft (icao24 from our link), a runway event at our airport within 3 min of our
take-off, and exactly one such flight must exist; its callsign must agree with the NM callsign where both are known,
and its departure airport must be ours. (Matching per aircraft picks up the inbound flight's events: tested
16 Jan 2025, 41 min before the true off-block at the median.) From the matched flight: its first taxiway entry at our
airport before the take-off (take-off minus that = time taxiing after joining the taxiway system, a lower bound on
the taxi-out), and the number of taxiway entries.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO, paths

KEY = "MVT_ID_mvt"
STOP_KT, STOP_MIN_S, MAX_STEP_S = 2.0, 20.0, 60.0
#: stops2: a gap of MAX_STEP_S..GAP_STOP_MAX_S between two points with < GAP_STOP_M of movement is a stop the
#: receivers did not report (stationary aircraft broadcast sparsely); ENTRY_M: the runway-entry area around the
#: departure's own last ground point
GAP_STOP_MAX_S, GAP_STOP_M, ENTRY_M = 900.0, 50.0, 500.0


def extra_dir() -> Path:
    d = paths()["extra"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def _turns(trk: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """(turns > 45 deg, total heading change in degrees) along the moving part: the reported track is the heading on
    the surface; changes are accumulated until they exceed 45 deg, which counts one turn."""
    h = trk[:-1][moving & np.isfinite(trk[:-1])]
    if len(h) < 2:
        return 0.0, 0.0
    d = (np.diff(h) + 180) % 360 - 180
    turns, acc = 0, 0.0
    for x in d:
        acc += x
        if abs(acc) >= 45:
            turns += 1; acc = 0.0
    return float(turns), float(np.abs(d).sum())


def _dynamics(t, la, lo, gs, start, mvt, trk=None) -> dict:
    m = (t >= start) & (t <= mvt)
    if m.sum() < 3:
        return {}
    t, la, lo, gs = t[m], la[m], lo[m], gs[m]
    trk = trk[m] if trk is not None else None
    dt = np.diff(t); step = _haversine(la[:-1], lo[:-1], la[1:], lo[1:]); ok = dt <= MAX_STEP_S
    g = np.where(np.isfinite(gs[:-1]), gs[:-1], np.where(dt > 0, step / np.maximum(dt, 1e-9) * 1.943844, 0.0))
    still = (g < STOP_KT) & ok; moving = (g >= STOP_KT) & ok
    out = {"td_points": float(m.sum()), "td_span_s": float(t[-1] - t[0]), "td_gap_max_s": float(dt.max()) if len(dt) else np.nan,
           "td_end_gap_s": float(mvt - t[-1]), "td_path_m": float(step[ok].sum()),
           "td_stationary_s": float(dt[still].sum()), "td_moving_s": float(dt[moving].sum())}
    first_move = np.argmax(moving) if moving.any() else len(dt)
    stops, cur, cur_start, last_end = [], 0.0, None, np.nan
    for i in range(first_move, len(dt)):
        if still[i]:
            cur_start = t[i] if cur == 0 else cur_start; cur += dt[i]
        else:
            if cur >= STOP_MIN_S:
                stops.append(cur); last_end = t[i]
            cur = 0.0
    if cur >= STOP_MIN_S:
        stops.append(cur); last_end = t[-1]
    out.update({"td_stops": float(len(stops)), "td_longest_stop_s": float(max(stops)) if stops else 0.0,
                "td_last_stop_s": float(stops[-1]) if stops else 0.0, "td_last_stop_end_to_mvt_s": float(mvt - last_end) if stops else np.nan})
    if trk is not None:
        out["td_turns"], out["td_heading_change_deg"] = _turns(trk, moving)
    gm = g[moving]
    out["td_mean_gs_moving"] = float(gm.mean()) if len(gm) else np.nan
    out["td_p90_gs"] = float(np.quantile(gm, 0.9)) if len(gm) else np.nan
    return out


def taxi_day(args) -> str:
    clock_dir, day, out_dir = args
    target = Path(out_dir) / f"taxi_{day}.parquet"
    if target.exists():
        return f"{day}: exists"
    c = pd.read_parquet(Path(clock_dir) / f"clock_{day}.parquet")
    cols = [x for x in (KEY, "icao24", "mvt", "ob_first_t") if x in c]
    c = c[cols].copy(); c = c[c.icao24.notna()] if "icao24" in c else c.iloc[:0]
    c[KEY] = pd.to_numeric(c[KEY], errors="coerce")                     # some days store the id as text
    rv = Path(clock_dir) / f"rev_{day}.parquet"
    if rv.exists():
        r_ = pd.read_parquet(rv)[[KEY, "rev_t"]]; r_[KEY] = pd.to_numeric(r_[KEY], errors="coerce")
        c = c.merge(r_.drop_duplicates(KEY), on=KEY, how="left")
    rows, cache = [], {}
    for r in c.itertuples():
        icao = str(r.icao24).lower(); p = Path(clock_dir) / "raw" / day / f"trace_full_{icao}.json.gz"
        if not p.exists():
            continue
        if icao not in cache:
            try:
                d = json.load(gzip.open(p, "rt", encoding="utf-8")); base = float(d["timestamp"])
                tr = [e for e in d.get("trace", []) if isinstance(e, list) and len(e) > 5 and e[1] is not None and isinstance(e[3], str)]
                cache[icao] = (np.array([base + e[0] for e in tr]), np.array([e[1] for e in tr], float), np.array([e[2] for e in tr], float),
                               np.array([e[4] if isinstance(e[4], (int, float)) else np.nan for e in tr], float),
                               np.array([e[5] if isinstance(e[5], (int, float)) else np.nan for e in tr], float))
            except Exception:
                cache[icao] = None
        if cache[icao] is None or not len(cache[icao][0]):
            continue
        t, la, lo, gs, trk = cache[icao]; mvt = float(r.mvt)
        cands = [getattr(r, "rev_t", np.nan), getattr(r, "ob_first_t", np.nan)]
        start = next((float(x) for x in cands if x == x and mvt - 5400 <= float(x) <= mvt), mvt - 5400)
        feats = _dynamics(t, la, lo, gs, start, mvt, trk)
        if feats:
            rows.append({KEY: r.MVT_ID_mvt, **feats})
    pd.DataFrame(rows).to_parquet(target, index=False)
    return f"{day}: {len(rows):,} departures"


def build_taxi(workers: int) -> None:
    clock_dir = paths()["adsb_clocks"]; out = extra_dir()
    days = sorted(p.name[6:16] for p in clock_dir.glob("clock_*.parquet"))
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(taxi_day, [(str(clock_dir), d, str(out)) for d in days]):
            print(msg, flush=True)


def build_opdi() -> None:
    opdi = paths()["opdi"]
    clock_dir = paths()["adsb_clocks"]; org = paths()["organiser"]
    cl = pd.concat([pd.read_parquet(f, columns=[KEY, "icao24", "mvt", "airport"]) for f in sorted(clock_dir.glob("clock_*.parquet"))],
                   ignore_index=True).dropna(subset=["icao24"]).drop_duplicates(KEY)
    cl["icao24"] = cl.icao24.str.lower()
    cs = pd.concat([pd.read_parquet(f, columns=[KEY, "PHASE_mvt", "CALLSIGN_flt"]) for f in sorted(org.glob("training_2025-*.parquet")) + [org / "ranking.parquet"]])
    cs = cs[cs.PHASE_mvt == "DEP"].drop_duplicates(KEY)[[KEY, "CALLSIGN_flt"]]
    cl = cl.merge(cs, on=KEY, how="left")
    airports = set(cl.airport.unique())
    fl = pd.concat([t[t.adep.isin(airports)] for t in (pd.read_parquet(f, columns=["id", "icao24", "flt_id", "adep"])
                                                       for f in sorted((opdi / "flight_list").glob("*.parquet")))]).drop_duplicates("id")
    fl["icao24"] = fl.icao24.astype(str).str.lower(); fl = fl.set_index("id")
    out = []
    for f in sorted((opdi / "flight_events").glob("*.parquet")):
        ev = pd.read_parquet(f, columns=["flight_id", "type", "event_time", "info"])
        ev = ev[ev.type.isin(["entry-taxiway", "entry-runway", "exit-runway"])].copy()
        if ev.empty:
            continue
        ev["t"] = (pd.to_datetime(ev.event_time, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
        ev["apt"] = ev["info"].str.extract(r'"osm_airport":"([A-Z]{4})"')[0]
        ev = ev[ev.flight_id.isin(fl.index)].copy()                           # flights departing one of our airports
        ev["icao24"] = fl.icao24.reindex(ev.flight_id).to_numpy()              # the aircraft, from the OPDI flight list
        lo, hi = ev.t.min() - 600, ev.t.max() + 600
        dep = cl[(cl.mvt >= lo) & (cl.mvt <= hi)]
        rw = ev[ev.type != "entry-taxiway"]
        c = dep.merge(rw[["flight_id", "icao24", "t", "apt"]], on="icao24")
        c = c[(c.apt == c.airport) & ((c.t - c.mvt).abs() <= 180)]
        fid = c.groupby(KEY).flight_id.agg(lambda s: s.unique().tolist())
        fid = fid[fid.map(len) == 1].map(lambda l: l[0])                      # exactly one candidate flight
        m = dep.set_index(KEY).loc[fid.index].assign(flight_id=fid.to_numpy()).reset_index()
        fx = fl.reindex(m.flight_id)
        a = fx.flt_id.fillna("").astype(str).str.replace(" ", "").str.upper().to_numpy()
        b = m.CALLSIGN_flt.fillna("").astype(str).str.replace(" ", "").str.upper().to_numpy()
        ok = ((a == "") | (b == "") | (a == b)) & (fx.adep.fillna("").to_numpy() == m.airport.to_numpy())
        m = m[ok]
        tw = ev[ev.type == "entry-taxiway"].merge(m[[KEY, "flight_id", "mvt", "airport"]], on="flight_id")
        tw = tw[(tw.apt == tw.airport) & (tw.t < tw.mvt) & (tw.t >= tw.mvt - 5400)]
        g = tw.groupby(KEY).agg(first_t=("t", "min"), n=("t", "size"), mvt=("mvt", "first"))
        res = pd.DataFrame({KEY: m[KEY].to_numpy(), "op_matched": 1.0}).set_index(KEY)
        res["op_cp_taxiway"] = (g.mvt - g.first_t).reindex(res.index)
        res["op_taxiway_entries"] = g.n.reindex(res.index).fillna(0.0)
        out.append(res.reset_index()); print(f"{f.name}: {len(res):,} departures matched per flight, {int(res.op_cp_taxiway.notna().sum()):,} with a taxiway entry", flush=True)
    o = pd.concat(out, ignore_index=True).drop_duplicates(KEY, keep="first")
    o.to_parquet(extra_dir() / "opdi_events.parquet", index=False)
    print(f"wrote {extra_dir() / 'opdi_events.parquet'}: {len(o):,} departures, {int(o.op_cp_taxiway.notna().sum()):,} with a taxiway entry")


def build_types() -> None:
    """OPDI aircraft type for every departure it can match: same callsign, same departure airport, first heard at most
    6 h before and 30 min after our take-off and last heard after it (the nearest such flight). A flight first heard
    the day before (night departures) still joins. Validated on NM departures, where the organiser's type is known:
    OPDI agrees on 99.1% of 1.37 M matched rows (LIRF 97.9%). Used by fields.py to fill blank types, never labels."""
    opdi = paths()["opdi"]
    org = paths()["organiser"]
    files = sorted(org.glob("training_2025-*.parquet")) + [org / "ranking.parquet"]
    d = pd.concat([pd.read_parquet(f, columns=[KEY, "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt"]) for f in files]).drop_duplicates(KEY)
    d = d[d.PHASE_mvt == "DEP"].copy()
    secs = lambda x: pd.to_datetime(x, utc=True).dt.tz_convert(None).astype("datetime64[s]").astype("int64")
    d["t"] = secs(d.MVT_TIME_UTC_mvt); d["day"] = d.t // 86400
    d["cs"] = d.FLIGHT_mvt.astype(str).str.upper().str.replace(r"\s", "", regex=True)
    fl = pd.concat([pd.read_parquet(f, columns=["flt_id", "adep", "first_seen", "last_seen", "typecode", "icao_operator"])
                    for f in sorted((opdi / "flight_list").glob("*.parquet"))])
    fl = fl[fl.adep.isin(d.ADEP_mvt.unique()) & fl.flt_id.notna() & fl.typecode.notna()].copy()
    fl["cs"] = fl.flt_id.astype(str).str.upper().str.replace(r"\s", "", regex=True)
    for c in ("first_seen", "last_seen"):
        fl[c] = secs(fl[c])
    fl["day"] = fl.first_seen // 86400; fl = pd.concat([fl, fl.assign(day=fl.day + 1)], ignore_index=True)
    m = d[[KEY, "ADEP_mvt", "cs", "day", "t"]].merge(fl.drop(columns=["flt_id"]), left_on=["ADEP_mvt", "cs", "day"], right_on=["adep", "cs", "day"])
    m = m[(m.first_seen <= m.t + 1800) & (m.last_seen >= m.t - 600) & (m.first_seen >= m.t - 6 * 3600)]
    m["dt"] = np.abs(m.first_seen - m.t); m = m.sort_values("dt").drop_duplicates(KEY)
    o = m[[KEY, "typecode", "icao_operator"]].rename(columns={"typecode": "opdi_type", "icao_operator": "opdi_operator"})
    o.to_parquet(extra_dir() / "opdi_types.parquet", index=False)
    print(f"wrote {extra_dir() / 'opdi_types.parquet'}: {len(o):,} of {len(d):,} departures with an OPDI type")
    build_trace_types()


def build_trace_types() -> None:
    """Aircraft type from the adsb.lol trace file ('t'), for every departure our ADS-B pipeline linked to an aircraft.
    The type belongs to the aircraft, so one trace file per ICAO24 is read. Validated on NM departures: 99.0% agreement
    with the organiser's type (2.07 M rows), 99.6% with OPDI's."""
    from concurrent.futures import ThreadPoolExecutor
    cd = paths()["adsb_clocks"]
    cl = pd.concat([pd.read_parquet(f, columns=[KEY, "icao24", "date"]) for f in sorted(cd.glob("clock_*.parquet"))]).dropna(subset=["icao24"])
    cl["icao24"] = cl.icao24.str.lower(); first = cl.drop_duplicates("icao24")

    def read(r):
        try:
            return r.icao24, json.loads(gzip.open(cd / "raw" / str(r.date)[:10] / f"trace_full_{r.icao24}.json.gz").read()).get("t")
        except Exception:
            return r.icao24, None
    with ThreadPoolExecutor(16) as ex:
        ty = pd.DataFrame(list(ex.map(read, first.itertuples(index=False))), columns=["icao24", "trace_type"])
    o = cl[[KEY, "icao24"]].merge(ty.dropna(), on="icao24")[[KEY, "trace_type"]]
    o.to_parquet(extra_dir() / "trace_types.parquet", index=False)
    print(f"wrote {extra_dir() / 'trace_types.parquet'}: {len(o):,} departures with a trace-file type ({ty.trace_type.notna().sum():,} aircraft)")


def stops_day(args) -> str:
    """Stop episodes of each departure's taxi-out (ground points only, same window as taxi_day): every run of < 2 kt
    lasting >= 20 s after the aircraft first moved, with where and when; plus the taxi's first ground point and its
    last ground point before the take-off (the runway entry). Label-free."""
    clock_dir, day, out_dir = args[:3]; gap_stops = len(args) > 3 and bool(args[3])
    t_s, t_e = Path(out_dir) / f"stops_{day}.parquet", Path(out_dir) / f"taxiends_{day}.parquet"
    if t_s.exists() and t_e.exists():
        return f"{day}: exists"
    c = pd.read_parquet(Path(clock_dir) / f"clock_{day}.parquet")
    cols = [x for x in (KEY, "icao24", "mvt", "ob_first_t", "airport") if x in c]
    c = c[cols].copy(); c = c[c.icao24.notna()] if "icao24" in c else c.iloc[:0]
    c[KEY] = pd.to_numeric(c[KEY], errors="coerce")
    rv = Path(clock_dir) / f"rev_{day}.parquet"
    if rv.exists():
        r_ = pd.read_parquet(rv)[[KEY, "rev_t"]]; r_[KEY] = pd.to_numeric(r_[KEY], errors="coerce")
        c = c.merge(r_.drop_duplicates(KEY), on=KEY, how="left")
    stops, ends, cache = [], [], {}
    for r in c.itertuples():
        icao = str(r.icao24).lower(); p_ = Path(clock_dir) / "raw" / day / f"trace_full_{icao}.json.gz"
        if not p_.exists():
            continue
        if icao not in cache:
            try:
                d = json.load(gzip.open(p_, "rt", encoding="utf-8")); base = float(d["timestamp"])
                tr = [e for e in d.get("trace", []) if isinstance(e, list) and len(e) > 5 and e[1] is not None and isinstance(e[3], str)]
                cache[icao] = (np.array([base + e[0] for e in tr]), np.array([e[1] for e in tr], float), np.array([e[2] for e in tr], float),
                               np.array([e[4] if isinstance(e[4], (int, float)) else np.nan for e in tr], float))
            except Exception:
                cache[icao] = None
        if cache[icao] is None or not len(cache[icao][0]):
            continue
        t, la, lo, gs = cache[icao]; mvt = float(r.mvt)
        cands = [getattr(r, "rev_t", np.nan), getattr(r, "ob_first_t", np.nan)]
        start = next((float(x) for x in cands if x == x and mvt - 5400 <= float(x) <= mvt), mvt - 5400)
        m = (t >= start) & (t <= mvt)
        if m.sum() < 3:
            continue
        t, la, lo, gs = t[m], la[m], lo[m], gs[m]
        dt = np.diff(t); step = _haversine(la[:-1], lo[:-1], la[1:], lo[1:]); ok = dt <= MAX_STEP_S
        g = np.where(np.isfinite(gs[:-1]), gs[:-1], np.where(dt > 0, step / np.maximum(dt, 1e-9) * 1.943844, 0.0))
        hidden = (dt > MAX_STEP_S) & (dt <= GAP_STOP_MAX_S) & (step < GAP_STOP_M) if gap_stops else np.zeros(len(dt), bool)
        still = ((g < STOP_KT) & ok) | hidden; moving = (g >= STOP_KT) & ok & ~hidden
        ap = getattr(r, "airport", None)
        ends.append({KEY: r.MVT_ID_mvt, "airport": ap, "first_t": t[0], "first_lat": la[0], "first_lon": lo[0],
                     "last_t": t[-1], "last_lat": la[-1], "last_lon": lo[-1], "n_points": int(m.sum()),
                     # entry into the runway-entry area: the first point after the last one > ENTRY_M from the own last point
                     "entry_t": float(t[min(len(t) - 1, int(np.flatnonzero(np.r_[True, _haversine(la, lo, la[-1], lo[-1])[:-1] > ENTRY_M])[-1]))]),
                     "gap_s": float(dt[dt > MAX_STEP_S].sum()), "hidden_stop_s": float(dt[hidden].sum())})
        first_move = int(np.argmax(moving)) if moving.any() else len(dt)
        i, k = first_move, 0
        while i < len(dt):
            if still[i]:
                j = i
                while j < len(dt) and still[j]:
                    j += 1
                dur = float(t[j] - t[i])
                if dur >= STOP_MIN_S:
                    stops.append({KEY: r.MVT_ID_mvt, "airport": ap, "k": k, "t_start": float(t[i]), "t_end": float(t[j]), "dur": dur,
                                  "hidden_s": float(dt[i:j][hidden[i:j]].sum()),
                                  "lat": float(la[i:j + 1].mean()), "lon": float(lo[i:j + 1].mean())}); k += 1
                i = j
            else:
                i += 1
    pd.DataFrame(stops).to_parquet(t_s, index=False); pd.DataFrame(ends).to_parquet(t_e, index=False)
    return f"{day}: {len(ends):,} departures, {len(stops):,} stops"


def build_stops(workers: int, gap_stops: bool = False) -> None:
    """stops: as reported; stops2: also the stops hidden in gaps between points (-> <extra>/stops2/)."""
    clock_dir = paths()["adsb_clocks"]; out = extra_dir() / ("stops2" if gap_stops else "stops"); out.mkdir(exist_ok=True)
    days = sorted(p.name[6:16] for p in clock_dir.glob("clock_*.parquet"))
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(stops_day, [(str(clock_dir), d, str(out), gap_stops) for d in days]):
            print(msg, flush=True)


def build_hotspots() -> None:
    """hs_ features from the stop episodes (needs `build_extra stops` first) -> <extra>/hotspots.parquet."""
    from . import data, hotspots
    from .fields import fill_fields
    m, _ = data.load(paths()["organiser"]); m, _ = fill_fields(m, extra_dir())
    out, meta = hotspots.build(extra_dir() / "stops", m)
    out.to_parquet(extra_dir() / "hotspots.parquet", index=False)
    print(f"wrote {extra_dir() / 'hotspots.parquet'}: {len(out):,} departures; {meta}")


def build_hotspots2() -> None:
    """hs2_ features from `build_extra stops2` -> <extra>/hotspots2.parquet."""
    from . import data, hotspots
    from .fields import fill_fields
    m, _ = data.load(paths()["organiser"]); m, _ = fill_fields(m, extra_dir())
    out, meta = hotspots.build2(extra_dir() / "stops2", m)
    out.to_parquet(extra_dir() / "hotspots2.parquet", index=False)
    print(f"wrote {extra_dir() / 'hotspots2.parquet'}: {len(out):,} departures; {meta}")


def build_inbound() -> None:
    """ib_: the departing aircraft's previous arrival at our airport. The aircraft is our ICAO24 link (clock files);
    its inbound is the latest OPDI flight of that aircraft into our airport that was last heard before our take-off
    (within 36 h), matched to the organiser's arrival row by callsign and time (+-60 min) for its schedule, stand and
    taxi-in (supplied on arrival rows). Label-free.
    ib_ground_s    our take-off - the inbound's in-block (turnaround + our taxi-out; like the stand turnaround age)
    ib_slack_s     our scheduled departure - the inbound's in-block (negative: the aircraft was not there on time)
    ib_arr_delay_s the inbound's arrival delay; ib_taxi_in_s its taxi-in; ib_same_stand whether it arrived at our stand
    -> <extra>/inbound.parquet"""
    from . import data
    opdi = paths()["opdi"]
    m, _ = data.load(paths()["organiser"])
    dep = m[m.PHASE_mvt == "DEP"][[KEY, "airport", "t", "SCHED_TIME_UTC_mvt", "STAND_mvt"]].drop_duplicates(KEY)
    arr = m[(m.PHASE_mvt == "ARR") & np.isfinite(m.taxi_in)][["airport", "FLIGHT_mvt", "t", "SCHED_TIME_UTC_mvt", "STAND_mvt", "taxi_in"]].copy()
    arr["cs"] = arr.FLIGHT_mvt.astype(str).str.upper().str.replace(r"\s", "", regex=True)
    cl = pd.concat([pd.read_parquet(f, columns=[KEY, "icao24"]) for f in sorted(paths()["adsb_clocks"].glob("clock_*.parquet"))]).dropna()
    cl[KEY] = pd.to_numeric(cl[KEY], errors="coerce"); cl["icao24"] = cl.icao24.str.lower()
    dep = dep.merge(cl.drop_duplicates(KEY), on=KEY)
    secs = lambda x: pd.to_datetime(x, utc=True).dt.tz_convert(None).astype("datetime64[s]").astype("int64")
    fl = pd.concat([pd.read_parquet(f, columns=["icao24", "flt_id", "ades", "last_seen"]) for f in sorted((opdi / "flight_list").glob("*.parquet"))])
    fl = fl[fl.ades.isin(dep.airport.unique()) & fl.icao24.notna()].copy()
    fl["icao24"] = fl.icao24.astype(str).str.lower(); fl["last_seen"] = secs(fl.last_seen).astype(float)
    fl["cs"] = fl.flt_id.astype(str).str.upper().str.replace(r"\s", "", regex=True)
    fl = fl.sort_values("last_seen")
    for d_ in (dep, fl, arr):
        for c_ in ("icao24", "airport", "ades", "cs"):
            if c_ in d_:
                d_[c_] = d_[c_].astype(str).astype(object)
    # latest inbound of the same aircraft into our airport, last heard before our take-off and within 36 h
    j = pd.merge_asof(dep.sort_values("t"), fl.rename(columns={"ades": "airport"}), left_on="t", right_on="last_seen", by=["icao24", "airport"],
                      direction="backward", tolerance=36 * 3600)
    j = j[j.last_seen.notna()]
    # the organiser's arrival row of that inbound: same airport and callsign, landing within 60 min of OPDI's last message
    a = arr.sort_values("t"); j = j.sort_values("last_seen")
    k = pd.merge_asof(j, a.rename(columns={"t": "arr_t", "SCHED_TIME_UTC_mvt": "arr_sched", "STAND_mvt": "arr_stand"}),
                      left_on="last_seen", right_on="arr_t", by=["airport", "cs"], direction="nearest", tolerance=3600)
    inblock = np.where(k.arr_t.notna(), k.arr_t + k.taxi_in.fillna(0), k.last_seen)
    out = pd.DataFrame({KEY: k[KEY].to_numpy(), "ib_found": 1.0, "ib_matched_arrival": k.arr_t.notna().astype(float).to_numpy(),
                        "ib_ground_s": (k.t - inblock).to_numpy(), "ib_slack_s": (k.SCHED_TIME_UTC_mvt - inblock).to_numpy(),
                        "ib_arr_delay_s": (k.arr_t - k.arr_sched).to_numpy(), "ib_taxi_in_s": k.taxi_in.to_numpy(),
                        "ib_same_stand": (k.arr_stand.astype(str) == k.STAND_mvt.astype(str)).astype(float).where(k.arr_t.notna()).to_numpy()})
    out = out[(out.ib_ground_s > 0) | out.ib_ground_s.isna()]
    out.to_parquet(extra_dir() / "inbound.parquet", index=False)
    print(f"wrote {extra_dir() / 'inbound.parquet'}: {len(out):,} of {len(dep):,} linked departures have an inbound; "
          f"{int(out.ib_matched_arrival.sum()):,} matched to an organiser arrival row")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["taxi", "opdi", "types", "stops", "stops2", "hotspots", "hotspots2", "inbound"]); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    {"taxi": lambda: build_taxi(a.workers), "opdi": build_opdi, "types": build_types, "stops": lambda: build_stops(a.workers), "stops2": lambda: build_stops(a.workers, True), "hotspots": build_hotspots,
     "hotspots2": build_hotspots2, "inbound": build_inbound}[a.what]()
