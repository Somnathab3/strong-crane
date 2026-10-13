# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1 family hs_ (v4): waiting points found in the aircraft's own ADS-B tracks, and the queue observed at them.

Built from the stop episodes of `build_extra stops` (every stop < 2 kt lasting >= 20 s during a departure's taxi-out,
with its position and time) and each taxi's first ground point and last ground point before the take-off. Waiting
points are found from the data, no map needed:
  * holding point of a runway: the median of departures' last ground points before take-off on that runway; a stop
    within HOLD_M of it is a holding-point wait
  * de-icing / heavy queues: pads next to runway entries cannot be told from holding areas, so the activity itself is
    measured: the share of the airport's tracked departures in the last hour with a stop > 5 min away from the stand
  * start area: within START_M of where the taxi-out began (pushback / engine-start pauses)
  * junction: any other stop; a stop > 600 s elsewhere is flagged as a possible tow or repositioning
Per departure: its own waits where its track exists (hs_own_*), and for every departure at the airport the queue other
aircraft show (hs_q_*), which also describes departures whose own track was not captured. Label-free.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

KEY = "MVT_ID_mvt"
HOLD_M, START_M, CELL_M = 400.0, 250.0, 150.0


def _xy(lat, lon, lat0, lon0):
    return (np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * 111320.0, (np.asarray(lat) - lat0) * 110540.0


def build(stops_dir: Path, movements: pd.DataFrame) -> pd.DataFrame:
    import pyarrow.parquet as pq
    read = lambda pat: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(stops_dir / pat))) if KEY in pq.read_schema(f).names], ignore_index=True)
    st, en = read("stops_*.parquet"), read("taxiends_*.parquet")          # days without a linked departure are empty files
    dep = movements[movements.PHASE_mvt == "DEP"][[KEY, "airport", "RUNWAY_mvt", "t", "month"]].drop_duplicates(KEY)
    dep["rw"] = dep.airport.astype(str) + ":" + dep.RUNWAY_mvt.astype(str)
    en = en.drop(columns=["airport"], errors="ignore").merge(dep[[KEY, "airport", "rw", "t", "month"]], on=KEY)
    st = st.drop(columns=["airport"], errors="ignore").merge(dep[[KEY, "airport", "rw", "t", "month"]], on=KEY)
    ref = en.groupby("airport")[["last_lat", "last_lon"]].median()                      # a local origin per airport
    for d_ in (en, st):
        lat0 = d_.airport.map(ref.last_lat).to_numpy(); lon0 = d_.airport.map(ref.last_lon).to_numpy()
    en["lx"], en["ly"] = _xy(en.last_lat, en.last_lon, en.airport.map(ref.last_lat), en.airport.map(ref.last_lon))
    en["fx"], en["fy"] = _xy(en.first_lat, en.first_lon, en.airport.map(ref.last_lat), en.airport.map(ref.last_lon))
    st["x"], st["y"] = _xy(st.lat, st.lon, st.airport.map(ref.last_lat), st.airport.map(ref.last_lon))
    # holding point per runway: median of the last ground points (2025 only, so 2026 cannot shift the map)
    hp = en[en.month.between(1, 12) & (en.t < 1767225600)].groupby("rw")[["lx", "ly"]].median()
    st = st.merge(en[[KEY, "fx", "fy"]], on=KEY, how="left")
    d_hold = np.hypot(st.x - st.rw.map(hp.lx), st.y - st.rw.map(hp.ly))
    # near any runway's holding point of the same airport
    near_any = np.zeros(len(st), bool)
    for rw_, row in hp.iterrows():
        ap = rw_.split(":")[0]; sel = (st.airport == ap).to_numpy()
        near_any[sel] |= np.hypot(st.x.to_numpy()[sel] - row.lx, st.y.to_numpy()[sel] - row.ly) < HOLD_M
    at_start = np.hypot(st.x - st.fx, st.y - st.fy) < START_M
    st["cell"] = st.airport + ":" + (st.x // CELL_M).astype(int).astype(str) + ":" + (st.y // CELL_M).astype(int).astype(str)
    winter = st.month.isin([12, 1, 2, 3]).to_numpy()
    c = st[~at_start & ~near_any & (st.t < 1767225600)].assign(w=winter[~at_start & ~near_any & (st.t < 1767225600).to_numpy()])
    cs = c.groupby("cell").agg(n=("dur", "size"), med=("dur", "median"), win=("w", "mean"))
    pads = set(cs[(cs.n >= 30) & (cs.med >= 180) & (cs.win >= 0.7)].index)
    st["kind"] = np.select([at_start, d_hold.to_numpy() < HOLD_M, st.cell.isin(pads).to_numpy(), st.dur.to_numpy() > 600],
                           ["start", "hold", "pad", "long_other"], "junction")
    # ---- own waits (departures with a track)
    g = st.groupby([KEY, "kind"]).dur.agg(["sum", "size"]).unstack(fill_value=0)
    own = pd.DataFrame(index=en[KEY].unique())
    for k_ in ("hold", "junction", "long_other"):
        own[f"hs_own_{k_}_s"] = g[("sum", k_)].reindex(own.index).fillna(0.0) if ("sum", k_) in g else 0.0
    own["hs_own_junction_stops"] = g[("size", "junction")].reindex(own.index).fillna(0.0) if ("size", "junction") in g else 0.0
    own["hs_own_track"] = 1.0
    # ---- the observed queue, for every departure
    hold = st[st.kind == "hold"][["rw", "t_start", "t_end", "dur", "t"]]
    out = pd.DataFrame({KEY: dep[KEY].to_numpy()}, index=dep.index)
    q_now = np.zeros(len(dep)); prev3 = np.full(len(dep), np.nan)
    for rw_, grp in dep.groupby("rw"):
        h = hold[hold.rw == rw_]
        if h.empty:
            continue
        ts, te = h.t_start.to_numpy(), h.t_end.to_numpy(); tt = grp.t.to_numpy()
        o = np.argsort(ts); ts, te = ts[o], te[o]
        # waiting at our runway's holding point at any time in the 10 min before our take-off (and not us)
        lo = np.searchsorted(ts, tt - 3600); hi = np.searchsorted(ts, tt)
        q_now[dep.index.get_indexer(grp.index)] = [np.sum(te[a:b] > t_ - 600) for a, b, t_ in zip(lo, hi, tt)]
        # the mean holding wait of the last 3 departures on the runway that took off before us
        ho = h.sort_values("t"); tk, dw = ho.t.to_numpy(), ho.dur.to_numpy()
        k = np.searchsorted(tk, tt)
        prev3[dep.index.get_indexer(grp.index)] = [dw[max(0, j - 3):j].mean() if j > 0 else np.nan for j in k]
    out["hs_q_hold_10"] = q_now; out["hs_q_hold_wait_prev3"] = prev3
    # de-icing: share of the airport's tracked departures in the last hour that stopped at a pad, and their wait
    #: de-icing / heavy-queue activity: pads cannot be told from holding areas at airports that de-ice next to the runway
    #: entry (EDDM, LSZH: long stops 5-9x more frequent below 2 C), so the signal is the share of tracked departures in
    #: the last hour that made a stop > 5 min away from their stand
    longw = st[(st.kind != "start") & (st.dur > 300)].groupby(KEY).size().rename("n_long")
    en2 = en[[KEY, "airport", "t"]].merge(longw, left_on=KEY, right_index=True, how="left").assign(hs_own_pad_s=lambda d_: d_.n_long.fillna(0))
    en2 = en2.sort_values(["airport", "t"])
    pad_share = np.full(len(dep), np.nan); cov = np.zeros(len(dep))
    for ap, grp in dep.groupby("airport"):
        e = en2[en2.airport == ap]; tk = e.t.to_numpy(); pv = (e.hs_own_pad_s.fillna(0).to_numpy() > 0).astype(float)
        cs_ = np.r_[0, np.cumsum(pv)]; tt = grp.t.to_numpy()
        a = np.searchsorted(tk, tt - 3600); b = np.searchsorted(tk, tt)
        n = b - a; idx = dep.index.get_indexer(grp.index)
        cov[idx] = n; pad_share[idx] = np.where(n > 0, (cs_[b] - cs_[a]) / np.maximum(n, 1), np.nan)
    out["hs_q_longwait_share_60"] = pad_share; out["hs_q_tracked_60"] = cov
    out = out.merge(own, left_on=KEY, right_index=True, how="left")
    out["hs_own_track"] = out.hs_own_track.fillna(0.0)
    meta = {"holding_points": len(hp), "pads": len(pads), "stops": len(st), "kinds": st.kind.value_counts().to_dict()}
    return out, meta


# ---- hs2_ (v4 follow-up): stops hidden in gaps, per-entry holding, the queue measured by presence
ENTRY_HOLD_M = 400.0
PRESENCE_LAGS = (120, 300, 600)


def build2(stops_dir: Path, movements: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """From `build_extra stops2` (stops incl. those hidden in gaps; each taxi's entry into the area around its own
    last ground point). Holding is measured per runway entry: the departure's own last ground point is its entry, so
    intersection departures are not pooled with full-length ones. The queue is the number of other tracked departures
    of the runway inside their entry area (entry_t <= T < take-off) at T = our take-off minus 2 / 5 / 10 min, which
    counts aircraft whose earlier taxi was never seen as long as their runway end was. Label-free."""
    import pyarrow.parquet as pq
    read = lambda pat: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(stops_dir / pat))) if KEY in pq.read_schema(f).names], ignore_index=True)
    st, en = read("stops_*.parquet"), read("taxiends_*.parquet")
    dep = movements[movements.PHASE_mvt == "DEP"][[KEY, "airport", "RUNWAY_mvt", "t"]].drop_duplicates(KEY).reset_index(drop=True)
    dep["rw"] = dep.airport.astype(str) + ":" + dep.RUNWAY_mvt.astype(str)
    en = en.drop(columns=["airport"], errors="ignore").drop_duplicates(KEY).merge(dep[[KEY, "airport", "rw", "t"]], on=KEY)
    st = st.drop(columns=["airport"], errors="ignore").merge(en[[KEY, "first_lat", "first_lon", "last_lat", "last_lon"]], on=KEY)
    lat0 = st.last_lat.to_numpy()
    d_entry = np.hypot(*_xy(st.lat, st.lon, lat0, st.last_lon.to_numpy()))
    d_start = np.hypot(*_xy(st.lat, st.lon, st.first_lat.to_numpy(), st.first_lon.to_numpy()))
    st["kind"] = np.select([d_start < START_M, d_entry < ENTRY_HOLD_M], ["start", "hold"], "taxiway")
    g = st.groupby([KEY, "kind"]).dur.sum().unstack(fill_value=0.0)
    own = en[[KEY]].set_index(KEY)
    for k_ in ("hold", "taxiway", "start"):
        own[f"hs2_own_{k_}_s"] = g[k_].reindex(own.index).fillna(0.0) if k_ in g else 0.0
    e = en.set_index(KEY)
    own["hs2_own_entry_s"] = (e.last_t - e.entry_t).clip(lower=0)
    own["hs2_own_hidden_s"] = e.hidden_stop_s; own["hs2_own_gap_s"] = e.gap_s
    own["hs2_own_span_s"] = e.last_t - e.first_t
    # queue by presence, per runway
    out = pd.DataFrame({KEY: dep[KEY].to_numpy()})
    q = {lag: np.zeros(len(dep)) for lag in PRESENCE_LAGS}
    prev3 = np.full(len(dep), np.nan); ntrk = np.zeros(len(dep))
    for rw_, grp in dep.groupby("rw"):
        ee = en[en.rw == rw_]
        if ee.empty:
            continue
        a_in, a_out, keys = ee.entry_t.to_numpy(), ee.last_t.to_numpy(), ee[KEY].to_numpy()
        idx = grp.index.to_numpy(); tt = grp.t.to_numpy().astype(float); me = grp[KEY].to_numpy()
        o_in, o_out = np.sort(a_in), np.sort(a_out)
        for lag in PRESENCE_LAGS:
            T = tt - lag                                                     # present at T: entered <= T and left > T
            n = np.searchsorted(o_in, T, side="right") - np.searchsorted(o_out, T, side="right")
            self_in = np.isin(me, keys)                                      # do not count ourselves
            own_in = pd.Series(a_in, index=keys).reindex(me).to_numpy(); own_out = pd.Series(a_out, index=keys).reindex(me).to_numpy()
            n = n - (self_in & (own_in <= T) & (own_out > T))
            q[lag][idx] = np.maximum(n, 0)
        srt = np.argsort(a_out); w_sorted = (a_out - a_in)[srt]; out_sorted = a_out[srt]
        k = np.searchsorted(out_sorted, tt - 30)                             # entry-area waits of the last 3 take-offs before ours
        prev3[idx] = [w_sorted[max(0, j - 3):j].mean() if j > 0 else np.nan for j in k]
        ntrk[idx] = np.searchsorted(out_sorted, tt) - np.searchsorted(out_sorted, tt - 3600)
    for lag in PRESENCE_LAGS:
        out[f"hs2_q_present_{lag // 60}m"] = q[lag]
    out["hs2_q_entry_wait_prev3"] = prev3; out["hs2_q_tracked_rwy_60"] = ntrk
    out = out.merge(own, left_on=KEY, right_index=True, how="left")
    out["hs2_own_track"] = out[KEY].isin(en[KEY]).astype(float)
    meta = {"stops": len(st), "kinds": st.kind.value_counts().to_dict(), "hidden_stop_share": float((st.get("hidden_s", 0) > 0).mean()),
            "tracked": int(out.hs2_own_track.sum())}
    return out, meta
