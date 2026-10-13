# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1c: one row per departure with its features. Nothing here reads a label or the erased block time.

Families: NM clocks, calendar, traffic (both sides of the take-off: this is a reconstruction of the past, every
movement's take-off/landing is supplied), the NM-proxy runway queue, completed arrivals (taxi-in is supplied on
arrival rows) and the stand turnaround, identity, the no-NM schedule evidence, and the ADS-B clocks as
"take-off minus event" durations (cp_*), which the anchor and model stages turn into disagreements.
"""
from __future__ import annotations

from . import context_extra, queue_arrivals
import numpy as np
import pandas as pd

from .config import AIRPORT_TZ, KEY

NM_LO, NM_HI = -7200.0, 172800.0
T_SCALE = 10 ** 11                      # int64 composite key: group code * T_SCALE + epoch second


def _codes(*cols: pd.Series) -> np.ndarray:
    key = cols[0].astype(str)
    for c in cols[1:]:
        key = key + "|" + c.astype(str)
    return pd.factorize(key)[0].astype(np.int64)


def _composite(code: np.ndarray, t: np.ndarray) -> np.ndarray:
    return code * T_SCALE + np.round(t).astype(np.int64)


def _count(ref_code, ref_t, q_code, q_t, lo, hi, left_closed=True):
    """Number of reference events in the same group with lo <= t - q_t < hi (offsets in seconds, relative to the query)."""
    r = np.sort(_composite(ref_code, ref_t))
    a = np.searchsorted(r, _composite(q_code, q_t + lo), side="left" if left_closed else "right")
    b = np.searchsorted(r, _composite(q_code, q_t + hi), side="left" if left_closed else "right")
    return (b - a).astype(float)


def nm_clocks(d: pd.DataFrame) -> pd.DataFrame:
    t = d.t.to_numpy()
    out = {}
    deltas = {}
    for name, col in (("a3", "AOBT_3_flt"), ("lobt", "LOBT_flt"), ("eobt", "EOBT_1_flt"), ("iobt", "IOBT_flt")):
        x = t - d[col].to_numpy()
        x = np.where((x >= NM_LO) & (x <= NM_HI), x, np.nan)
        deltas[name] = x; out[f"nm_{name}"] = x
    b = deltas["a3"].copy(); src = np.where(np.isfinite(b), 3, -1)          # 3 AOBT_3, 2 LOBT, 1 EOBT_1, 0 IOBT, -1 none
    for k, name in enumerate(("lobt", "eobt", "iobt")):
        fill = ~np.isfinite(b) & np.isfinite(deltas[name]); b[fill] = deltas[name][fill]; src[fill] = 2 - k
    out["nm_b"] = b; out["nm_src"] = src.astype(float)
    stack = np.vstack(list(deltas.values()))
    with np.errstate(all="ignore"):
        out["nm_spread"] = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    out["nm_missing"] = (~np.isfinite(stack)).sum(axis=0).astype(float)
    out["nm_z"] = np.clip(np.nan_to_num(deltas["eobt"] - b), -7200, 7200)          # model A's calibration input
    out["has_nm"] = d.IOBT_flt.notna().to_numpy().astype(float)                      # NM record present (the population split)
    out["nm_b_ok"] = np.isfinite(b).astype(float)                                   # and a usable clock (the anchor's split)
    for name in ("lobt", "eobt", "iobt"):
        out[f"nm_{name}_minus_b"] = deltas[name] - b
    names = ["a3", "lobt", "eobt", "iobt"]
    for i in range(4):
        for j in range(i + 1, 4):
            out[f"csp_{names[i]}_minus_{names[j]}"] = deltas[names[i]] - deltas[names[j]]
    s = t - d.SCHED_TIME_UTC_mvt.to_numpy()
    out["sched"] = s; out["sched_minus_b"] = s - b
    out["sched_abs_log"] = np.log1p(np.abs(s)); out["sched_band"] = np.digitize(s, [-3600, 0, 1800, 3600, 3 * 3600, 12 * 3600, 30 * 3600]).astype(float)
    out["flight_dur_plan"] = d.ARVT_1_flt.to_numpy() - d.EOBT_1_flt.to_numpy()
    return pd.DataFrame(out, index=d.index)


def calendar(d: pd.DataFrame) -> pd.DataFrame:
    utc = pd.to_datetime(d.t, unit="s", utc=True)
    local_h = np.zeros(len(d))
    for ap, tz in AIRPORT_TZ.items():
        m = (d.airport == ap).to_numpy()
        if m.any():
            lt = utc[m].dt.tz_convert(tz); local_h[m] = lt.dt.hour.to_numpy() + lt.dt.minute.to_numpy() / 60
    doy = utc.dt.dayofyear.to_numpy()
    return pd.DataFrame({"hour_local": local_h, "hour_sin": np.sin(2 * np.pi * local_h / 24), "hour_cos": np.cos(2 * np.pi * local_h / 24),
                         "dow": utc.dt.dayofweek.to_numpy().astype(float), "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                         "doy_cos": np.cos(2 * np.pi * doy / 365.25), "month_num": d.month.to_numpy().astype(float)}, index=d.index)


def traffic(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    dep, arr = m[m.PHASE_mvt == "DEP"], m[m.PHASE_mvt == "ARR"]
    ap_all = pd.concat([m.airport, d.airport]); ap_code = pd.factorize(ap_all.astype(str))[0].astype(np.int64)
    ap_map = dict(zip(ap_all.astype(str), ap_code))
    code = lambda s: s.astype(str).map(ap_map).to_numpy(np.int64)
    rw_key = pd.concat([dep.airport + ":" + dep.RUNWAY_mvt.astype(str), d.airport + ":" + d.RUNWAY_mvt.astype(str)])
    rw_code_all = pd.factorize(rw_key)[0].astype(np.int64); nd = len(dep)
    rw_dep, rw_q = rw_code_all[:nd], rw_code_all[nd:]
    qa, qt = code(d.airport), d.t.to_numpy()
    out = {}
    for w in (15, 30, 60):
        s = w * 60
        out[f"dep_before_{w}"] = _count(code(dep.airport), dep.t.to_numpy(), qa, qt, -s, 0)
        out[f"dep_after_{w}"] = _count(code(dep.airport), dep.t.to_numpy(), qa, qt, 1, s + 1)
        out[f"arr_before_{w}"] = _count(code(arr.airport), arr.t.to_numpy(), qa, qt, -s, 0)
        out[f"arr_after_{w}"] = _count(code(arr.airport), arr.t.to_numpy(), qa, qt, 1, s + 1)
        out[f"rwy_dep_before_{w}"] = _count(rw_dep, dep.t.to_numpy(), rw_q, qt, -s, 0)
        out[f"rwy_dep_after_{w}"] = _count(rw_dep, dep.t.to_numpy(), rw_q, qt, 1, s + 1)
    # headways on the runway
    r = np.sort(_composite(rw_dep, dep.t.to_numpy())); q = _composite(rw_q, qt)
    i = np.searchsorted(r, q, side="left"); j = np.searchsorted(r, q, side="right")
    prev = np.where((i > 0), r[np.maximum(i - 1, 0)], -1); nxt = np.where(j < len(r), r[np.minimum(j, len(r) - 1)], -1)
    same_prev = (prev // T_SCALE) == rw_q; same_next = (nxt // T_SCALE) == rw_q
    out["rwy_prev_gap"] = np.where(same_prev, q - prev, np.nan).astype(float)
    out["rwy_next_gap"] = np.where(same_next, nxt - q, np.nan).astype(float)
    # NM-proxy queue: departures NM believes off-block but not yet airborne at our take-off
    ok = dep.AOBT_3_flt.notna() & (dep.AOBT_3_flt <= dep.t)
    for name, gc, qc in (("q_airport", code(dep.airport[ok]), qa), ("q_runway", rw_dep[ok.to_numpy()], rw_q)):
        off = np.sort(_composite(gc, dep.AOBT_3_flt[ok].to_numpy())); up = np.sort(_composite(gc, dep.t[ok].to_numpy()))
        qq = _composite(qc, qt); lo = _composite(qc, np.zeros_like(qt))
        n = (np.searchsorted(off, qq, "left") - np.searchsorted(off, lo, "left")) - (np.searchsorted(up, qq, "left") - np.searchsorted(up, lo, "left"))
        self_in = (d.AOBT_3_flt.to_numpy() < qt) & np.isfinite(d.AOBT_3_flt.to_numpy())
        out[name] = (n - self_in).astype(float)
        # queue composition (qc_): how long the queued flights had been off-block when ours took off
        offv = dep.AOBT_3_flt[ok].to_numpy(); co = _composite(gc, offv); cu = _composite(gc, dep.t[ok].to_numpy())
        oo, uo = np.argsort(co, kind="stable"), np.argsort(cu, kind="stable")
        cso = np.concatenate([[0.0], np.cumsum(offv[oo])]); csu = np.concatenate([[0.0], np.cumsum(offv[uo])])
        so, su = co[oo], cu[uo]
        s1 = cso[np.searchsorted(so, qq, "left")] - cso[np.searchsorted(so, lo, "left")]
        s2 = csu[np.searchsorted(su, qq, "left")] - csu[np.searchsorted(su, lo, "left")]
        own = np.where(self_in, np.nan_to_num(d.AOBT_3_flt.to_numpy()), 0.0)
        members = n - self_in
        out[f"qc_{name}_mean_wait"] = np.where(members > 0, qt - (s1 - s2 - own) / np.maximum(members, 1), np.nan)
    out["qc_runway_queue_per_departure_30"] = out["q_runway"] / (out["rwy_dep_before_30"] + 1.0)
    out["qc_airport_queue_per_departure_30"] = out["q_airport"] / (out["dep_before_30"] + 1.0)
    # completed arrivals: in-block (landing + taxi-in) strictly before our take-off
    inb = arr.t.to_numpy() + arr.taxi_in.to_numpy(); ok_a = np.isfinite(inb)
    ac, it, ti = code(arr.airport)[ok_a], inb[ok_a], arr.taxi_in.to_numpy()[ok_a]
    order = np.argsort(_composite(ac, it)); comp = _composite(ac, it)[order]; cs = np.concatenate([[0.0], np.cumsum(ti[order])])
    for w in (30, 60):
        a = np.searchsorted(comp, _composite(qa, qt - w * 60), "left"); b = np.searchsorted(comp, _composite(qa, qt), "left")
        n = b - a; out[f"arr_done_{w}"] = n.astype(float)
        out[f"arr_taxiin_mean_{w}"] = np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)
    # stand turnaround: age of the last arrival completed at this stand (within 24 h)
    st_all = pd.concat([arr.airport[ok_a] + ":" + arr.STAND_mvt[ok_a].astype(str), d.airport + ":" + d.STAND_mvt.astype(str)])
    sc = pd.factorize(st_all)[0].astype(np.int64); sa, sq = sc[:int(ok_a.sum())], sc[int(ok_a.sum()):]
    comp_s = np.sort(_composite(sa, it)); qs = _composite(sq, qt)
    k = np.searchsorted(comp_s, qs, "left"); last = comp_s[np.maximum(k - 1, 0)]
    age = np.where((k > 0) & ((last // T_SCALE) == sq), qs - last, np.nan).astype(float)
    out["stand_turn_age"] = np.where(age <= 86400, age, np.nan)
    ym_a = pd.to_datetime(arr.t, unit="s", utc=True).dt.strftime("%Y-%m"); ym_d = pd.to_datetime(d.t, unit="s", utc=True).dt.strftime("%Y-%m")
    for name, ka, kd in (("pm_stand_taxiin_median", arr.airport + ":" + arr.STAND_mvt.astype(str), d.airport + ":" + d.STAND_mvt.astype(str)),
                         ("pm_runway_taxiin_median", arr.airport + ":" + arr.RUNWAY_mvt.astype(str), d.airport + ":" + d.RUNWAY_mvt.astype(str))):
        med = pd.DataFrame({"k": (ka + "|" + ym_a).to_numpy(), "v": arr.taxi_in.to_numpy()}).dropna().groupby("k").v.median()
        out[name] = (kd + "|" + ym_d).map(med).to_numpy(dtype=float)
    return pd.DataFrame(out, index=d.index)


MALFORMED = r"^([A-Z]{3})([A-Z]{1,2})(\d.*)$"


def stamp_evidence(d: pd.DataFrame) -> pd.DataFrame:
    """Evidence that the airport stamped the block at the schedule (label-free, from the airport-side flight number).
    A doubled operator prefix (ITYTY672, RYRR90BN) marks a known ingestion fault that co-occurs with stamped blocks;
    a malformed number's de-duplicated twin flying regularly separates the fault from a genuinely rare flight."""
    svc = d.FLIGHT_mvt.astype("string").str.strip().str.upper().fillna("")
    m = svc.str.extract(MALFORMED, expand=True)
    p, rep, num = m[0].fillna(""), m[1].fillna(""), m[2].fillna("")
    doubled = (rep != "") & ((rep == p) | (p.str[1:] == rep) | (p.str[2:] == rep))
    ends = np.array([a.endswith(r) if r else False for a, r in zip(p, rep)])
    normalised = pd.Series(np.where(ends, p + num, ""), index=d.index)
    utc = pd.to_datetime(d.t, unit="s", utc=True); day = utc.dt.strftime("%Y-%m-%d"); mon = utc.dt.strftime("%Y-%m")
    sday = pd.to_datetime(d.SCHED_TIME_UTC_mvt, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    key = pd.DataFrame({"a": d.airport.astype(str), "d": day, "m": mon, "s": svc}, index=d.index)
    named = svc.ne("").to_numpy()
    per_month = key.groupby(["a", "m", "s"])["s"].transform("size")
    dups = key.groupby(["a", "d", "s"])["s"].transform("size") - 1
    counts = key.groupby(["a", "m", "s"]).size()
    twin = pd.Series(counts.reindex(pd.MultiIndex.from_arrays([key.a, key.m, normalised])).to_numpy(), index=d.index)
    return pd.DataFrame({"svc_doubled_prefix": doubled.astype(float).to_numpy(), "svc_malformed": normalised.ne("").astype(float).to_numpy(),
                         "svc_per_month": np.where(named, per_month, np.nan), "svc_dups_same_day": np.where(named, dups, np.nan),
                         "svc_twin_per_month": np.where(normalised.ne(""), twin.fillna(0), np.nan),
                         "sched_prev_day": (sday < day).astype(float).to_numpy()}, index=d.index)


def identity(d: pd.DataFrame) -> pd.DataFrame:
    op = d.CALLSIGN_flt.fillna(d.FLIGHT_mvt).astype(str).str.extract(r"^([A-Z]{3})", expand=False).fillna("UNK")
    s = lambda c: d[c].fillna("NA").astype(str)
    return pd.DataFrame({"airport": s("airport"), "runway": s("airport") + ":" + s("RUNWAY_mvt"), "stand": s("airport") + ":" + s("STAND_mvt"),
                         "stand_runway": s("airport") + ":" + s("STAND_mvt") + ">" + s("RUNWAY_mvt"), "actype": s("AIRCRAFT_TYPE_mvt"),
                         "ap_actype": s("airport") + ":" + s("AIRCRAFT_TYPE_mvt"), "wake": s("WK_TBL_CAT_flt"),
                         "operator": op, "market": s("MARKET_SEGMENT_flt"), "flight_type": s("FLIGHT_TYPE_flt"), "rule": s("FLIGHT_RULE_mvt"),
                         "ades": s("ADES_mvt"),
                         # v4: the organiser's raw values (before field filling), read only by the stamp classifier when
                         # Settings.regime_raw_fields; every other model drops them (model.DROP)
                         **{f"{k}_raw": s(raw if raw in d else filled) for k, raw, filled in (
                             ("actype", "AIRCRAFT_TYPE_raw", "AIRCRAFT_TYPE_mvt"), ("market", "MARKET_SEGMENT_raw", "MARKET_SEGMENT_flt"),
                             ("flight_type", "FLIGHT_TYPE_raw", "FLIGHT_TYPE_flt"), ("wake", "WK_TBL_CAT_raw", "WK_TBL_CAT_flt"))},
                         # 28 Sep field filling: where the type and market came from (0 organiser; codes in fields.py)
                         "type_src": d["type_src"].to_numpy() if "type_src" in d else np.zeros(len(d)),
                         "market_src": d["market_src"].to_numpy() if "market_src" in d else np.zeros(len(d))}, index=d.index)


WAKE_RANK = {"L": 1.0, "M": 2.0, "H": 3.0, "J": 4.0}


def wake_sequence(d: pd.DataFrame) -> pd.DataFrame:
    """wk_: the wake-turbulence separation ladder, read from the movement records alone. Consecutive take-offs on the
    same runway are spaced by the pair's wake categories (median gap: medium behind medium ~99 s, medium behind heavy
    ~133 s, behind a super ~179 s). Per departure: its category (inferred from the aircraft type where NM gives none),
    the leader's, the pair's typical gap (median over all consecutive pairs <= 5 min apart, label-free), the actual gap
    minus that (compression), and the wake cost of the runway's last 30 minutes."""
    wk = d.WK_TBL_CAT_flt.astype("string").str.strip().str.upper()
    by_type = pd.DataFrame({"t": d.AIRCRAFT_TYPE_mvt.astype(str), "w": wk}).dropna().query("w in ['L','M','H','J']").groupby("t").w.agg(lambda x: x.mode().iat[0])
    own = wk.where(wk.isin(list(WAKE_RANK)), d.AIRCRAFT_TYPE_mvt.astype(str).map(by_type)).fillna("M").astype(str)
    rw = (d.airport.astype(str) + ":" + d.RUNWAY_mvt.astype(str)).to_numpy()
    order = np.lexsort((d.t.to_numpy(), rw)); t = d.t.to_numpy()[order]; r = rw[order]; w = own.to_numpy()[order]
    same = np.r_[False, r[1:] == r[:-1]]
    prev_w = np.where(same, np.concatenate([[""], w[:-1]]), ""); gap = np.where(same, np.concatenate([[np.nan], np.diff(t)]), np.nan)
    pair = pd.Series(np.char.add(np.char.add(prev_w.astype(str), ">"), w.astype(str)))
    ok = same & (gap > 0) & (gap <= 300)
    ladder = pd.Series(gap[ok]).groupby(pair[ok].to_numpy()).median()
    expected = pair.map(ladder).to_numpy(dtype=float)
    heavy = np.isin(w, ["H", "J"]).astype(float)
    # runway wake cost of the last 30 minutes: sum of the pairs' typical gaps and the heavy/super share
    code = pd.factorize(r)[0].astype(np.int64); comp = code * 10 ** 11 + np.round(t).astype(np.int64)
    cs_e = np.concatenate([[0.0], np.cumsum(np.nan_to_num(expected))]); cs_h = np.concatenate([[0.0], np.cumsum(heavy)])
    lo = np.searchsorted(comp, comp - 1800, "left"); hi = np.arange(len(comp))
    n = hi - lo
    out = pd.DataFrame({"wk_own": own.map(WAKE_RANK).to_numpy()[order], "wk_leader": pd.Series(prev_w).map(WAKE_RANK).to_numpy(),
                        "wk_pair_expected_gap": expected, "wk_gap_minus_expected": gap - expected,
                        "wk_expected_gap_sum_30": cs_e[hi] - cs_e[lo], "wk_heavy_share_30": np.where(n > 0, (cs_h[hi] - cs_h[lo]) / np.maximum(n, 1), np.nan)})
    res = pd.DataFrame(index=np.arange(len(d)), columns=out.columns, dtype=float)
    res.iloc[order] = out.to_numpy()
    res.index = d.index
    return res


def agreement(d: pd.DataFrame) -> pd.DataFrame:
    """Do the airport's and NM's copies of the flight agree? (agr_: 1 agree, 0 disagree, NaN when either is missing).
    Disagreement goes with stale or mismatched records, which go with large errors."""
    def cmp(x, y):
        a_, b_ = d[x].astype("string").str.strip().str.upper(), d[y].astype("string").str.strip().str.upper()
        both = a_.notna() & b_.notna() & (a_ != "") & (b_ != "")
        return np.where(both.to_numpy(), (a_ == b_).fillna(False).to_numpy().astype(float), np.nan)
    out = {"agr_type": cmp("AIRCRAFT_TYPE_raw" if "AIRCRAFT_TYPE_raw" in d else "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), "agr_origin": cmp("ADEP_mvt", "ADEP_flt"),
           "agr_destination": cmp("ADES_mvt", "ADES_flt"), "agr_destination_filed": cmp("ADES_mvt", "ADES_FILED_flt"),
           "agr_rule": cmp("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt")}
    dig = lambda c: d[c].astype("string").str.extract(r"(\d+)", expand=False).str.lstrip("0")
    a_, b_ = dig("FLIGHT_mvt"), dig("CALLSIGN_flt")
    out["agr_service_number"] = np.where((a_.notna() & b_.notna()).to_numpy(), (a_ == b_).fillna(False).to_numpy().astype(float), np.nan)
    f = pd.DataFrame(out, index=d.index)
    f["agr_disagreements"] = (f == 0).sum(axis=1).astype(float)
    return f


def seasonal_mix(d: pd.DataFrame) -> pd.DataFrame:
    """sea_: how many departures at the airport that day / month carry no NM record, and their share. Label-free:
    the no-NM mix shifts with the season (e.g. LIRF: more ordinary flights in the busy summer months)."""
    utc = pd.to_datetime(d.t, unit="s", utc=True)
    k = pd.DataFrame({"a": d.airport.astype(str).to_numpy(), "day": utc.dt.strftime("%Y-%m-%d").to_numpy(),
                      "m": utc.dt.strftime("%Y-%m").to_numpy(), "nn": d.IOBT_flt.isna().to_numpy().astype(float)}, index=d.index)
    out = pd.DataFrame(index=d.index)
    out["sea_no_nm_day"] = k.groupby(["a", "day"]).nn.transform("sum").to_numpy()
    out["sea_no_nm_month"] = k.groupby(["a", "m"]).nn.transform("sum").to_numpy()
    out["sea_no_nm_share_month"] = k.groupby(["a", "m"]).nn.transform("mean").to_numpy()
    out["sea_departures_month"] = k.groupby(["a", "m"]).nn.transform("size").to_numpy().astype(float)
    return out


def adsb(d: pd.DataFrame, clocks: pd.DataFrame) -> pd.DataFrame:
    c = d[[KEY]].merge(clocks, on=KEY, how="left"); c.index = d.index
    t = d.t.to_numpy()
    out = {"adsb_status": c.status.fillna("none").astype(str), "adsb_ok": c.clock_b_ok.fillna(False).astype(float),
           "cp_first": t - c.ob_first_t.to_numpy(), "cp_rev": t - c.rev_t.to_numpy(), "cp_first_out": t - c.clock_first_out.to_numpy(),
           "cp_last_in": t - c.clock_last_in.to_numpy(), "cp_hold": t - c.hold_start.to_numpy(),
           "first_dist_m": c.ob_first_dist_m.to_numpy(), "first_gs": c.ob_first_gs.to_numpy(), "gap_before_s": c.ob_gap_before_s.to_numpy(),
           "first_is_trace_start": c.ob_first_is_trace_start.to_numpy(), "n_ground": c.n_ground.to_numpy(), "hold_s": c.hold_s.to_numpy(),
           "stand_seen_dist_m": c.stand_seen_dist_m.to_numpy(), "assigned_dist_m": c.assigned_dist_m.to_numpy(),
           "rev_n": c.rev_n.to_numpy(), "rev_gs": c.rev_gs.to_numpy(), "rev_dur": (c.rev_end_t - c.rev_t).to_numpy(),
           "rev_minus_first": (c.rev_t - c.ob_first_t).to_numpy()}
    f = pd.DataFrame(out, index=d.index)
    #: a detected pushback that cannot be this departure's (28 Sep, found by tests/test_pretrain.py): less than 2 min
    #: before the take-off (motion on the runway), faster than 15 kt, or more than 30 min before the aircraft's first
    #: point (an earlier tow or repositioning). 1.6% of pushbacks were off by > 15 min and held 87% of the pushback
    #: clock's squared error; on the rows these rules reject the fallback clock is 2-4x closer to the label, and they
    #: keep 97-99% of good pushbacks. The rejected pushback is dropped and flagged.
    bad = (f.cp_rev < 120) | (f.rev_gs > 15) | (f.rev_minus_first < -1800)
    f["rev_invalid"] = bad.astype(float)
    for c in ("cp_rev", "rev_n", "rev_gs", "rev_dur", "rev_minus_first"):
        f.loc[bad, c] = np.nan
    f["adsb_any"] = (np.isfinite(f.cp_first) | np.isfinite(f.cp_rev) | np.isfinite(f.cp_first_out)).astype(float)
    return f


def build(m: pd.DataFrame, clocks: pd.DataFrame, weather: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    d = m[m.PHASE_mvt == "DEP"].copy()
    parts = [d[[KEY, "set", "month", "label", "t"]], identity(d), nm_clocks(d), stamp_evidence(d), agreement(d), seasonal_mix(d), wake_sequence(d),
             calendar(d), traffic(m, d), queue_arrivals.build(m, d), adsb(d, clocks)]
    out = pd.concat(parts + list(weather or []), axis=1)
    out = pd.concat([out, context_extra.build(m, d, out)], axis=1)          # v4: cx_ family (context_extra.py)
    #: drop columns that carry no information on ANY row (2025 and 2026 alike): e.g. METAR ice accretion (never
    #: reported at these airports), ATFM de-icing / routing causes (never recorded here), the departure-airport
    #: agreement (always equal). Same rule on both years, so train and serve stay aligned.
    keep_always = {KEY, "set", "month", "label", "t"}
    dead = [c for c in out.columns if c not in keep_always and out[c].nunique(dropna=True) <= 1 and not c.startswith(("op_", "td_"))]
    return out.drop(columns=dead)
