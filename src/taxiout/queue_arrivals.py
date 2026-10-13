# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1 families rb_ and ac_ (28 Sep): the rest of the old repository's "Run B" queue state, and arrivals that had
already reached their stand. Label-free: departure clocks and arrival rows (whose taxi-in the organiser supplies in
both years) only.

rb_  queue members on our runway (NM believes them off-block, not yet airborne at our take-off): their age at our take-off
     (median, p90, max) and our rank among them; the runway headway sequence (5 take-offs before and after: mean,
     median, min, coefficient of variation); take-offs within 5/10/20 min either side; queue pressure = members / service
     rate (take-offs per minute within +-20 min). The earlier repository already had the member count, their mean wait and the
     adjacent gaps; in an earlier implementation the full family was the most useful congestion input.
ac_  arrivals completed (in-block) before our take-off, at the airport and on our runway: count and taxi-in median / p90
     over 30 and 60 min, the last one's taxi-in and how long ago it finished; landings within +-10 min of our take-off
     at the airport and on our runway (mixed-mode interference).

Cross-month on v2's Jan/Jul holdout residual (out/residual_test.py), beyond the traffic features v2 already had: labels
<= 2 h 217.65 -> 217.32 (rb_) / 217.43 (ac_); NM rows <= 2 h SSE -3.4e7 / -4.4e7. Small, positive, both months.
Physical queue reconstructions (ready-time queue, arrival/service rates, resequencing, surface occupancy) and taxi
route distance were closed in the old repository with no signal; they are deliberately not here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

KEY = "MVT_ID_mvt"
LAG = 60                       # queue members are looked for among the next LAG take-offs on the same runway
MAX_AGE = 7200.0               # a member off-block more than 2 h before our take-off is a clock error or a day-shift, not a wait
RECENT = 7200.0                # an arrival completed more than 2 h ago says nothing about the apron now (and data gaps between
                               # January and July 2026 would otherwise read as 150-day "recencies")
_S = 10 ** 11


def _queue(d: pd.DataFrame) -> pd.DataFrame:
    o = np.lexsort((d.t.to_numpy(), pd.factorize(d.airport.astype(str) + ":" + d.RUNWAY_mvt.astype(str))[0]))
    s = d.iloc[o]
    rw = pd.factorize(s.airport.astype(str) + ":" + s.RUNWAY_mvt.astype(str))[0]
    t = s.t.to_numpy(float); a3 = s.AOBT_3_flt.to_numpy(float); n = len(s); idx = np.arange(n)
    ages = np.full((n, LAG), np.nan, dtype=np.float32); rank = np.zeros(n)
    for k in range(1, LAG + 1):
        jj = np.minimum(idx + k, n - 1)
        same = (idx + k < n) & (rw[jj] == rw) & np.isfinite(a3[jj]) & (a3[jj] < t) & (t - a3[jj] <= MAX_AGE)
        ages[:, k - 1] = np.where(same, t - a3[jj], np.nan)
        rank += same & np.isfinite(a3) & (a3[jj] < a3)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cnt = np.isfinite(ages).sum(1)
        out["rb_queue_n"] = cnt.astype(float)
        out["rb_age_median"] = np.where(cnt > 0, np.nanmedian(ages, 1), np.nan)
        out["rb_age_p90"] = np.where(cnt > 0, np.nanpercentile(ages, 90, axis=1), np.nan)
        out["rb_age_max"] = np.where(cnt > 0, np.nanmax(ages, 1), np.nan)
        out["rb_own_rank"] = np.where(np.isfinite(a3), rank, np.nan)
        gaps = np.full((n, 10), np.nan)
        for c, k in enumerate(list(range(-5, 0)) + list(range(1, 6))):
            j = idx + k; ok = (j >= 0) & (j < n); jj = np.clip(j, 0, n - 1); jp = np.clip(j - np.sign(k), 0, n - 1)
            gaps[:, c] = np.where(ok & (rw[jj] == rw) & (rw[jp] == rw), np.abs(t[jj] - t[jp]), np.nan)
        gaps = np.where(gaps > 3600, np.nan, gaps)
        out["rb_headway_mean"] = np.nanmean(gaps, 1); out["rb_headway_median"] = np.nanmedian(gaps, 1)
        out["rb_headway_min"] = np.nanmin(gaps, 1); out["rb_headway_cv"] = np.nanstd(gaps, 1) / out["rb_headway_mean"]
    key = rw.astype(np.int64) * _S + np.round(t).astype(np.int64)
    me_l, me_r = np.searchsorted(key, key, "left"), np.searchsorted(key, key, "right")
    for w in (5, 10, 20):
        out[f"rb_tko_before_{w}"] = (me_l - np.searchsorted(key, key - w * 60, "left")).astype(float)
        out[f"rb_tko_after_{w}"] = (np.searchsorted(key, key + w * 60, "right") - me_r).astype(float)
    out["rb_pressure"] = out["rb_queue_n"] / ((out["rb_tko_before_20"] + out["rb_tko_after_20"] + 1) / 40.0)
    return pd.DataFrame(out, index=s.index).reindex(d.index)


def _arrivals(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    arr = m[(m.PHASE_mvt == "ARR") & np.isfinite(m.taxi_in) & (m.taxi_in > 0) & (m.taxi_in < 7200)]
    ib_all = (arr.t + arr.taxi_in).to_numpy(); ac = {}
    levels = {"ap": (arr.airport.astype(str), d.airport.astype(str)),
              "rwy": (arr.airport.astype(str) + ":" + arr.RUNWAY_mvt.astype(str), d.airport.astype(str) + ":" + d.RUNWAY_mvt.astype(str))}
    q_t = d.t.to_numpy()
    for tag, (ka, kd) in levels.items():
        codes = pd.factorize(pd.concat([ka, kd]))[0]; ca, cdp = codes[:len(arr)], codes[len(arr):]
        o = np.lexsort((ib_all, ca)); ib = ib_all[o]; ti = arr.taxi_in.to_numpy()[o]; ga = ca[o]
        kk = ga.astype(np.int64) * _S + np.round(ib).astype(np.int64); q = cdp.astype(np.int64) * _S + np.round(q_t).astype(np.int64)
        for w in (30, 60):
            lo = np.searchsorted(kk, q - w * 60, "left"); hi = np.searchsorted(kk, q, "left"); cnt = hi - lo
            med = np.full(len(d), np.nan); p90 = np.full(len(d), np.nan)
            for i in np.flatnonzero(cnt > 0):
                v = ti[lo[i]:hi[i]]; med[i] = np.median(v); p90[i] = np.percentile(v, 90)
            if tag != "ap":                                  # the airport count duplicates traffic()'s arr_done_{w}
                ac[f"ac_{tag}_done_{w}"] = cnt.astype(float)
            ac[f"ac_{tag}_taxiin_med_{w}"] = med; ac[f"ac_{tag}_taxiin_p90_{w}"] = p90
        last = np.searchsorted(kk, q, "left") - 1; lc = np.clip(last, 0, None)
        okl = (last >= 0) & (ga[lc] == cdp) & (q_t - ib[lc] <= RECENT)
        ac[f"ac_{tag}_last_taxiin"] = np.where(okl, ti[lc], np.nan)
        ac[f"ac_{tag}_since_last_s"] = np.where(okl, q_t - ib[lc], np.nan)
        o2 = np.lexsort((arr.t.to_numpy(), ca)); k2 = ca[o2].astype(np.int64) * _S + np.round(arr.t.to_numpy()[o2]).astype(np.int64)
        ac[f"ac_{tag}_landings_pm10"] = (np.searchsorted(k2, q + 600, "right") - np.searchsorted(k2, q - 600, "left")).astype(float)
    return pd.DataFrame(ac, index=d.index)


def build(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_queue(d), _arrivals(m, d)], axis=1)
