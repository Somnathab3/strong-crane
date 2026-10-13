# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1b: ADS-B ground evidence per departure, from the daily clock files built from the adsb.lol archive (ODbL).

Per departure (label-free; built by the ADS-B scripts of the main repository, see README "ADS-B clocks"):
  ob_first_t        first trace point after the last > 600 s reporting gap (transponder back on before leaving)
  clock_b_ok        that point is within 100 m of the assigned stand and <= 6 kt ("seen at the stand")
  ob_first_dist_m / ob_first_gs / ob_gap_before_s / ob_first_is_trace_start / n_ground / status
  hold_start, hold_s  parked hold; clock_first_out / clock_last_in: leaving / last entering the stand area
  stand_seen_dist_m, assigned_dist_m
  rev_t, rev_end_t, rev_n, rev_gs   pushback signature: the last run moving against the reported heading
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import KEY

NUM = ["ob_first_t", "ob_first_dist_m", "ob_first_gs", "ob_gap_before_s", "ob_first_is_trace_start", "n_ground",
       "hold_start", "hold_s", "clock_first_out", "clock_last_in", "stand_seen_dist_m", "assigned_dist_m"]
REV = ["rev_t", "rev_end_t", "rev_n", "rev_gs"]


TIMES = {"hold_start", "clock_first_out", "clock_last_in"}      # stored as timestamps in some files, epoch seconds in others


def _seconds(s: pd.Series, name: str) -> pd.Series:
    if s.dtype == bool or s.dtype == object and name not in TIMES:
        return pd.to_numeric(s.map(lambda v: float(v) if v is not None and v == v else np.nan), errors="coerce")
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)


def load(clock_dir: Path) -> tuple[pd.DataFrame, str]:
    parts, h = [], hashlib.sha256()
    for f in sorted(clock_dir.glob("clock_*.parquet")):
        c = pd.read_parquet(f)
        keep = [x for x in [KEY, "status", "clock_b_ok"] + NUM if x in c]
        c = c[keep].copy()
        r = f.with_name(f.name.replace("clock_", "rev_"))
        if r.exists():
            c = c.merge(pd.read_parquet(r)[[KEY] + REV].drop_duplicates(KEY), on=KEY, how="left")
            h.update(r.read_bytes())
        parts.append(c); h.update(f.read_bytes())
    c = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="last")
    for col in NUM + REV:
        c[col] = _seconds(c[col], col) if col in c else np.nan
    c["clock_b_ok"] = c.get("clock_b_ok", pd.Series(False, index=c.index)).fillna(False).astype(bool)
    c["status"] = c.get("status", pd.Series("none", index=c.index)).fillna("none").astype(str)
    return c, h.hexdigest()
