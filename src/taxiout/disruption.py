# SPDX-License-Identifier: GPL-3.0-only
"""dis_ family (9 Oct 2026, Settings.disruption): how disrupted the airport is around this departure, from the schedule only.

Computed on the loaded feature table (not part of the cached build, so the cache is unchanged), separately per set so 2025
and the ranking rows never mix. Label-free: every column uses only the schedule gap (`sched`, seconds between schedule and
take-off) of the departures at the same airport.
  dis_hour_late_share  share of the airport's departures in the same UTC take-off hour that are >= 1 h behind schedule
  dis_hour_n           departures at the airport in that hour
  dis_day_late_share   the same share over the UTC day
  dis_gap_rank_hour    this departure's schedule-gap percentile within the airport-hour
Residual screen on the V7f4 final run's 2025 out-of-fold predictions (month left out): -0.92 RMSE against its control,
the only new family with a signal (shift, ADS-B-vs-schedule, NM-vs-schedule: ~0)."""
from __future__ import annotations

import pandas as pd


def add(f: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame({"set": f["set"].astype(str).to_numpy(), "a": f.airport.astype(str).to_numpy(),
                      "h": (f.t // 3600).to_numpy(), "d": (f.t // 86400).to_numpy(), "s": f.sched.to_numpy()}, index=f.index)
    k["late"] = (k.s >= 3600).astype(float)
    gh, gd = k.groupby(["set", "a", "h"]), k.groupby(["set", "a", "d"])
    f = f.copy()
    f["dis_hour_late_share"] = gh.late.transform("mean").to_numpy()
    f["dis_hour_n"] = gh.late.transform("size").to_numpy().astype(float)
    f["dis_day_late_share"] = gd.late.transform("mean").to_numpy()
    f["dis_gap_rank_hour"] = gh.s.rank(pct=True).to_numpy()
    return f
