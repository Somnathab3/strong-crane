# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1a: the organiser files as one movement table (2025 training months + the 2026 ranking file).

Times become epoch seconds. `airport` is the movement's airport (ADEP for departures, ADES for arrivals).
`label` is the taxi-out time on 2025 departures and NaN on ranking departures; arrivals keep their taxi-in
(supplied in both files) as `taxi_in`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import KEY, LABEL

TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
             "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True)
    return ((s - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(organiser: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(organiser.glob("training_2025-*.parquet")) + [organiser / "ranking.parquet"]
    parts = []
    for f in files:
        t = pd.read_parquet(f)
        t["set"] = "rank" if f.name == "ranking.parquet" else "train"
        parts.append(t)
    m = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="first")
    for c in TIME_COLS:
        m[c] = epoch(m[c])
    dep = m.PHASE_mvt.eq("DEP")
    m["airport"] = np.where(dep, m.ADEP_mvt, m.ADES_mvt).astype(str)
    m["t"] = m.MVT_TIME_UTC_mvt
    m["label"] = np.where(dep & m["set"].eq("train"), m[LABEL], np.nan)
    m["taxi_in"] = np.where(~dep, m[LABEL], np.nan)
    m["month"] = pd.to_datetime(m.t, unit="s", utc=True).dt.month.to_numpy()
    m = m.drop(columns=[LABEL, "BLOCK_TIME_UTC_mvt"])        # the label and the erased clock never reach a feature
    hashes = {f.name: file_hash(f) for f in files}
    return m, hashes
