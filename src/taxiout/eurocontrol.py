# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1 (airport_state): the network's and the airport's day, from EUROCONTROL's open Aviation Intelligence Portal.

Four daily tables per airport (ansperformance.eu): arrival ATFM delay by cause (`airport_arrival_atfm_delay`), airport
traffic (`airport_traffic`), all-cause pre-departure delay (`all_pre_departure_delays`) and ATFM slot adherence
(`atfm_slot_adherence`). The task reconstructs past days, so same-day values are used alongside yesterday's value,
the mean of the previous 3 and 7 days, and a z-score against the airport's own 2025 distribution.
ATFM delay is attached at the departure airport (`ast_`) and at the destination (`ast_des_`: the network holding
the flight back). Columns are prefixed `ast_`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ATFM_CAUSES = {"C": "atc_capacity", "W": "weather", "G": "aerodrome_capacity", "S": "atc_staffing", "I": "industrial_atc",
               "D": "de_icing", "E": "equipment", "A": "accident", "M": "military", "O": "other", "P": "special_event",
               "R": "routing", "T": "equipment_non_atc", "V": "environment", "N": "industrial_non_atc"}
TRENDED = ["atfm_delay_per_arrival", "atfm_share_delayed_15", "departures", "pre_delay_per_flight", "slot_late_share"]
Z = ["departures", "pre_delay_per_flight", "slot_late_share"]


def _read(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t["date"] = pd.to_datetime(t.FLT_DATE).dt.strftime("%Y-%m-%d"); t["airport"] = t.APT_ICAO.astype(str).str.upper()
    return t


def daily(eurocontrol_dir: Path, atfm_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(per airport-day table for the departure airport, per airport-day ATFM table for destinations, hash)."""
    h = hashlib.sha256()
    for p in (atfm_path, *(eurocontrol_dir / f for f in ("airport_traffic.parquet", "all_pre_departure_delays.parquet", "atfm_slot_adherence.parquet"))):
        h.update(p.read_bytes())
    a = _read(atfm_path)
    n = pd.to_numeric(a.FLT_ARR_1, errors="coerce").replace(0, np.nan)
    atfm = pd.DataFrame({"airport": a.airport, "date": a.date, "atfm_arrivals": n,
                         "atfm_delay_per_arrival": pd.to_numeric(a.DLY_APT_ARR_1, errors="coerce").fillna(0) / n,
                         "atfm_share_delayed": pd.to_numeric(a.FLT_ARR_1_DLY, errors="coerce").fillna(0) / n,
                         "atfm_share_delayed_15": pd.to_numeric(a.FLT_ARR_1_DLY_15, errors="coerce").fillna(0) / n})
    for code, name in ATFM_CAUSES.items():
        col = f"DLY_APT_ARR_{code}_1"
        if col in a:
            atfm[f"atfm_cause_{name}"] = pd.to_numeric(a[col], errors="coerce").fillna(0) / n
    atfm = atfm.drop_duplicates(["airport", "date"], keep="last")
    tr = _read(eurocontrol_dir / "airport_traffic.parquet")
    tr = pd.DataFrame({"airport": tr.airport, "date": tr.date, "departures": pd.to_numeric(tr.FLT_DEP_1, errors="coerce"),
                       "arrivals": pd.to_numeric(tr.FLT_ARR_1, errors="coerce")}).drop_duplicates(["airport", "date"], keep="last")
    tr["arrival_share"] = tr.arrivals / (tr.departures + tr.arrivals)
    pd_ = _read(eurocontrol_dir / "all_pre_departure_delays.parquet")
    pd_ = pd.DataFrame({"airport": pd_.airport, "date": pd_.date, "pre_delay_per_flight":
                        pd.to_numeric(pd_.DLY_ALL_PRE_2, errors="coerce") / pd.to_numeric(pd_.FLT_DEP_IFR_2, errors="coerce").replace(0, np.nan)}
                       ).drop_duplicates(["airport", "date"], keep="last")
    sa = _read(eurocontrol_dir / "atfm_slot_adherence.parquet")
    reg = pd.to_numeric(sa.FLT_DEP_REG_1, errors="coerce").replace(0, np.nan)
    sa = pd.DataFrame({"airport": sa.airport, "date": sa.date,
                       "regulated_share": reg / pd.to_numeric(sa.FLT_DEP_1, errors="coerce").replace(0, np.nan),
                       "slot_late_share": pd.to_numeric(sa.FLT_DEP_OUT_LATE_1, errors="coerce") / reg,
                       "slot_early_share": pd.to_numeric(sa.FLT_DEP_OUT_EARLY_1, errors="coerce") / reg}).drop_duplicates(["airport", "date"], keep="last")
    day = atfm.merge(tr, on=["airport", "date"], how="outer").merge(pd_, on=["airport", "date"], how="outer").merge(sa, on=["airport", "date"], how="outer")
    day = day.sort_values(["airport", "date"]).reset_index(drop=True)
    g = day.groupby("airport")
    for c in TRENDED:
        s = g[c]
        day[f"{c}_lag1"] = s.shift(1)
        day[f"{c}_mean3d"] = s.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        day[f"{c}_mean7d"] = s.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    base = day[day.date.str.startswith("2025")].groupby("airport")
    for c in Z:
        mu, sd = base[c].mean(), base[c].std()
        day[f"{c}_z"] = (day[c] - day.airport.map(mu)) / day.airport.map(sd)
    return day, atfm[["airport", "date", "atfm_delay_per_arrival", "atfm_share_delayed_15"]], h.hexdigest()


def features(d: pd.DataFrame, day: pd.DataFrame, atfm: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(d.t, unit="s", utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    q = pd.DataFrame({"airport": d.airport.astype(str).to_numpy(), "date": date, "ades": d.ADES_mvt.astype(str).str.upper().to_numpy()})
    x = q.merge(day, on=["airport", "date"], how="left")
    cols = [c for c in day.columns if c not in ("airport", "date")]
    out = x[cols].add_prefix("ast_")
    des = q.merge(atfm.rename(columns={"airport": "ades"}), on=["ades", "date"], how="left")
    out["ast_des_atfm_delay_per_arrival"] = des.atfm_delay_per_arrival.to_numpy()
    out["ast_des_atfm_share_delayed_15"] = des.atfm_share_delayed_15.to_numpy()
    out["ast_des_minus_dep_delay"] = out.ast_des_atfm_delay_per_arrival - out.ast_atfm_delay_per_arrival
    out["ast_missing"] = x.departures.isna().astype(float).to_numpy()
    out.index = d.index
    return out


def monthly(eurocontrol_dir: Path, d: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """eum_: EUROCONTROL's published monthly airport statistics (open data, ansperformance.eu) for the take-off month.
    Taxi-out: mean actual taxi-out ((reference + additional time) / reference flights), mean additional time, and the
    share of departures EUROCONTROL counted (lower in winter: its mean then leaves out long, de-icing flights; on
    2025 the organiser's mean exceeds it by up to 164 s in January and matches within a few seconds in summer).
    Taxi-in and arrival sequencing (ASMA) additional time describe the arrival side. Label-free for our rows: these
    are published aggregates, available for January and July 2026 like every other month."""
    h = hashlib.sha256(); out = pd.DataFrame(index=d.index)
    ym = pd.to_datetime(d.t, unit="s", utc=True)
    q = pd.DataFrame({"airport": d.airport.astype(str).to_numpy(), "YEAR": ym.dt.year.to_numpy(), "MONTH_NUM": ym.dt.month.to_numpy()})
    for name, prefix in (("taxi_out_additional_time", "eum_to_"), ("taxi_in_additional_time", "eum_ti_"), ("asma_additional_time", "eum_asma_")):
        p = eurocontrol_dir / f"{name}.parquet"
        if not p.exists():
            continue
        h.update(p.read_bytes()); t = pd.read_parquet(p)
        t = t.rename(columns={"APT_ICAO": "airport"}); nb = pd.to_numeric(t.get("TOTAL_REF_NB_FL"), errors="coerce").replace(0, np.nan)
        add = pd.to_numeric(t.get("TOTAL_ADD_TIME_MIN"), errors="coerce") * 60 / nb
        v = pd.DataFrame({"airport": t.airport.astype(str), "YEAR": pd.to_numeric(t.YEAR, errors="coerce"), "MONTH_NUM": pd.to_numeric(t.MONTH_NUM, errors="coerce"),
                          f"{prefix}add_s": add})
        if "TOTAL_REF_TIME_MIN" in t:
            v[f"{prefix}mean_s"] = (pd.to_numeric(t.TOTAL_REF_TIME_MIN, errors="coerce") * 60 / nb) + add
        if "TF" in t:
            v[f"{prefix}counted_share"] = nb / pd.to_numeric(t.TF, errors="coerce").replace(0, np.nan)
        v = v.drop_duplicates(["airport", "YEAR", "MONTH_NUM"], keep="last")
        j = q.merge(v, on=["airport", "YEAR", "MONTH_NUM"], how="left")
        for c in v.columns[3:]:
            out[c] = j[c].to_numpy()
    return out, h.hexdigest()
