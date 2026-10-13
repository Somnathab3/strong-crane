# SPDX-License-Identifier: GPL-3.0-only
"""Departure -> aircraft address (icao24) through the OPDI flight list.

The organiser files carry no airframe identifier. OPDI's flight list does (icao24 per flight), keyed by callsign,
departure airport and date of flight. Keys are tried in order: the NM callsign (CALLSIGN_flt, the ICAO form the
transponder broadcasts), then the movement's FLIGHT_mvt, then its doubled-prefix correction (traces.callsign_variants).
Among candidates the one whose first_seen is nearest the take-off is kept.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .traces import callsign_variants

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
RAW_COLUMNS = ["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt",
               "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt"]


def attach_icao24(departures: pd.DataFrame, date: str, opdi_dir: Path) -> pd.DataFrame:
    ym = pd.Timestamp(date).strftime("%Y%m")
    opdi = pd.read_parquet(Path(opdi_dir) / "flight_list" / f"flight_list_{ym}.parquet",
                           columns=["icao24", "flt_id", "dof", "adep", "registration", "typecode", "first_seen"])
    opdi["flt_id"] = opdi["flt_id"].astype("string").str.strip().str.upper()
    opdi["adep"] = opdi["adep"].astype("string").str.strip().str.upper()
    opdi["dof"] = pd.to_datetime(opdi["dof"], errors="coerce").dt.strftime("%Y-%m-%d")
    opdi["first_seen"] = pd.to_datetime(opdi["first_seen"], errors="coerce", utc=True)
    opdi = opdi[opdi["dof"].eq(date) & opdi["adep"].isin(AIRPORTS) & opdi["icao24"].notna()]
    out = departures.copy()
    out["icao24"] = pd.NA; out["registration"] = pd.NA; out["opdi_candidates"] = 0; out["opdi_callsign_used"] = pd.NA; out["opdi_match_rule"] = pd.NA
    mvt = pd.to_datetime(out["MVT_TIME_UTC_mvt"], utc=True); unresolved = pd.Series(True, index=out.index)
    keys = [("nm_callsign", out["CALLSIGN_flt"].astype("string").str.strip().str.upper()),
            ("literal_callsign", out["FLIGHT_mvt"].astype(str).map(lambda s: (callsign_variants(s) + [None])[0])),
            ("deduplicated_prefix", out["FLIGHT_mvt"].astype(str).map(lambda s: (callsign_variants(s) + [None, None])[1]))]
    for rule, variant in keys:
        keyed = pd.DataFrame({"flt_id": variant.astype("string").str.upper(), "adep": out["ADEP_mvt"].astype("string"), "mvt": mvt, "_row": out.index})
        keyed = keyed[unresolved.to_numpy() & keyed["flt_id"].notna()]
        if keyed.empty:
            continue
        merged = keyed.merge(opdi, on=["flt_id", "adep"], how="inner")
        if merged.empty:
            continue
        merged["_gap"] = (merged["first_seen"] - merged["mvt"]).abs()
        counts = merged.groupby("_row").size(); best = merged.sort_values("_gap").drop_duplicates("_row").set_index("_row"); rows = best.index
        out.loc[rows, "icao24"] = best["icao24"].astype(str).str.strip().str.lower().to_numpy()
        out.loc[rows, "registration"] = best["registration"].to_numpy()
        out.loc[rows, "opdi_candidates"] = counts.reindex(rows).to_numpy()
        out.loc[rows, "opdi_callsign_used"] = best["flt_id"].to_numpy(); out.loc[rows, "opdi_match_rule"] = rule
        unresolved.loc[rows] = False
    return out
