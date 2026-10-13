# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1 (weather): METAR observations and Open-Meteo archived forecasts at the departure airport.

METAR (one file per airport, `<ICAO>_metar.parquet`, from the Iowa Environmental Mesonet ASOS/METAR archive):
the last report within 90 min before the take-off (temperature, dew point, humidity, wind as sine/cosine, gust,
visibility, pressure, ice accretion, fog / snow / rain / thunderstorm / low-visibility / de-icing flags, report
age) and its change over the last 1 h and 3 h (temperature, visibility, pressure, wind).
Open-Meteo (`open_meteo_hourly.parquet`, historical forecast archive, CC BY 4.0): the forecast for the take-off hour
(all numeric fields) plus derived flags: de-icing conditions, snow, fog-prone, freezing precipitation.
Weather widens the error more than it biases it, so these are inputs for the learners, not corrections.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

METAR_NUM = ["tmpc", "dwpc", "relh", "sknt", "gust", "vsby", "alti", "ice_accretion_1hr"]
METAR_FLAGS = ["is_fog_mist", "is_snow_ice", "is_rain", "is_thunderstorm", "is_low_visibility", "is_deicing_risk"]
TREND = ["tmpc", "vsby", "alti", "sknt"]


def _asof(d: pd.DataFrame, obs: pd.DataFrame, shift_s: float, cols: list[str], tol_s: float, suffix: str) -> pd.DataFrame:
    q = pd.DataFrame({"airport": d.airport.to_numpy(), "tq": d.t.to_numpy() - shift_s, "_i": np.arange(len(d))}).sort_values("tq")
    o = obs[["airport", "to"] + cols].sort_values("to")
    j = pd.merge_asof(q, o, left_on="tq", right_on="to", by="airport", direction="backward", tolerance=tol_s)
    j = j.sort_values("_i")
    out = j[cols].add_suffix(suffix).reset_index(drop=True)
    out[f"age{suffix}"] = (j.tq - j.to).to_numpy()
    return out


def metar(d: pd.DataFrame, metar_dir: Path) -> tuple[pd.DataFrame, str]:
    h = hashlib.sha256(); parts = []
    for ap in sorted(d.airport.unique()):
        p = metar_dir / f"{ap}_metar.parquet"
        if not p.exists():
            continue
        h.update(p.read_bytes()); m = pd.read_parquet(p)
        m["airport"] = ap; m["to"] = (m.valid_utc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
        for c in METAR_FLAGS:
            m[c] = m[c].astype(float)
        m["wdir_sin"] = np.sin(np.deg2rad(m.drct)); m["wdir_cos"] = np.cos(np.deg2rad(m.drct))
        parts.append(m[["airport", "to"] + METAR_NUM + METAR_FLAGS + ["wdir_sin", "wdir_cos"]])
    obs = pd.concat(parts, ignore_index=True).dropna(subset=["to"])
    now = _asof(d, obs, 0, METAR_NUM + METAR_FLAGS + ["wdir_sin", "wdir_cos"], 5400, "")
    out = now.rename(columns={c: f"mt_{c}" for c in now.columns})
    for lag, name in ((3600, "1h"), (10800, "3h")):
        past = _asof(d, obs, lag, TREND, 5400, "_p")
        for c in TREND:
            out[f"mt_{c}_chg_{name}"] = now[c].to_numpy() - past[f"{c}_p"].to_numpy()
    out.index = d.index
    return out, h.hexdigest()


def open_meteo(d: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, str]:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    o = pd.read_parquet(path)
    o["hour"] = ((o.valid_utc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).astype(np.int64) // 3600
    num = [c for c in o.columns if c not in ("icao", "valid_utc", "hour") and pd.api.types.is_numeric_dtype(o[c])]
    o = o.drop_duplicates(["icao", "hour"]).rename(columns={"icao": "airport"})
    q = pd.DataFrame({"airport": d.airport.to_numpy(), "hour": (d.t.to_numpy() // 3600).astype(np.int64)})
    j = q.merge(o[["airport", "hour"] + num], on=["airport", "hour"], how="left")
    out = j[num].add_prefix("om_")
    t, rh, pr = j.temperature_2m, j.relative_humidity_2m, j.precipitation
    out["om_deicing"] = ((t <= 3) & ((pr > 0) | (rh >= 90))).astype(float)
    out["om_snow"] = ((j.snowfall > 0) | (j.snow_depth > 0)).astype(float)
    out["om_fog_prone"] = ((j.visibility < 1000) | ((j.boundary_layer_height < 150) & (rh >= 95))).astype(float)
    out["om_freezing_precip"] = ((pr > 0) & (t <= 0)).astype(float)
    # forecast horizon (omh_): the three hours after and before the take-off hour
    oi = o.set_index(["airport", "hour"])
    for name, hours in (("next3h", (1, 2, 3)), ("prev3h", (-1, -2, -3))):
        acc = {c: np.zeros(len(q)) for c in ("precipitation", "snowfall")}; gust = np.full(len(q), -np.inf); seen = np.zeros(len(q))
        for dh in hours:
            k = pd.MultiIndex.from_arrays([q.airport, q.hour + dh])
            r = oi.reindex(k)
            for c in acc:
                acc[c] += np.nan_to_num(r[c].to_numpy(dtype=float))
            gust = np.fmax(gust, r.wind_gusts_10m.to_numpy(dtype=float)); seen += r.temperature_2m.notna().to_numpy()
        for c, v in acc.items():
            out[f"omh_{c}_{name}"] = np.where(seen > 0, v, np.nan)
        out[f"omh_max_gust_{name}"] = np.where(np.isfinite(gust), gust, np.nan)
    out.index = d.index
    return out, h
