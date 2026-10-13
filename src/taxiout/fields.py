# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1a': positional field filling of the organiser table, before any feature is built (28 Sep).

Only fields that describe the flight, never its times or label. Every fill is measured on NM departures, where the
organiser's own value is known (accuracy = agreement with it), and the type and market get a source column so the
models can tell a filled value from an organiser one.

| field | problem | fill, in order | accuracy on NM rows |
| --- | --- | --- | --- |
| AIRCRAFT_TYPE_mvt | blank on 99.9% of LIRF no-NM departures (2025), 100% (2026) | OPDI type (callsign + origin + time); the aircraft's ADS-B trace-file type; the same flight number's commonest type | 99.1%; 99.0%; ~87% |
| AIRCRAFT_TYPE_mvt | differs from the NM flight type on 0.32% of NM rows | the NM flight type where OPDI agrees with it (OPDI sides with it 73% of the time) | - |
| WK_TBL_CAT_flt | blank on every no-NM departure | the type's wake category | 100% |
| MARKET_SEGMENT_flt | blank on every no-NM departure | same flight number; operator + type; operator | 97.2%; 99.1%; 91.6% |
| FLIGHT_TYPE_flt | blank on every no-NM departure | operator | 98.8% |
| RUNWAY_mvt | LIRF "7" beside "07" | two-digit runway numbers | - |

Source codes: type_src 0 organiser, 1 NM flight type backed by OPDI, 2 OPDI, 4 trace file, 3 flight number, 9 unknown;
market_src 0 organiser, 1 flight number, 2 operator + type, 3 operator, 9 unknown.
The NM population flag (IOBT present) and every NM time are untouched. The organiser's type is kept as
AIRCRAFT_TYPE_raw so the movement/flight disagreement feature still sees what the organiser sent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY = "MVT_ID_mvt"


def _blank(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    return (s.isna() | x.isna() | x.isin(["", "NA", "nan", "None"])).astype(bool)


def _mode_map(key: pd.Series, val: pd.Series) -> pd.Series:
    """key -> its commonest value (ties: first by count, then as sorted)."""
    g = pd.DataFrame({"k": key.astype("object"), "v": val.astype(str)}).dropna()
    c = g.groupby(["k", "v"]).size().sort_values(ascending=False, kind="stable").reset_index()
    return c.drop_duplicates("k").set_index("k").v


def _fill(target: pd.Series, src: np.ndarray, key: pd.Series, table: pd.Series, code: int) -> None:
    need = (_blank(target) & key.notna()).to_numpy()
    val = key[need].map(table); ok = val.notna().to_numpy()
    pos = np.flatnonzero(need)[ok]
    target.iloc[pos] = val.to_numpy()[ok]; src[pos] = code


def fill_fields(m: pd.DataFrame, extra: Path) -> tuple[pd.DataFrame, dict]:
    m = m.copy(); rep = {}
    nm = m.IOBT_flt.notna().to_numpy()
    cs = m.FLIGHT_mvt.astype("string").str.upper().str.replace(r"\s", "", regex=True).astype("object")
    op = m.CALLSIGN_flt.fillna(m.FLIGHT_mvt).astype(str).str.extract(r"^([A-Z]{3})", expand=False).astype("object")
    m["AIRCRAFT_TYPE_raw"] = m.AIRCRAFT_TYPE_mvt
    for c_ in ("MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"):      # the organiser's own values, for the stamp classifier
        m[c_.replace("_flt", "_raw")] = m[c_]
    # ---- aircraft type
    typ = m.AIRCRAFT_TYPE_mvt.astype("object").copy(); tsrc = np.where(_blank(typ), 9, 0)
    f = extra / "opdi_types.parquet"
    o = m[KEY].map(pd.read_parquet(f).set_index(KEY).opdi_type) if f.exists() else pd.Series(np.nan, index=m.index, dtype="object")
    flt = m.AIRCRAFT_TYPE_flt.astype("object")
    swap = (nm & ~_blank(typ).to_numpy() & ~_blank(flt).to_numpy() & (typ.astype(str) != flt.astype(str)).to_numpy()
            & (o.astype(str) == flt.astype(str)).to_numpy())
    typ[swap] = flt[swap]; tsrc[swap] = 1
    fill_o = (_blank(typ) & o.notna()).to_numpy()
    typ[fill_o] = o[fill_o]; tsrc[fill_o] = 2
    ft_ = extra / "trace_types.parquet"                     # the aircraft's own type from its ADS-B trace file (99.0%)
    if ft_.exists():
        tt = m[KEY].map(pd.read_parquet(ft_).drop_duplicates(KEY).set_index(KEY).trace_type)
        fill_t = (_blank(typ) & tt.notna()).to_numpy(); typ[fill_t] = tt[fill_t]; tsrc[fill_t] = 4
    known = ~_blank(typ).to_numpy() & (tsrc <= 1)
    _fill(typ, tsrc, cs, _mode_map(cs[known], typ[known]), 3)
    tsrc[_blank(typ).to_numpy()] = 9
    m["AIRCRAFT_TYPE_mvt"] = typ; m["type_src"] = tsrc.astype(float)
    rep["type"] = {"swapped_to_flight_type": int(swap.sum()), "opdi": int((tsrc == 2).sum()), "trace_file": int((tsrc == 4).sum()),
                   "flight_number": int((tsrc == 3).sum()), "still_blank": int((tsrc == 9).sum())}
    # ---- wake from type (the organiser's wake is a pure function of the type)
    wk = m.WK_TBL_CAT_flt.astype("object").copy(); wsrc = np.where(_blank(wk), 9, 0)
    w_ok = (~_blank(wk) & ~_blank(typ)).to_numpy()
    _fill(wk, wsrc, typ.where(~_blank(typ)), _mode_map(typ[w_ok].astype(str), wk[w_ok]), 1)
    m["WK_TBL_CAT_flt"] = wk
    # ---- market segment: same flight number, else operator + type, else operator
    mk = m.MARKET_SEGMENT_flt.astype("object").copy(); msrc = np.where(_blank(mk), 9, 0); k_ok = ~_blank(mk).to_numpy()
    opt = (op.astype(str) + ":" + typ.astype(str)).where(op.notna())
    for code, key in ((1, cs), (2, opt), (3, op)):
        _fill(mk, msrc, key, _mode_map(key[k_ok], mk[k_ok]), code)
    m["MARKET_SEGMENT_flt"] = mk; m["market_src"] = msrc.astype(float)
    # ---- flight type from the operator
    ft = m.FLIGHT_TYPE_flt.astype("object").copy(); fsrc = np.where(_blank(ft), 9, 0); f_ok = ~_blank(ft).to_numpy()
    _fill(ft, fsrc, op, _mode_map(op[f_ok], ft[f_ok]), 1)
    m["FLIGHT_TYPE_flt"] = ft
    rep.update({"wake_filled": int((wsrc == 1).sum()), "market_filled": int(((msrc >= 1) & (msrc <= 3)).sum()),
                "flight_type_filled": int((fsrc == 1).sum())})
    # ---- runway numbers as two digits ("7" -> "07", "7L" -> "07L")
    rw = m.RUNWAY_mvt.astype("string"); pad = rw.str.fullmatch(r"\d[LRC]?").fillna(False).astype(bool).to_numpy()
    m.loc[pad, "RUNWAY_mvt"] = "0" + rw[pad]; rep["runway_padded"] = int(pad.sum())
    return m, rep
