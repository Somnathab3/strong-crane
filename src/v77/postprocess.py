# SPDX-License-Identifier: GPL-3.0-only
"""Stage 12: reconcile the predictions with EUROCONTROL's published daily pre-departure delay, and the orphan window.

EUROCONTROL publishes, per airport and day, the total pre-departure delay of the day's departures
(all_pre_departure_delays, DLY_ALL_PRE_2, minutes). On the 2025 labels it equals, to within rounding, the sum over the
day's departures of max(0, d - y), where d = take-off - schedule and y = taxi-out (correlation 0.9999 with the day of
the actual off-block). The total is public for every 2026 day, so a day's predictions can be checked against it.

  1. Daily reconciliation. For each airport and predicted off-block day whose departure count matches the published
     count (tolerance `count_tolerance`), the predictions of the day's delayed departures (d > p) are lowered or raised
     by one equal amount s_g until sum max(0, d - p) equals the published total (iterated as departures enter or leave
     the delayed set). 2025 out of fold: 230.91 -> 229.01 s.
  2. Loaded day shift. The equal shift s_g is the day's common error; every departure of the day responds to it with
     a loading b that depends on its group (delayed or not, schedule-gap band): p <- p0 - b s_g. The loadings are
     fitted on the 2025 out-of-fold predictions by a through-origin regression of (p0 - y) on s_g per group, clipped.
     2025 out of fold: 229.01 -> 228.85 s.
  3. Orphan window. A LIRF / LTFM departure without a flight-plan record whose flight has an arrival record with a
     LOBT at the destination (same flight number and route, landing 10 min to 20 h after take-off) did not match its
     plan, so its off-block is outside take-off - LOBT +- 3,606 s (0 of 190 such 2025 labels inside). A prediction
     inside that window is moved to its nearer edge. 2025 out of fold, with 2: 228.57 s.

    python src/v77/postprocess.py fit  --oof <dev_oof.parquet> --out <fit.json>          # loadings from a 2025 OOF file
    python src/v77/postprocess.py apply --file <submission> --fit <fit.json> --out <file> # on a 2026 submission file
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from taxiout.config import KEY, LABEL, paths   # noqa: E402

COLS = [KEY, "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "FLIGHT_ID_mvt"]
DEFAULTS = {"daily_table": "all_pre_departure_delays.parquet", "count_tolerance": 5, "iterations": 8,
            "bands_s": [900, 1800, 2700, 3600], "loading_clip": [-0.5, 2.5], "orphan_airports": ["LIRF", "LTFM"], "half_width_s": 3606}


def settings(cfg: dict | None) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("postprocess") or {})}


def _epoch(s: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).to_numpy(float)


def movements(files: list[Path]) -> pd.DataFrame:
    """Departures and arrivals of the organiser files with the fields the stage needs."""
    m = pd.concat([pd.read_parquet(f, columns=COLS) for f in files]).drop_duplicates(KEY).reset_index(drop=True)
    m["t"] = _epoch(m.MVT_TIME_UTC_mvt); m["lobt"] = _epoch(m.LOBT_flt); m["d"] = m.t - _epoch(m.SCHED_TIME_UTC_mvt)
    return m


def daily_table(st: dict) -> pd.DataFrame | None:
    p = paths().get("eurocontrol")
    f = (Path(p) / st["daily_table"]) if p else None
    if f is None or not f.exists():
        return None
    e = pd.read_parquet(f); e["day"] = pd.to_datetime(e.FLT_DATE).dt.strftime("%Y-%m-%d")
    return e[["APT_ICAO", "day", "FLT_DEP_IFR_2", "DLY_ALL_PRE_2"]]


def reconcile(dep: pd.DataFrame, p: np.ndarray, E: pd.DataFrame, st: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Equal shift of each matched day's delayed departures; returns the new predictions, the day shift per row and
    the share of rows on reconciled days."""
    X = dep[[KEY, "ADEP_mvt", "d"]].copy(); X["p"] = p; X["day"] = pd.to_datetime(dep.t.to_numpy() - p, unit="s").strftime("%Y-%m-%d")
    X["n_day"] = X.groupby(["ADEP_mvt", "day"])[KEY].transform("size")
    X = X.merge(E, left_on=["ADEP_mvt", "day"], right_on=["APT_ICAO", "day"], how="left")
    ok = ((X.FLT_DEP_IFR_2 - X.n_day).abs() <= st["count_tolerance"]) & X.p.notna()
    out = p.copy(); shift = np.zeros(len(p)); C = (X.DLY_ALL_PRE_2 * 60).to_numpy(float); dd = X.d.to_numpy(float)
    for _, idx in X[ok].groupby(["ADEP_mvt", "day"]).groups.items():
        ii = np.asarray(idx); d = dd[ii]; q = out[ii].copy(); c = C[ii][0]
        for _ in range(st["iterations"]):
            act = d - q > 0
            if not act.any():
                break
            gap = c - np.maximum(0, d - q).sum()
            if abs(gap) < 1:
                break
            q = np.maximum(q - gap * act / act.sum(), 0)
        out[ii] = q; moved = p[ii] - q; shift[ii] = moved[moved != 0].mean() if (moved != 0).any() else 0.0
    return out, shift, float(ok.mean())


def groups(d: np.ndarray, p0: np.ndarray, st: dict) -> np.ndarray:
    return np.digitize(d, st["bands_s"]) * 2 + ((d - p0) > 0).astype(int)


def fit_loadings(d: np.ndarray, p0: np.ndarray, y: np.ndarray, shift: np.ndarray, st: dict) -> dict:
    g = groups(d, p0, st); ok = (shift != 0) & np.isfinite(p0) & np.isfinite(y); lo, hi = st["loading_clip"]; b = {}
    for k in np.unique(g[ok]):
        s = ok & (g == k); b[int(k)] = float(np.clip(-(shift[s] * (y[s] - p0[s])).sum() / max((shift[s] ** 2).sum(), 1e-9), lo, hi))
    return b


def loaded_shift(d: np.ndarray, p0: np.ndarray, p1: np.ndarray, shift: np.ndarray, b: dict, st: dict) -> np.ndarray:
    g = groups(d, p0, st); bb = np.array([b.get(int(k), 0.0) for k in g])
    return np.where(shift != 0, np.maximum(p0 - bb * shift, 0), p1)


def orphan_window(dep: pd.DataFrame, arr: pd.DataFrame, p: np.ndarray, st: dict) -> tuple[np.ndarray, int, int]:
    """Departures without a plan at the orphan airports, joined to their flight's arrival record with a LOBT."""
    A = arr[arr.FLIGHT_ID_mvt.notna() & np.isfinite(arr.lobt)][["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "t", "lobt"]].rename(columns={"t": "t_arr", "lobt": "lobt_arr"})
    O = dep[dep.FLIGHT_ID_mvt.isna() & dep.ADEP_mvt.isin(st["orphan_airports"])][[KEY, "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "t"]]
    J = O.merge(A, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"]); J = J[(J.t_arr - J.t > 600) & (J.t_arr - J.t < 72000)].sort_values("t_arr").drop_duplicates(KEY)
    out = p.copy()
    if not len(J):
        return out, 0, 0
    pos = pd.Series(np.arange(len(dep)), index=dep[KEY]).reindex(J[KEY]).to_numpy(); r = (J.t - J.lobt_arr).to_numpy()
    lo, up = r - st["half_width_s"], r + st["half_width_s"]; q = out[pos]; ins = np.isfinite(q) & (q > lo) & (q < up)
    out[pos[ins]] = np.where(q[ins] - lo[ins] < up[ins] - q[ins], np.maximum(lo[ins], 60), up[ins])
    return out, int(len(J)), int(ins.sum())


def fit(oof: pd.DataFrame, cfg: dict | None = None, column: str = "final") -> dict:
    """Loadings from a 2025 out-of-fold prediction file (KEY + column) and the training files of configs/paths.json."""
    st = settings(cfg); E = daily_table(st)
    if E is None:
        return {"status": "skipped: EUROCONTROL daily table not found", "loadings": {}}
    files = sorted(Path(paths()["organiser"]).glob("training_2025-*.parquet")); M = movements(files)
    dep = M[M.PHASE_mvt == "DEP"].reset_index(drop=True)
    lab = pd.concat([pd.read_parquet(f, columns=[KEY, LABEL]) for f in files]).drop_duplicates(KEY)
    D = dep.merge(lab, on=KEY, how="left").merge(oof[[KEY, column]].rename(columns={column: "p"}), on=KEY, how="left")
    p0 = D.p.to_numpy(float); y = D[LABEL].to_numpy(float); p1, shift, cover = reconcile(D, p0, E, st)
    b = fit_loadings(D.d.to_numpy(float), p0, y, shift, st); m = np.isfinite(p0)
    r = lambda q: float(np.sqrt(np.nanmean((y[m] - q[m]) ** 2)))
    p2 = loaded_shift(D.d.to_numpy(float), p0, p1, shift, b, st); p3, n_rec, n_mv = orphan_window(D, M[M.PHASE_mvt == "ARR"], p2, st)
    return {"status": "fitted", "rows": int(m.sum()), "loadings": {str(k): v for k, v in b.items()},
            "oof_rmse": {"before": r(p0), "reconciled": r(p1), "loaded": r(p2), "orphan": r(p3)}, "share_on_reconciled_days": cover,
            "orphans_recovered": n_rec, "orphans_moved": n_mv, "labels_inside_orphan_window": 0 if n_rec == 0 else None}


def apply(sub: pd.DataFrame, fitted: dict, cfg: dict | None = None, organiser_file: str = "ranking.parquet",
          steps: tuple[str, ...] = ("reconcile", "loaded", "orphan")) -> tuple[pd.DataFrame, dict]:
    """The steps on a submission frame (KEY, LABEL); its rows must be departures of <organiser>/<organiser_file>."""
    st = settings(cfg); E = daily_table(st); b = {int(k): v for k, v in (fitted.get("loadings") or {}).items()}
    if E is None:
        return sub, {"status": "skipped: EUROCONTROL daily table not found"}
    M = movements([Path(paths()["organiser"]) / organiser_file]); dep = M[M.PHASE_mvt == "DEP"].reset_index(drop=True)
    D = dep.merge(sub[[KEY, LABEL]].rename(columns={LABEL: "p"}), on=KEY, how="inner")
    p0 = D.p.to_numpy(float); p1, shift, cover = reconcile(D, p0, E, st)
    if st.get("round_between_steps"):                 # the submitted file was built from the rounded reconciled file: the day
        p1 = np.round(p1); moved = p0 - p1            # shift is the mean rounded move of the day's moved departures
        day = pd.Series(pd.to_datetime(D.t.to_numpy() - p0, unit="s").strftime("%Y-%m-%d"))
        shift = pd.Series(np.where(moved != 0, moved, np.nan)).groupby([D.ADEP_mvt.to_numpy(), day.to_numpy()]).transform("mean").fillna(0).to_numpy()
    p2 = loaded_shift(D.d.to_numpy(float), p0, p1, shift, b, st) if (b and "loaded" in steps) else p1
    p3, n_rec, n_mv = orphan_window(D, M[M.PHASE_mvt == "ARR"], p2, st) if "orphan" in steps else (p2, 0, 0)
    out = sub[[KEY]].merge(pd.DataFrame({KEY: D[KEY].to_numpy(), LABEL: np.clip(np.round(p3), 0, None)}), on=KEY, how="left")
    out[LABEL] = out[LABEL].fillna(sub[LABEL]).astype(sub[LABEL].dtype)
    moved = np.abs(out[LABEL].to_numpy(float) - sub[LABEL].to_numpy(float))
    rep = {"status": "applied", "rows": int(len(D)), "share_on_reconciled_days": cover, "loadings_used": bool(b), "orphans_recovered": n_rec,
           "orphans_moved": n_mv, "rows_moved": int((moved >= 1).sum()), "mean_abs_move_s": float(moved.mean()), "mean_before": float(sub[LABEL].mean()), "mean_after": float(out[LABEL].mean())}
    return out, rep


def main() -> None:
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("--oof", required=True); f.add_argument("--out", required=True); f.add_argument("--column", default="final")
    a_ = sp.add_parser("apply"); a_.add_argument("--file", required=True); a_.add_argument("--fit", required=True); a_.add_argument("--out", required=True)
    a_.add_argument("--organiser-file", default="ranking.parquet")
    a = ap.parse_args()
    if a.cmd == "fit":
        rep = fit(pd.read_parquet(a.oof, columns=[KEY, a.column]), None, a.column); Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    else:
        sub = pd.read_parquet(a.file); out, rep = apply(sub, json.loads(Path(a.fit).read_text(encoding="utf-8")), None, a.organiser_file)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out, index=False)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
