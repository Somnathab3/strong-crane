# SPDX-License-Identifier: GPL-3.0-only
"""Stage 3: the anchor. A closed-form, per-airport correction of the best clock (no trees, a few numbers per airport).

NM rows (a usable NM clock):   anchor = b + c0 + c1 * z + c2 * e + c3 * d
    b  = take-off minus the best NM clock (AOBT_3, else LOBT, EOBT_1, IOBT)
    z  = EOBT_1 gap minus b                          (the intercept/slope calibration of the first repository)
    e  = clip(ADS-B duration - b, +-900 s)           (0 in the "none" group)
    d  = first-point distance from the stand, km     (in the "near" and "mid" groups: the seconds-per-metre term)
    one set of coefficients per airport x ADS-B evidence band:
      "rev"   pushback seen (the backwards run)
      "stand" first point within 100 m of the assigned stand and <= 6 kt
      "near"  first point 100 m - 1 km from the stand
      "mid"   first point 1 - 1.5 km from the stand
      "none"  no ADS-B, or the first point > 1.5 km away with no pushback (transponder on mid-taxi: not trusted)
    The ridge decides how far each band is trusted: c2 near 1 means the anchor follows the ADS-B clock.
No-NM rows: "nonm_adsb" (pushback, stand or near): anchor = c0 + c1 * clip(ADS-B duration, 60, 7200); otherwise
    the airport's median ordinary taxi. Then (stage 2) p * schedule gap + (1 - p) * that, p = P(stamped at the schedule).
Fitted by ridge on the fitting rows only (per fold in cross-validation). Groups with < 200 rows use the pooled fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ROWS = 200
ALPHA = 10.0
NM_GROUPS = ("rev", "stand", "near", "mid", "none")
FAST_KT = 40.0


def _ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = X.T @ X + ALPHA * np.diag([0.0] + [1.0] * (X.shape[1] - 1))
    return np.linalg.solve(A, X.T @ y)


def evidence(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group name, ADS-B duration (take-off minus the event) and first-point distance (km) per row."""
    #: a first point faster than FAST_KT without a pushback is a landing or a fast taxi-in, not an off-block
    #: (EGLL: 41% of its remaining error came from such first points > 1.5 km at > 60 kt; LFPG: inbound landings)
    fast = (np.nan_to_num(f.first_gs.to_numpy(dtype=float)) > FAST_KT) & ~np.isfinite(f.cp_rev.to_numpy(dtype=float))
    first_ok = f.cp_first.between(30, 7200).to_numpy() & ~fast; dist = f.first_dist_m.to_numpy()
    rev = (np.isfinite(f.cp_rev) & f.cp_rev.between(30, 7200)).to_numpy()
    stand = ~rev & (f.adsb_ok.to_numpy() > 0) & first_ok
    near = ~rev & ~stand & first_ok & (dist <= 1000)
    mid = ~rev & ~stand & first_ok & (dist > 1000) & (dist <= 1500)
    cp = np.where(rev, f.cp_rev, np.where(stand | near | mid, f.cp_first, np.nan))
    nm = ((f.has_nm > 0) & np.isfinite(f.nm_b)).to_numpy()      # NM formula needs a usable clock; otherwise no-NM formulas
    band = np.select([rev, stand, near, mid], ["rev", "stand", "near", "mid"], "none")
    group = np.where(nm, band, np.where(rev | stand | near, "nonm_adsb", "nonm_none"))
    dkm = np.where(near | mid, np.nan_to_num(dist) / 1000.0, 0.0)
    return group, cp, dkm


def _clock_terms(f: pd.DataFrame) -> np.ndarray:
    """An earlier model's anchor inputs: every NM clock's gap to the chosen one, their overall spread and the
    schedule gap, each clipped, missing as 0 plus a missing flag. At LTFM, where take-off minus AOBT_3 alone scores 531 s
    on the 2025 holdout, these carry most of the anchor (318 s vs 362 s without them)."""
    c = lambda col, lim: np.clip(np.nan_to_num(f[col].to_numpy(dtype=float)), -lim, lim)
    return np.column_stack([c("nm_lobt_minus_b", 7200), c("nm_iobt_minus_b", 7200), np.clip(np.nan_to_num(f.nm_spread.to_numpy(dtype=float)), 0, 7200),
                            c("sched_minus_b", 10800), f.nm_lobt_minus_b.isna().to_numpy(float), f.nm_iobt_minus_b.isna().to_numpy(float)])


def _design(z, b, cp, dkm, g, extra=None):
    e = np.clip(np.nan_to_num(cp - b), -900, 900) if g != "none" else np.zeros(len(z))
    base = np.column_stack([np.ones(len(z)), z, e, dkm])
    return base if extra is None else np.column_stack([base, extra])


CLOCKS = ("nm_a3", "nm_lobt", "nm_eobt", "nm_iobt")


def clock_matrix(f: pd.DataFrame) -> np.ndarray:
    """An earlier model's anchor design, exactly: the four raw clock gaps (take-off minus AOBT_3, LOBT,
    EOBT_1, IOBT) and three spreads against AOBT_3, unclipped. Rows with any clock missing are NaN."""
    d = np.column_stack([f[c].to_numpy(dtype=float) for c in CLOCKS])
    return np.column_stack([d, d[:, 2] - d[:, 0], d[:, 3] - d[:, 0], d[:, 1] - d[:, 0]])


def _ridge_int(x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Ridge with the intercept last and unpenalised (as in the earlier model)."""
    d = np.hstack([x, np.ones((len(x), 1))]); p = alpha * np.eye(d.shape[1]); p[-1, -1] = 0.0
    return np.linalg.solve(d.T @ d + p, d.T @ y)


class ClockRidge:
    """The anchor: one ridge per (airport, runway) with >= MIN_ROWS rows, else per airport, else global, fitted on
    every labelled NM row whose four clocks exist. On the 2025 holdout it scores LTFM 273 / 352 (Jan / Jul) where
    the best clock alone scores 484 / 563: at airports whose AOBT_3 is unreliable the other clocks carry the anchor."""

    def fit(self, f: pd.DataFrame) -> "ClockRidge":
        x = clock_matrix(f); y = f.label.to_numpy(dtype=float)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y) & (f.has_nm.to_numpy() > 0)
        ap, rw = f.airport.astype(str).to_numpy(), f.runway.astype(str).to_numpy()
        self.m = {"__global__": _ridge_int(x[ok], y[ok])}
        for a in np.unique(ap[ok]):
            r = ok & (ap == a)
            if r.sum() >= MIN_ROWS:
                self.m[a] = _ridge_int(x[r], y[r])
            for w in np.unique(rw[r]):
                q = r & (rw == w)
                if q.sum() >= MIN_ROWS:
                    self.m[f"{a}|{w}"] = _ridge_int(x[q], y[q])
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        x = clock_matrix(f); d = np.hstack([x, np.ones((len(x), 1))])
        ap, rw = f.airport.astype(str).to_numpy(), f.runway.astype(str).to_numpy()
        out = np.full(len(f), np.nan)
        for (a, w), idx in pd.DataFrame({"a": ap, "w": rw}).groupby(["a", "w"]).indices.items():
            c = self.m.get(f"{a}|{w}", self.m.get(a, self.m["__global__"]))
            out[idx] = d[idx] @ c
        out[~np.isfinite(x).all(axis=1)] = np.nan
        return out


class Anchor:
    def __init__(self, rich: bool = True, fit_rows: str = "all", clock_ridge: bool = True):
        #: rich: earlier-model clock terms and a runway level ((airport, runway, band) when >= MIN_ROWS rows, else
        #: (airport, band), else the pooled band); False: the 27 Sep anchor (best clock + EOBT gap, per airport x band)
        self.rich = rich; self.fit_rows = fit_rows; self.clock_ridge = clock_ridge

    def fit(self, f: pd.DataFrame) -> "Anchor":
        group, cp, dkm = evidence(f)
        y = f.label.to_numpy(); b = f.nm_b.to_numpy(); ap = f.airport.to_numpy(); z = f.nm_z.to_numpy()
        rw = f.runway.astype(str).to_numpy(); ex = _clock_terms(f) if self.rich else None
        if getattr(self, "clock_ridge", False):
            #: the clock ridge as the NM base (where all four clocks exist); the band terms then only add ADS-B
            self.cr = ClockRidge().fit(f); br = self.cr.predict(f)
            b = np.where(np.isfinite(br) & np.isfinite(b), br, b); ex = None
        ok = np.isfinite(y) & (y >= 0) & (y <= 7200)
        self.coef, self.median, self.rows = {}, {}, {}
        #: which rows the NM ridge is fitted on. "all": every labelled NM row up to a day, so the flights whose
        #: AOBT_3 is badly off teach the ridge to lean on the other clocks (RMSE squares exactly those errors);
        #: "robust": only labels within 1 h of the clock (the 27 Sep anchor)
        mode = getattr(self, "fit_rows", "all")
        ok_all = np.isfinite(y) & (y >= 0) & (y <= 86400)
        nm_ok = (ok_all & np.isfinite(b)) if mode == "all" else (ok & np.isfinite(b) & (np.abs(y - b) < 3600))
        for g in NM_GROUPS:
            X = _design(z, b, cp, dkm, g, ex); rows = nm_ok & (group == g)
            pooled = _ridge(X[rows], (y - b)[rows]) if rows.sum() >= MIN_ROWS else np.zeros(X.shape[1])
            for a in np.unique(ap):
                r = rows & (ap == a); self.rows[(a, g)] = int(r.sum())
                self.coef[(a, g)] = _ridge(X[r], (y - b)[r]) if r.sum() >= MIN_ROWS else pooled
                if self.rich:
                    for w in np.unique(rw[r]):
                        q = r & (rw == w)
                        if q.sum() >= MIN_ROWS:
                            self.coef[(a, w, g)] = _ridge(X[q], (y - b)[q]); self.rows[(a, w, g)] = int(q.sum())
        nn = ok & ~np.isfinite(b)
        X = np.column_stack([np.ones(len(f)), np.clip(np.nan_to_num(cp), 60, 7200)])
        rows = nn & (group == "nonm_adsb")
        pooled = _ridge(X[rows], y[rows]) if rows.sum() >= MIN_ROWS else np.array([900.0, 0.0])
        glob = float(np.median(y[nn])) if nn.any() else 1000.0
        for a in np.unique(ap):
            r = rows & (ap == a); self.rows[(a, "nonm_adsb")] = int(r.sum())
            self.coef[(a, "nonm_adsb")] = _ridge(X[r], y[r]) if r.sum() >= MIN_ROWS else pooled
            q = nn & (ap == a)
            self.median[a] = float(np.median(y[q])) if q.sum() >= 50 else glob
        self.glob = glob
        return self

    def predict(self, f: pd.DataFrame, p_stamp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """p_stamp: P(block stamped at the schedule) per row (stage 2); the anchor becomes the expected value
        p * (take-off - schedule) + (1 - p) * ordinary anchor, the RMSE-optimal choice under that uncertainty."""
        out, group = self._ordinary(f)
        if p_stamp is not None:
            s = np.clip(f.sched.to_numpy(), 0, None)
            out = np.where(p_stamp > 0, p_stamp * s + (1 - p_stamp) * out, out)
        return out, group

    def _ordinary(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        group, cp, dkm = evidence(f)
        b = f.nm_b.to_numpy(); ap = f.airport.to_numpy(); z = f.nm_z.to_numpy(); rw = f.runway.astype(str).to_numpy()
        rich = getattr(self, "rich", False); ex = _clock_terms(f) if rich else None
        if getattr(self, "clock_ridge", False):
            br = self.cr.predict(f); b = np.where(np.isfinite(br) & np.isfinite(b), br, b); ex = None
        out = np.full(len(f), np.nan); done = np.zeros(len(f), bool)
        # most specific level first: (airport, runway, band), then (airport, band)
        for key in sorted(self.coef, key=len, reverse=True):
            c = self.coef[key]; a, g = key[0], key[-1]
            r = (ap == a) & (group == g) & ~done
            if len(key) == 3:
                r &= rw == key[1]
            if not r.any():
                continue
            if g == "nonm_adsb":
                out[r] = c[0] + c[1] * np.clip(cp[r], 60, 7200)
            else:
                out[r] = b[r] + _design(z[r], b[r], cp[r], dkm[r], g, ex[r] if ex is not None else None) @ c
            done |= r
        miss = np.isnan(out)
        out[miss] = [self.median.get(a, self.glob) for a in ap[miss]]
        return np.clip(out, 0, None), group

    def table(self) -> dict:
        return ({"|".join(k): {"coef": [round(float(x), 4) for x in c], "rows": self.rows.get(k)} for k, c in self.coef.items()}
                | {f"{a}|median": v for a, v in self.median.items()})
