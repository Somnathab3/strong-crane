# SPDX-License-Identifier: GPL-3.0-only
"""Stage 7b: LIRF departures without an NM record that left an hour or more after schedule (28 Sep).

Their label is one of three things: the schedule gap itself (the block stamped at the scheduled time), 24 h plus a
taxi (the movement logged the day after the aircraft left) or an ordinary taxi. 2025 orders them by the gap:

| schedule gap | 2025 labels |
| --- | --- |
| 1-8 h | stamped (639) or ordinary (679) |
| 8-14 h | stamped (30 of 30) |
| 14-20 h | stamped (8) or day-shift (7) |
| >= 20 h | day-shift (4 of 4) |

Below 8 h a dedicated classifier (every stage-1 feature plus label-free LIRF traffic context; fold-honest AUC 0.93 on
2025, calibrated on its own out-of-fold probabilities) gives p(stamped), the prediction p * gap + (1 - p) * ordinary is
blended 50/50 with the stack's (the two err on opposite rows: the stack on ordinary and day-shift rows, the classifier
on stamped ones). From 8 to 14 h the prediction is the gap. From 14 h on nothing changes: the 2025 evidence there is 15 rows,
too few to move a prediction by a day.
Fold-honest on the Jan/Jul 2025 holdout this stage lowers these rows' squared error from 4.60e9 to 4.12e9
(holdout RMSE 293.99 -> 291.63).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

H = 3600.0
EXCL = {"MVT_ID_mvt", "set", "label", "t", "month", "has_nm", "airport", "taxi_in", "actype_raw", "market_raw", "flight_type_raw", "wake_raw"}


def context(f: pd.DataFrame) -> pd.DataFrame:
    """Label-free traffic around each LIRF departure (all LIRF departures in the frame, NM or not): departures, long
    delays and long no-NM delays within +-1 h and +-3 h, and long no-NM delays that day."""
    L = (f.airport == "LIRF").to_numpy(); out = pd.DataFrame(index=f.index)
    g = f[L]; o = np.argsort(g.t.to_numpy(), kind="stable"); t = g.t.to_numpy()[o]
    long_ = (g.sched.to_numpy()[o] >= H).astype(float); nonm = (g.has_nm.to_numpy()[o] == 0).astype(float)
    cols = {}
    for w in (1, 3):
        a = np.searchsorted(t, t - w * H, "left"); b = np.searchsorted(t, t + w * H, "right")
        for name, col in (("long", long_), ("nonm_long", long_ * nonm)):
            cs = np.r_[0, np.cumsum(col)]; cols[f"nx_{name}_{w}h"] = cs[b] - cs[a]
        cols[f"nx_n_{w}h"] = (b - a).astype(float)
    day = (t // 86400).astype(np.int64)
    cols["nx_day_nonm_long"] = pd.Series(long_ * nonm).groupby(day).transform("sum").to_numpy()
    idx = g.index[o]
    for k, v in cols.items():
        out.loc[idx, k] = v
    return out


def population(f: pd.DataFrame) -> np.ndarray:
    return ((f.airport == "LIRF") & (f.has_nm == 0) & (f.sched >= H)).to_numpy()


def _frame(f: pd.DataFrame, cols: list, cats: list) -> pd.DataFrame:
    x = pd.concat([f, context(f)], axis=1).reindex(columns=cols)
    for c in cats:
        x[c] = x[c].astype(str)
    return x


def _clf(s, k: int = 0):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=10, verbose=0, random_seed=s.seed + 97 * k,
                              thread_count=s.threads, allow_writing_files=False)


def _fit_avg(s, X, y, cats, weight=None) -> list:
    """v4: the classifier averaged over Settings.lirf_regime_seeds seeds (1: the single model of v3); v77: optional row
    weights (what a wrong call costs)."""
    return [_clf(s, k).fit(X, y, cat_features=cats, sample_weight=weight) for k in range(max(1, int(getattr(s, "lirf_regime_seeds", 1))))]


DAY = 86400.0
DS_BANDS = (("14-16", 14.0, 16.0), ("16-20", 16.0, 20.0), ("20+", 20.0, 1e9))


def crossing(f: pd.DataFrame) -> np.ndarray:
    """v77: could the real off-block have fallen on a later Rome date than the schedule? The off-block is unknown but lies
    5-40 min before the take-off: both bracket ends on a later local date than the schedule's. Label-free."""
    t = f.t.to_numpy(); g = f.sched.to_numpy()
    day = lambda sec: pd.to_datetime(sec, unit="s", utc=True).tz_convert("Europe/Rome").tz_localize(None).normalize().to_numpy()
    d_s = day(t - g)
    return (day(t - 300) > d_s) & (day(t - 2400) > d_s)


def _dayshift_table(tr: pd.DataFrame, pop: np.ndarray) -> dict:
    """v77: among fitting rows >= 14 h late that cross local midnight, the share of day-shift (label > 20 h) and stamped
    labels per gap band (smoothed +1 / +2), and the day-shift taxi (label - 24 h, median)."""
    y = tr.label.to_numpy(); g = tr.sched.to_numpy(); cr = crossing(tr); st = np.abs(y - g) <= 180; ds = y > 20 * H
    bands = {}
    for name, lo, hi in DS_BANDS:
        m = pop & cr & (g >= lo * H) & (g < hi * H); n = int(m.sum())
        p_ds = (float(ds[m].sum()) + 1) / (n + 2); p_st = min((float(st[m].sum()) + 1) / (n + 2), 1 - p_ds)
        bands[name] = [p_ds, p_st, n]
    taxi = float(np.median(y[pop & ds] - DAY)) if (pop & ds).any() else 1020.0
    return {"bands": bands, "taxi": taxi}


def _proba(models, X) -> np.ndarray:
    models = models if isinstance(models, list) else [models]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def fit(tr: pd.DataFrame, s, log=print, oof_pred: np.ndarray | None = None) -> dict | None:
    """oof_pred: the stack's out-of-fold prediction for every row of `tr` (before this stage); with
    Settings.lirf_regime_weight_auto the blend weight is chosen on it, otherwise the fixed lirf_regime_weight."""
    from sklearn.isotonic import IsotonicRegression
    pop = population(tr) & np.isfinite(tr.label.to_numpy())
    y = tr.label.to_numpy(); g = tr.sched.to_numpy(); st = np.abs(y - g) <= 180
    hard = pop & (g < 8 * H)
    if hard.sum() < 60 or st[hard].all() or not st[hard].any():
        log(f"  LIRF regime stage: {int(hard.sum())} rows below 8 h, not enough to fit; stage off"); return None
    ctx = context(tr)
    x_all = pd.concat([tr, ctx], axis=1)
    cats = [c for c in x_all.columns if c not in EXCL and (x_all[c].dtype == object or str(x_all[c].dtype) in ("string", "str", "category"))]
    cols = [c for c in x_all.columns if c not in EXCL and not c.startswith("nm_") and x_all.loc[pop, c].astype(str).nunique() > 1]
    cats = [c for c in cats if c in cols]
    X = _frame(tr[hard], cols, cats); t_ = st[hard].astype(int)
    o = pop & ~st & (y < 7200); ordinary = float(np.median(y[o])); w = float(s.lirf_regime_weight)
    wt = None
    if getattr(s, "lirf_regime_weighted", False):          # v77: a wrong call costs (gap - ordinary); weight the rows by its square
        wt = np.maximum(g[hard] - ordinary, 1.0) ** 2; wt = wt / wt.mean()
    months = tr.month.to_numpy()[hard]; blocks = np.array_split(np.unique(months), 4)
    pin = np.full(hard.sum(), np.nan)                      # the classifier's own out-of-fold probabilities, for calibration
    for b in blocks:
        te = np.isin(months, b)
        if te.any() and (~te).any() and 0 < t_[~te].mean() < 1:
            pin[te] = _proba(_fit_avg(s, X[~te], t_[~te], cats, None if wt is None else wt[~te]), X[te])
    ok = np.isfinite(pin)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pin[ok], t_[ok])
    m = _fit_avg(s, X, t_, cats, wt)
    if getattr(s, "lirf_regime_weight_auto", False) and oof_pred is not None:
        # the weight that minimises the out-of-fold squared error on these rows: the classifier's side from its own
        # out-of-fold (calibrated) probabilities, the stack's side from its out-of-fold predictions
        pc = np.interp(pin, iso.X_thresholds_, iso.y_thresholds_); gh = g[hard]; yh = y[hard]; sh = np.asarray(oof_pred)[hard]
        use = ok & np.isfinite(yh) & np.isfinite(sh)
        cls = pc * gh + (1 - pc) * ordinary
        sse = {round(w_, 1): float(np.sum((w_ * cls[use] + (1 - w_) * sh[use] - yh[use]) ** 2)) for w_ in np.linspace(0, 1, 11)}
        w = min(sse, key=sse.get)
        log(f"  LIRF regime stage: weight chosen out-of-fold {w:.1f} (SSE at 0.5: {sse[0.5]:.3e}, at {w:.1f}: {sse[w]:.3e}; {int(use.sum())} rows)")
    stage = {"clf": m, "iso_x": list(map(float, iso.X_thresholds_)), "iso_y": list(map(float, iso.y_thresholds_)),
             "ordinary": ordinary, "cols": cols, "cats": cats, "weight": w, "weighted": wt is not None}
    if getattr(s, "lirf_dayshift_ev", False):               # v77: expected value over day-shift / stamped / ordinary from 14 h on
        stage["dayshift"] = _dayshift_table(tr, pop)
        log("  LIRF day-shift expected value: " + ", ".join(f"{k} day-shift {v[0]:.2f} stamped {v[1]:.2f} ({v[2]} rows)" for k, v in stage["dayshift"]["bands"].items())
            + f"; day-shift taxi {stage['dayshift']['taxi']:.0f} s")
    auc_rows = ok.sum()
    log(f"  LIRF regime stage: classifier on {int(hard.sum())} rows below 8 h ({t_.mean():.0%} stamped{', weighted' if wt is not None else ''}; "
        f"calibrated on {auc_rows} out-of-fold probabilities); ordinary taxi {stage['ordinary']:.0f} s")
    return stage


def apply(stage: dict | None, new: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(adjusted prediction, p(stamped) where the classifier ran, else NaN)."""
    out = pred.copy(); p_out = np.full(len(new), np.nan)
    if stage is None:
        return out, p_out
    pop = population(new); g = new.sched.to_numpy()
    low = pop & (g < 8 * H); mid = pop & (g >= 8 * H) & (g < 14 * H)
    if low.any():
        raw = _proba(stage["clf"], _frame(new[low], stage["cols"], stage["cats"]))
        p = np.interp(raw, stage["iso_x"], stage["iso_y"]); p_out[low] = p
        w = stage["weight"]
        out[low] = w * (p * g[low] + (1 - p) * stage["ordinary"]) + (1 - w) * pred[low]
    out[mid] = g[mid]
    if stage.get("dayshift"):                               # v77: rows >= 14 h late that cross local midnight
        cr = crossing(new); tb = stage["dayshift"]
        for name, lo, hi in DS_BANDS:
            r = pop & cr & (g >= lo * H) & (g < hi * H)
            if r.any():
                p_ds, p_st, _ = tb["bands"][name]
                out[r] = p_ds * (DAY + tb["taxi"]) + p_st * g[r] + (1 - p_ds - p_st) * stage["ordinary"]
    return out, p_out


def save(stage: dict | None, d: Path) -> None:
    if stage is None:
        return
    models = stage["clf"] if isinstance(stage["clf"], list) else [stage["clf"]]
    for k, m in enumerate(models):
        m.save_model(str(d / f"lirf_regime_{k}.cbm"))
    (d / "lirf_regime.json").write_text(json.dumps({**{k: v for k, v in stage.items() if k != "clf"}, "n_models": len(models)}), encoding="utf-8")


def load(d: Path) -> dict | None:
    if not (d / "lirf_regime.json").exists():
        return None
    from catboost import CatBoostClassifier
    stage = json.loads((d / "lirf_regime.json").read_text(encoding="utf-8"))
    n = stage.get("n_models", 0)
    stage["clf"] = ([CatBoostClassifier().load_model(str(d / f"lirf_regime_{k}.cbm")) for k in range(n)] if n
                    else [CatBoostClassifier().load_model(str(d / "lirf_regime.cbm"))])          # bundles written before v4
    return stage
