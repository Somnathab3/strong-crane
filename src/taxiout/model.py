# SPDX-License-Identifier: GPL-3.0-only
"""Stage 4 (model A): one CatBoost model on the anchor's residual. Stage 8: the ADS-B gates on the stack's own residual.
Also the learners of stages 2 (stamp regime), 6 (LightGBM), 6b (airport experts) and 7 (no-NM specialist); the stage
numbers are those of pipeline.py.

Both learn "how wrong is the thing below me": stage 4 learns label - anchor (clipped +-7200 s), stage 8 learns
label - stage-7 prediction (clipped +-1800 s) on rows with ADS-B evidence, trained on stage 7's out-of-fold
predictions, so it sees the same kind of error it will correct on the ranking rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import Settings

CATS = ["airport", "runway", "stand", "stand_runway", "actype", "ap_actype", "wake", "operator", "market", "flight_type", "rule", "ades",
        "adsb_status", "anchor_group"]
#: feature families that can be left out for an ablation (--drop a,b): column-name prefixes
FAMILIES = {"weather": ("mt_", "om_"), "horizon": ("omh_",), "stamp": ("svc_", "sched_prev_day"), "airport_state": ("ast_",),
            "agreement": ("agr_",), "queue_comp": ("qc_",), "postops": ("pm_",), "spreads": ("csp_",), "seasonal": ("sea_",),
            "pairs": ("ap_actype",), "queue_detail": ("rb_",), "arrival_ctx": ("ac_",), "context": ("cx_",), "hotspots": ("hs_",), "hotspots2": ("hs2_",), "inbound": ("ib_",), "wake": ("wk_",), "taxi_dyn": ("td_",), "opdi": ("op_",), "eu_monthly": ("eum_",), "disruption": ("dis_",),
            #: "where the aircraft was first heard": the adsb.lol receiver network changed between 2025 and 2026
            #: (drift audit: LEMD first-point distance median 432 m -> 193 m, EGLL ground points 37 -> 94)
            "adsb_where": ("first_dist_m", "first_gs", "gap_before_s", "first_is_trace_start", "n_ground", "stand_seen_dist_m",
                           "assigned_dist_m", "adsb_status"),
            #: Open-Meteo fields whose 2026 distribution moved for model reasons, not weather (drift audit PSI > 0.25)
            "om_drift": ("om_visibility", "om_soil_moisture_0_to_1cm", "om_snow_depth", "om_freezing_level_height"),
            #: v7: Open-Meteo showers - missing for 12% of 2025 rows, never in 2026 (recording change, PSI 0.84)
            "om_showers": ("om_showers",)}
DROPPED: tuple = ()


def mask_families(f: pd.DataFrame, family_airports: dict) -> pd.DataFrame:
    """v4: a family used only at some airports (chosen where it helps in both held-out months, or by a label-free
    coverage rule): its columns are set missing at every other airport, for every stage alike."""
    if not family_airports:
        return f
    f = f.copy(); ap = f.airport.astype(str).to_numpy()
    for fam, aps in family_airports.items():
        cols = [c for c in f.columns if c.startswith(FAMILIES[fam])]
        off = ~np.isin(ap, list(aps))
        for c in cols:
            if pd.api.types.is_numeric_dtype(f[c]):
                f.loc[off, c] = np.nan
    return f


def set_dropped(families) -> None:
    global DROPPED
    DROPPED = tuple(p for f in families for p in FAMILIES[f])


def kept(cols) -> list:
    return [c for c in cols if not c.startswith(DROPPED)] if DROPPED else list(cols)
DROP = {"MVT_ID_mvt", "set", "month", "label", "t", "anchor", "actype_raw", "market_raw", "flight_type_raw", "wake_raw"}
CP = ["cp_first", "cp_rev", "cp_first_out", "cp_last_in", "cp_hold"]
GATE_RAW = ["adsb_ok", "first_dist_m", "first_gs", "gap_before_s", "first_is_trace_start", "n_ground", "hold_s", "stand_seen_dist_m",
            "assigned_dist_m", "rev_n", "rev_gs", "rev_dur", "rev_minus_first", "has_nm", "nm_b_ok", "nm_spread",
            # 28 Sep: how the aircraft moved between its start and the take-off (build_extra taxi), and OPDI's
            # taxiway entry (flight-level match); missing where there is no trace / no match
            "td_span_s", "td_path_m", "td_stationary_s", "td_stops", "td_last_stop_end_to_mvt_s", "td_mean_gs_moving", "td_end_gap_s", "td_turns",
            "op_cp_taxiway"]


def relative(f: pd.DataFrame, base: np.ndarray, prefix: str) -> pd.DataFrame:
    """ADS-B, NM and schedule durations as disagreements with a base prediction."""
    out = {f"{prefix}{c[3:]}": f[c].to_numpy() - base for c in CP}
    out[f"{prefix}b"] = f.nm_b.to_numpy() - base
    out[f"{prefix}sched"] = f.sched.to_numpy() - base
    return pd.DataFrame(out, index=f.index)


def learner(s: Settings, trees: int, depth: int, cats: list[str], lr: float | None = None, **extra) -> CatBoostRegressor:
    kw = dict(iterations=trees, depth=depth, learning_rate=lr or s.learning_rate, l2_leaf_reg=s.l2, loss_function="RMSE",
              random_seed=s.seed, verbose=s.verbose, cat_features=cats, task_type=s.device, allow_writing_files=False)
    kw.update(extra)
    if s.device == "CPU":
        kw["thread_count"] = s.threads
    return CatBoostRegressor(**kw)


def main_frame(f: pd.DataFrame, anchor: np.ndarray, group: np.ndarray, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage-4/5/6 inputs: every stage-1 feature, the anchor and its group, the disagreements with the anchor, and
    (stage 5 on) the runway-neighbour features."""
    x = f.drop(columns=[c for c in f.columns if c in DROP or (DROPPED and c.startswith(DROPPED))]).copy()
    x["anchor"] = anchor; x["anchor_group"] = group
    parts = [x, relative(f, anchor, "e_")] + ([extra] if extra is not None else [])
    return pd.concat(parts, axis=1)


def fit_main(f: pd.DataFrame, anchor, group, s: Settings, extra: pd.DataFrame | None = None) -> CatBoostRegressor:
    x = main_frame(f, anchor, group, extra)
    y = np.clip(f.label.to_numpy() - anchor, -s.residual_clip, s.residual_clip)
    ms = []
    for k in range(max(1, int(getattr(s, "main_seeds", 1)))):        # v7: averaged over seeds (one seed moves a month by ~3 RMSE)
        m = learner(s, s.trees, s.depth, [c for c in CATS if c in x], s.a_learning_rate, l2_leaf_reg=s.a_l2,
                    border_count=s.a_border_count, bagging_temperature=s.a_bagging_temperature, random_seed=s.seed + 1009 * k); m.fit(x, y); ms.append(m)
    return ms[0] if len(ms) == 1 else AvgModel(ms)


def fit_expert(f: pd.DataFrame, anchor, group, s: Settings, extra: pd.DataFrame | None = None) -> CatBoostRegressor:
    """Stage 6b: model A's inputs and target, fitted on one airport's rows only."""
    x = main_frame(f, anchor, group, extra)
    y = np.clip(f.label.to_numpy() - anchor, -s.residual_clip, s.residual_clip)
    m = learner(s, s.expert_trees, s.depth, [c for c in CATS if c in x], s.a_learning_rate, l2_leaf_reg=s.a_l2,
                border_count=s.a_border_count, bagging_temperature=s.a_bagging_temperature); m.fit(x, y)
    return m


def predict_main(m, f, anchor, group, extra: pd.DataFrame | None = None) -> np.ndarray:
    return np.clip(anchor + m.predict(main_frame(f, anchor, group, extra)), 0, None)


def seconds_per_metre(f: pd.DataFrame) -> dict:
    """Per airport: median (label - first-point duration) / distance, on first points 50-3,000 m from the stand
    (the time spent before the transponder came back on, per metre already covered)."""
    t = f[f.first_dist_m.between(50, 3000) & np.isfinite(f.label) & np.isfinite(f.cp_first)]
    return {a: float(np.median((g.label - g.cp_first) / g.first_dist_m)) for a, g in t.groupby("airport") if len(g) > 200}


def gate_frame(f: pd.DataFrame, pred: np.ndarray, spm: dict) -> pd.DataFrame:
    x = pd.concat([f[kept(["airport", "adsb_status"]) + [c for c in kept(GATE_RAW) if c in f]], relative(f, pred, "g_")], axis=1)
    x["pred"] = pred
    x["g_first_adj"] = x.g_first + f.airport.map(spm).fillna(0.0).to_numpy() * f.first_dist_m.fillna(0.0).to_numpy()
    return x


#: the gates (stage 8). "nm" and "no_nm" are separate models when Settings.gate_split (an earlier model's NM
#: and no-NM gates); "all" is one gate over both. The no-NM gate copies that design: 800 trees, depth 6, lr 0.05,
#: the schedule gap among its inputs (g_sched), predictions floored at 60 s.
GATES = {"nm": {"trees": None, "depth": None, "lr": None, "floor": 0.0},
         "no_nm": {"trees": 800, "depth": 6, "lr": 0.05, "floor": 60.0},
         "all": {"trees": None, "depth": None, "lr": None, "floor": 0.0},
         #: "rest": the NM rows no ADS-B gate reaches (no ground points, or the far/fast first point): a corrector on
         #: the out-of-fold stack prediction with every stage-1 feature, fitted on ordinary labels (<= TRIM_S) only and
         #: switched on per airport only where it lowers the out-of-fold error on those labels (peer tip, 28 Sep;
         #: aimed at LTFM/LFPG, which have no ground reception)
         "rest": {"trees": 800, "depth": 6, "lr": 0.05, "floor": 0.0}}
TRIM_S = 7200.0                 # "ordinary" label: at most two hours
FAR_M = 1500.0


def gate_rows(f: pd.DataFrame, which: str, genuine_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(fit rows, apply rows) of a gate. Fit on every row with ADS-B evidence (genuine_only: except rows whose block was
    stamped at the schedule, which an ADS-B clock cannot describe); apply except where the first point is > 1,500 m
    from the stand, not seen at it and no pushback was seen (the transponder came on mid-taxi: the old stack's
    far-point rule, which cross-month made those rows worse)."""
    nm = f.has_nm.to_numpy() > 0
    pop = f.adsb_any.to_numpy() > 0
    part = nm if which == "nm" else ~nm if which == "no_nm" else np.ones(len(f), bool)
    no_rev = ~np.isfinite(f.cp_rev.to_numpy())
    far = (f.first_dist_m.to_numpy() > FAR_M) & (f.adsb_ok.to_numpy() == 0) & no_rev
    far |= (np.nan_to_num(f.first_gs.to_numpy(dtype=float)) > 40.0) & no_rev & (f.adsb_ok.to_numpy() == 0)   # landing / fast taxi-in
    if which == "rest":
        rest = nm & (~pop | far); y = f.label.to_numpy() if "label" in f else np.full(len(f), np.nan)
        return rest & (np.nan_to_num(y, nan=np.inf) <= TRIM_S), rest
    fit = pop & part
    if genuine_only:
        y = f.label.to_numpy(); fit = fit & ~(np.isfinite(y) & (np.abs(y - f.sched.to_numpy()) <= 180))
    return fit, pop & part & ~far


def rest_frame(f: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return main_frame(f, pred, np.full(len(f), "stack", dtype=object))


def fit_gate(f: pd.DataFrame, pred: np.ndarray, s: Settings, which: str = "all"):
    g = GATES[which]; y = np.clip(f.label.to_numpy() - pred, -s.gate_clip, s.gate_clip)
    if which == "rest":
        x = rest_frame(f, pred); spm = {"_frame": "rest"}
        m = learner(s, g["trees"], g["depth"], [c for c in CATS if c in x], g["lr"]); m.fit(x, y)
        return m, spm
    spm = seconds_per_metre(f); x = gate_frame(f, pred, spm)
    m = learner(s, g["trees"] or s.gate_trees, g["depth"] or s.gate_depth, kept(["airport", "adsb_status"]), g["lr"]); m.fit(x, y)
    return m, spm


def predict_gate(model, f, pred) -> np.ndarray:
    m, spm = model
    if spm.get("_frame") == "rest":
        return m.predict(rest_frame(f, pred))
    return m.predict(gate_frame(f, pred, spm))


# ---- stage 2: the schedule-stamp regime (used by the anchor)
REGIME_CATS_ALL = ["airport", "runway", "stand", "actype", "operator", "market", "flight_type", "rule", "ades", "adsb_status"]
REGIME_NUM = ["sched", "sched_abs_log", "sched_band", "has_nm", "nm_src", "nm_spread", "nm_missing", "sched_minus_b", "hour_local", "dow",
              "month_num", "adsb_ok", "adsb_any", "cp_first", "cp_rev", "flight_dur_plan", "q_airport", "dep_before_60",
              "svc_doubled_prefix", "svc_malformed", "svc_per_month", "svc_dups_same_day", "svc_twin_per_month", "sched_prev_day",
              "op_rate", "op_rows", "op_rate_pop", "op_rows_pop"]
OP_PRIOR_ROWS = 20.0            # smoothing of the airline's stamped rate toward the overall rate
CAL_MIN_ROWS = 200              # calibration per airport where it has this many rows, else pooled


def month_blocks(f: pd.DataFrame, k: int) -> np.ndarray:
    """Block index 0..k-1 per row, by calendar month (months in order, dealt round-robin): a row never shares a
    block split with the rest of its own month."""
    months = sorted(pd.unique(f.month)); idx = {m: i % k for i, m in enumerate(months)}
    return f.month.map(idx).to_numpy()


def regime_candidates(f: pd.DataFrame) -> np.ndarray:
    """Rows where a schedule-stamped block would change the answer: no NM clock, or the schedule far from it."""
    nm = (f.has_nm.to_numpy() > 0) & np.isfinite(f.nm_b.to_numpy())
    far = np.abs(np.nan_to_num(f.sched_minus_b.to_numpy())) > 1800
    return (~nm | far) & (f.sched.to_numpy() > 60)


def stamped(f: pd.DataFrame) -> np.ndarray:
    return (np.abs(f.label.to_numpy() - f.sched.to_numpy()) <= 180).astype(int)


def _op_key(f: pd.DataFrame) -> pd.Series:
    return f.airport.astype(str) + "|" + f.operator.astype(str)


def _op_pop_key(f: pd.DataFrame) -> pd.Series:
    """The carrier at the airport, split by population: the schedule echo is mostly a carrier habit on flights without
    an NM match, so the same carrier's NM and no-NM flights get separate rates (peer tip, 28 Sep)."""
    return _op_key(f) + "|nm" + (f.has_nm.to_numpy() > 0).astype(int).astype(str)


def op_table(f: pd.DataFrame) -> tuple[dict, float]:
    """Stamped counts per airport|carrier and per airport|carrier|population, in one table (the keys never collide)."""
    st = stamped(f); out = {}
    for key in (_op_key(f), _op_pop_key(f)):
        g = pd.DataFrame({"k": key.to_numpy(), "s": st}).groupby("k").s.agg(["size", "sum"])
        out.update({k: (float(r["size"]), float(r["sum"])) for k, r in g.iterrows()})
    return out, float(st.mean()) if len(st) else 0.0


def _lookup(table: dict, keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    nk = [table.get(k, (0.0, 0.0)) for k in keys]
    return np.array([v[0] for v in nk]), np.array([v[1] for v in nk])


def with_op_rate(f: pd.DataFrame, table: dict, prior: float) -> pd.DataFrame:
    """op_rate: the carrier's stamped rate at the airport (smoothed toward the overall rate); op_rate_pop: the same for
    the row's population (NM or no-NM), smoothed toward op_rate, so a carrier seen in only one population falls back
    to its pooled rate."""
    x = f.copy(); n, k = _lookup(table, _op_key(f))
    x["op_rows"] = n; x["op_rate"] = (k + prior * OP_PRIOR_ROWS) / (n + OP_PRIOR_ROWS)
    n2, k2 = _lookup(table, _op_pop_key(f))
    x["op_rows_pop"] = n2; x["op_rate_pop"] = (k2 + x.op_rate.to_numpy() * OP_PRIOR_ROWS) / (n2 + OP_PRIOR_ROWS)
    return x


RAW_SWAP = {"actype": "actype_raw", "market": "market_raw", "flight_type": "flight_type_raw"}


def _classifier(s: Settings, seed: int | None = None):
    from catboost import CatBoostClassifier
    dev = getattr(s, "regime_device", s.device)
    kw = dict(iterations=300, depth=6, learning_rate=0.1, random_seed=s.seed if seed is None else seed, verbose=s.verbose,
              cat_features=kept(REGIME_CATS_ALL), allow_writing_files=False, task_type=dev)
    if dev == "CPU":
        kw["thread_count"] = s.threads
    return CatBoostClassifier(**kw)


def _regime_x(x: pd.DataFrame, raw_fields: bool) -> pd.DataFrame:
    """The stamp classifier's input columns; with raw_fields, type / market / flight type are the organiser's own values."""
    cols = kept(REGIME_CATS_ALL) + kept(REGIME_NUM)
    out = x[cols].copy()
    if raw_fields:
        for c, r in RAW_SWAP.items():
            if c in out and r in x:
                out[c] = x[r].to_numpy()
    return out


def fit_regime(f: pd.DataFrame, s: Settings) -> dict:
    """Classifier on labelled candidate rows. The airline's stamped rate enters as a feature: on the classifier's own
    training rows it is computed out-of-fold (five contiguous month blocks), so no row sees its own label."""
    x = f.copy(); blk = month_blocks(f, 5)                    # month blocks, not days: neighbouring days share flights
    OPC = ["op_rate", "op_rows", "op_rate_pop", "op_rows_pop"]
    for c in OPC:
        x[c] = np.nan
    for i in range(5):
        r = blk == i
        if r.any():
            tb, pr = op_table(f[~r]); y_ = with_op_rate(f[r], tb, pr)
            for c in OPC:
                x.loc[r, c] = y_[c].to_numpy()
    table, prior = op_table(f); y = stamped(f)
    if len(y) == 0 or y.min() == y.max():                  # one class only (small or single-airport samples): a constant
        return {"model": None, "const": float(y.mean()) if len(y) else 0.0, "op_table": table, "prior": prior, "cal": {}}
    raw = bool(getattr(s, "regime_raw_fields", False)); X = _regime_x(x, raw)
    models = []
    for k in range(max(1, int(getattr(s, "regime_seeds", 1)))):          # v4: averaged over seeds (less variance on the tail rows)
        m = _classifier(s, s.seed + 97 * k); m.fit(X, y); models.append(m)
    return {"model": models, "const": None, "op_table": table, "prior": prior, "cal": {}, "raw_fields": raw}


def regime_raw(R: dict, f: pd.DataFrame) -> np.ndarray:
    if R.get("model") is None:
        return np.full(len(f), float(R.get("const") or 0.0))
    X = _regime_x(with_op_rate(f, R["op_table"], R["prior"]), R.get("raw_fields", False))
    models = R["model"] if isinstance(R["model"], list) else [R["model"]]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def fit_calibration(p: np.ndarray, st: np.ndarray, airport: np.ndarray) -> dict:
    """Isotonic map from raw P to the observed stamped rate, per airport with >= CAL_MIN_ROWS rows, plus a pooled map."""
    from sklearn.isotonic import IsotonicRegression
    cal = {}
    for key, r in [("_pooled", np.ones(len(p), bool))] + [(a, airport == a) for a in np.unique(airport)]:
        if r.sum() >= CAL_MIN_ROWS and 0 < st[r].mean() < 1:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[r], st[r])
            cal[key] = (list(map(float, iso.X_thresholds_)), list(map(float, iso.y_thresholds_)))
    return cal


def calibrate(p: np.ndarray, airport: np.ndarray, cal: dict) -> np.ndarray:
    out = p.copy()
    for a in np.unique(airport):
        r = airport == a; c = cal.get(a, cal.get("_pooled"))
        if c:
            out[r] = np.interp(p[r], c[0], c[1])
    return out


def regime_fit(f: pd.DataFrame, s: Settings, fit_rows: np.ndarray, inner_folds: int = 2) -> tuple[np.ndarray, dict]:
    """Calibrated P(schedule-stamped) for every candidate row, and the fitted regime (classifier, airline table,
    calibration). Fitting rows get out-of-fold probabilities (by day blocks) and the calibration is fitted on those;
    other rows get the classifier fitted on all fitting rows. Non-candidates get 0."""
    cand = regime_candidates(f); p = np.zeros(len(f))
    tr = cand & fit_rows & np.isfinite(f.label.to_numpy())
    k = month_blocks(f, inner_folds)
    for i in range(inner_folds):
        a, b = tr & (k != i), tr & (k == i)
        if a.sum() > 200 and b.any():
            p[b] = regime_raw(fit_regime(f[a], s), f[b])
    R = fit_regime(f[tr], s)
    other = cand & ~tr
    if other.any():
        p[other] = regime_raw(R, f[other])
    ap = f.airport.to_numpy().astype(str)
    #: off by default: an isotonic map fitted on the development months copies their stamped rate (LIRF ~54%) onto
    #: months with a different mix (January 42%, July 33%); on the 27 Sep smoke it raised LIRF no-NM error 8,085 -> 9,013
    R["cal"] = fit_calibration(p[tr], stamped(f[tr]), ap[tr]) if s.calibrate_regime else {}
    p[cand] = calibrate(p[cand], ap[cand], R["cal"])
    return p, R


def regime_predict(R: dict, f: pd.DataFrame) -> np.ndarray:
    p = np.zeros(len(f)); c = regime_candidates(f)
    if c.any():
        p[c] = calibrate(regime_raw(R, f[c]), f.airport.to_numpy().astype(str)[c], R["cal"])
    return p


def regime_p(f: pd.DataFrame, s: Settings, fit_rows: np.ndarray, inner_folds: int = 2) -> np.ndarray:
    return regime_fit(f, s, fit_rows, inner_folds)[0]


# ---- stage 6: second learner (LightGBM on the same inputs), mixed with stage 5 by an out-of-fold weight
def to_lgb(x: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    x = x.copy()
    for c, lv in cat_levels.items():
        if c in x:
            x[c] = pd.Categorical(x[c].astype(str), categories=lv)
    return x


def fit_lgb(x: pd.DataFrame, y: np.ndarray, s: Settings, cat_levels: dict):
    import lightgbm as lgb
    #: Optuna 28 Sep (held-out months): 256.84 -> 250.68
    p = dict(objective="regression", learning_rate=0.0251, num_leaves=219, min_data_in_leaf=41, feature_fraction=0.553,
             bagging_fraction=0.688, bagging_freq=1, lambda_l2=10.49, max_bin=255, max_cat_to_onehot=8, cat_smooth=20,
             num_threads=s.threads, seed=s.seed, deterministic=True, force_row_wise=True, verbose=-1)
    cb = [lgb.log_evaluation(s.verbose)] if s.verbose else []
    d = lgb.Dataset(to_lgb(x, cat_levels), label=y, free_raw_data=True)
    return lgb.train(p, d, num_boost_round=s.lgb_trees, valid_sets=[d] if s.verbose else None, callbacks=cb)


def blend_weight(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """w minimising sum((w a + (1 - w) b - y)^2), clipped to [0, 1]."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y); d = a[ok] - b[ok]
    return float(np.clip(np.dot(d, y[ok] - b[ok]) / max(np.dot(d, d), 1e-9), 0, 1))


# ---- stage 7: the no-NM specialist (generalised from an earlier model's): a small model fitted only on no-NM rows,
# learning what the blend still gets wrong there; applied per airport where it lowers the out-of-fold error
SPEC_CLIP = 30000.0


class AvgModel:
    """v4.1: several fitted models whose predictions are averaged (a specialist fitted over several seeds)."""
    def __init__(self, models: list):
        self.models = models

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.models], axis=0)


def fit_specialist(x: pd.DataFrame, y_resid: np.ndarray, s: Settings, cats: list[str]):
    """v4.1: a specialist fitted on few rows (LIRF: ~1,400) is high-variance on a heavy-tailed target - v4's LIRF
    specialist moved some rows by hours when its inputs changed slightly. Below `spec_small_rows` rows the trees are
    shallower and more regularised, and every specialist is averaged over `spec_seeds` seeds."""
    small = len(x) < getattr(s, "spec_small_rows", 0)
    depth, l2 = (getattr(s, "spec_small_depth", 6), 10.0) if small else (6, 3.0)
    ms = []
    for k in range(max(1, int(getattr(s, "spec_seeds", 1)))):
        m = CatBoostRegressor(iterations=s.spec_trees, depth=depth, l2_leaf_reg=l2, learning_rate=0.05, loss_function="RMSE", random_seed=s.seed + 31 * k,
                              verbose=s.verbose, cat_features=cats, task_type=s.device, allow_writing_files=False,
                              **({"thread_count": s.threads} if s.device == "CPU" else {}))
        m.fit(x, np.clip(y_resid, -SPEC_CLIP, SPEC_CLIP)); ms.append(m)
    return ms[0] if len(ms) == 1 else AvgModel(ms)
