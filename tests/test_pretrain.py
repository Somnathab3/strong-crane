"""Pre-training checks: every feature family and every stage, on real rows, before an expensive run.

    python -m pytest tests/test_pretrain.py -s -q          (about 10 minutes; evidence in out/pretrain_report.md)

Three kinds of test:
* correctness: a feature recomputed by brute force from the raw movements for sample rows must equal the pipeline's;
* safety: no feature may read a label (rebuild with scrambled labels; static scan of the builders);
* usability: a feature family must be plausible (ranges, units, cross-source agreement) and carry signal (it must
  relate to the label on ordinary rows), so it informs the models rather than confusing them.
The stages then run end to end on the slice with tiny models, and the bundle must round-trip.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, SLICE_AIRPORT, SLICE_DAYS

K = "MVT_ID_mvt"
RNG = np.random.default_rng(0)


def sample(df: pd.DataFrame, n: int, mask=None) -> pd.DataFrame:
    d = df if mask is None else df[mask]
    return d.sample(min(n, len(d)), random_state=0)


def table(df: pd.DataFrame, n: int = 12) -> str:
    return df.head(n).to_string()


# ----------------------------------------------------------------------------------------------------- data + field filling
def test_01_slice_and_labels(fs, report):
    """Slice present; keys unique; ranking rows carry no label; training labels equal the raw organiser value."""
    from taxiout.config import paths
    assert fs[K].is_unique
    by_day = fs.groupby(["day", "set"]).size().unstack(fill_value=0)
    assert set(fs.day) == set(SLICE_DAYS), f"missing days {set(SLICE_DAYS) - set(fs.day)}"
    assert fs.loc[fs["set"] == "rank", "label"].isna().all(), "a ranking row carries a label"
    org = paths()["organiser"]
    raw = pd.concat([pd.read_parquet(p, columns=[K, "TAXITIME_SEC_mvt"]) for p in sorted(org.glob("training_2025-*.parquet"))])
    tr = fs[fs["set"] == "train"].merge(raw, on=K, how="left")
    assert np.allclose(tr.label, tr.TAXITIME_SEC_mvt, equal_nan=True), "a training label differs from the organiser file"
    report("01 slice and labels", f"{len(fs):,} {SLICE_AIRPORT} departures; per day and set:\n{by_day.to_string()}\n"
           f"training labels equal the organiser's on all {len(tr):,} rows; ranking rows carry none")


def test_02_field_filling_touches_only_positional_fields(raw_movements, movements, report):
    """Field filling may change only type, wake, market, flight type and runway spelling (and adds source columns)."""
    allowed = {"AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "RUNWAY_mvt"}
    changed = {}
    for c in raw_movements.columns:
        a, b = raw_movements[c], movements[c]
        same = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        n = int((~same).sum())
        if n:
            changed[c] = n
    assert set(changed) <= allowed, f"field filling changed non-positional columns: {set(changed) - allowed}"
    ex = movements[(movements.airport == "LIRF") & movements.IOBT_flt.isna() & (movements.PHASE_mvt == "DEP") & (movements.type_src > 0)]
    report("02 field filling", f"columns changed (rows): {changed}\nnew columns: AIRCRAFT_TYPE_raw, type_src, market_src\n"
           f"example LIRF no-NM rows after field filling:\n{table(ex[[K, 'FLIGHT_mvt', 'AIRCRAFT_TYPE_raw', 'AIRCRAFT_TYPE_mvt', 'type_src', 'WK_TBL_CAT_flt', 'MARKET_SEGMENT_flt', 'market_src']], 8)}")


# ----------------------------------------------------------------------------------------------------- leakage
def test_03_features_do_not_read_labels(ms, clocks, report):
    """Rebuild the slice's features with the departures' labels scrambled: every feature must be identical."""
    from taxiout import features
    a = features.build(ms, clocks)
    m2 = ms.copy(); dep = (m2.PHASE_mvt == "DEP").to_numpy()
    m2.loc[dep, "label"] = RNG.permutation(m2.loc[dep, "label"].to_numpy()) * 1.7 + 11
    b = features.build(m2, clocks)
    cols = [c for c in a.columns if c != "label"]
    assert list(a.columns) == list(b.columns)
    diff = [c for c in cols if not a[c].equals(b[c])]           # NaN-aware (as text, pandas 3 never matches a missing value)
    assert not diff, f"features change when the labels change: {diff}"
    report("03 no label leakage (rebuild)", f"{len(a):,} departures rebuilt twice ({len(cols)} feature columns); scrambling the labels changed none of them")


def test_04_builders_never_read_the_label(report):
    """Static scan of the code (not the prose): outside the loader, no feature builder touches the label, the block time
    or TAXITIME as a column name, attribute or variable, except the pass-through that carries the label to training."""
    import ast
    files = ["fields.py", "features.py", "queue_arrivals.py", "weather.py", "eurocontrol.py", "adsb.py", "build_extra.py"]
    banned = {"label", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"}
    hits = []
    for f in files:
        tree = ast.parse((REPO / "src" / "taxiout" / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = (node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else
                    node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None)
            if name in banned:
                hits.append(f"{f}:{node.lineno}: {name}")
    src = {f: (REPO / "src" / "taxiout" / f).read_text(encoding="utf-8").splitlines() for f in files}
    allowed = [h for h in hits if re.search(r'KEY, "set", "month", "label", "t"', src[h.split(":")[0]][int(h.split(":")[1]) - 1])]
    bad = [h for h in hits if h not in allowed]
    report("04 static scan", f"label / block time / TAXITIME in builder code: {len(hits)} uses, all the pass-through to training: {not bad}\n"
           + ("\n".join(bad) or "no other use"))
    assert not bad


def test_05_no_feature_equals_the_label(fs, report):
    """No feature may reproduce the label on a large share of training rows (the schedule gap on stamped rows is expected)."""
    tr = fs[(fs["set"] == "train") & np.isfinite(fs.label)]
    num = [c for c in tr.columns if c not in ("label", "t", "month", K) and pd.api.types.is_numeric_dtype(tr[c])]
    share = {c: float(np.mean(np.abs(tr[c].to_numpy(float) - tr.label.to_numpy()) <= 1)) for c in num}
    top = sorted(share.items(), key=lambda x: -x[1])[:8]
    report("05 features equal to the label", "share of training rows where the feature equals the label within 1 s (top 8):\n"
           + "\n".join(f"  {c}: {v:.2%}" for c, v in top))
    bad = {c: v for c, v in share.items() if v > 0.2 and c not in ("sched",)}
    assert not bad, f"features reproduce the label: {bad}"


# ----------------------------------------------------------------------------------------------------- recompute by hand
def test_06_nm_clocks_and_schedule(fs, movements, report):
    from taxiout.features import NM_HI, NM_LO
    s = sample(fs, 40, (fs.has_nm > 0).to_numpy()).merge(movements[[K, "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "SCHED_TIME_UTC_mvt"]], on=K)
    for name, col in (("a3", "AOBT_3_flt"), ("lobt", "LOBT_flt"), ("eobt", "EOBT_1_flt"), ("iobt", "IOBT_flt")):
        x = s.t - s[col]; exp = np.where((x >= NM_LO) & (x <= NM_HI), x, np.nan)
        assert np.allclose(s[f"nm_{name}"], exp, equal_nan=True), f"nm_{name} wrong"
    assert np.allclose(s.sched, s.t - s.SCHED_TIME_UTC_mvt), "sched wrong"
    report("06 NM clocks and schedule gap", f"40 NM rows recomputed from the raw clocks: nm_a3 / nm_lobt / nm_eobt / nm_iobt / sched all equal\n"
           f"{table(s[[K, 'nm_a3', 'nm_iobt', 'nm_b', 'nm_src', 'sched', 'label']], 6)}")


def test_07_traffic_counts_bruteforce(fs, movements, report):
    dep = movements[(movements.PHASE_mvt == "DEP") & (movements.airport == SLICE_AIRPORT)]
    arr = movements[(movements.PHASE_mvt == "ARR") & (movements.airport == SLICE_AIRPORT)]
    s = sample(fs, 30).merge(movements[[K, "RUNWAY_mvt"]], on=K); bad = []
    tt, dr, at = np.round(dep.t.to_numpy()), dep.RUNWAY_mvt.astype(str).to_numpy(), np.round(arr.t.to_numpy())
    ib = arr.t.to_numpy() + arr.taxi_in.to_numpy()
    for r in s.itertuples():
        t = round(r.t)
        exp = {"dep_before_30": np.sum((tt >= t - 1800) & (tt < t)), "dep_after_30": np.sum((tt >= t + 1) & (tt < t + 1801)),
               "rwy_dep_before_30": np.sum((tt >= t - 1800) & (tt < t) & (dr == str(r.RUNWAY_mvt))),
               "arr_before_30": np.sum((at >= t - 1800) & (at < t)),
               "arr_done_30": np.sum(np.isfinite(ib) & (np.round(ib) >= t - 1800) & (np.round(ib) < t))}
        for k, v in exp.items():
            if getattr(r, k) != v:
                bad.append((r.MVT_ID_mvt if hasattr(r, "MVT_ID_mvt") else None, k, getattr(r, k), v))
    report("07 traffic counts (brute force)", f"30 rows x 5 counts recomputed from the movements; mismatches: {bad[:10] or 'none'}")
    assert not bad


def test_08_queue_bruteforce(fs, movements, report):
    """q_runway and rb_queue_n: departures on our runway that NM believes off-block but not yet airborne at our take-off."""
    dep = movements[(movements.PHASE_mvt == "DEP") & (movements.airport == SLICE_AIRPORT)].copy()
    dep["rw"] = dep.RUNWAY_mvt.astype(str)
    s = sample(fs, 40).merge(dep[[K, "rw", "AOBT_3_flt"]], on=K); rows = []
    for r in s.itertuples():
        t = round(r.t); o = dep[(dep.rw == r.rw) & (dep[K] != r.MVT_ID_mvt)]
        a3 = np.round(o.AOBT_3_flt.to_numpy()); tj = np.round(o.t.to_numpy())
        valid = np.isfinite(a3) & (a3 <= tj)
        q = int(np.sum(valid & (a3 < t) & (tj >= t)))
        rb = int(np.sum(np.isfinite(a3) & (a3 < t) & (tj >= t) & (t - a3 <= 7200)))
        rows.append((r.q_runway, q, r.rb_queue_n, rb))
    d = pd.DataFrame(rows, columns=["q_runway", "brute_q", "rb_queue_n", "brute_rb"])
    exact_q = np.mean(d.q_runway == d.brute_q); near_rb = np.mean(np.abs(d.rb_queue_n - d.brute_rb) <= 1)
    report("08 runway queue (brute force)", f"q_runway exact on {exact_q:.0%} of 40 rows; rb_queue_n within 1 on {near_rb:.0%} "
           f"(ties in take-off second decide membership)\n{table(d, 8)}\n"
           f"rb_age_median vs label on training rows: spearman {fs[fs['set'] == 'train'][['rb_age_median', 'label']].corr('spearman').iloc[0, 1]:.3f}")
    assert exact_q >= 0.95 and near_rb >= 0.95


def test_09_arrival_context_bruteforce(fs, movements, report):
    arr = movements[(movements.PHASE_mvt == "ARR") & (movements.airport == SLICE_AIRPORT)]
    arr = arr[np.isfinite(arr.taxi_in) & (arr.taxi_in > 0) & (arr.taxi_in < 7200)]
    ib = np.round((arr.t + arr.taxi_in).to_numpy()); ti = arr.taxi_in.to_numpy(); lt = np.round(arr.t.to_numpy())
    bad = []
    for r in sample(fs, 30).itertuples():
        t = round(r.t); w = (ib >= t - 1800) & (ib < t)
        exp_n, exp_med = int(w.sum()), (float(np.median(ti[w])) if w.any() else np.nan)
        exp_l = int(np.sum((lt >= t - 600) & (lt <= t + 600)))
        if r.arr_done_30 != exp_n or not np.isclose(r.ac_ap_taxiin_med_30, exp_med, equal_nan=True) or r.ac_ap_landings_pm10 != exp_l:
            bad.append((r.arr_done_30, exp_n, r.ac_ap_taxiin_med_30, exp_med, r.ac_ap_landings_pm10, exp_l))
    since = fs[[c for c in fs.columns if c.endswith("since_last_s")]]
    report("09 arrival context (brute force)", f"30 rows: completed arrivals (30 min), their median taxi-in and landings +-10 min recomputed; "
           f"mismatches {bad[:5] or 'none'}\nrecency columns max {since.max().round(0).to_dict()} (must be <= 7,200 s)")
    assert not bad and (since.max().fillna(0) <= 7200).all()


# ----------------------------------------------------------------------------------------------------- external sources
def test_10_weather_plausible_and_consistent(fs, report):
    tr = fs
    ok = tr.mt_age.notna()
    assert (tr.mt_age[ok] >= 0).all() and (tr.mt_age[ok] <= 5400).all(), "a METAR report is from the future or older than 90 min"
    t_ok = tr.mt_tmpc.between(-30, 45) | tr.mt_tmpc.isna()
    both = tr.mt_tmpc.notna() & tr.om_temperature_2m.notna()
    dtemp = (tr.mt_tmpc - tr.om_temperature_2m)[both].abs()
    report("10 weather", f"METAR found for {ok.mean():.1%} (age 0-{tr.mt_age.max():.0f} s, never ahead of take-off); temperature in range on {t_ok.mean():.1%}\n"
           f"METAR vs Open-Meteo temperature: median |difference| {dtemp.median():.1f} C, 95th pct {dtemp.quantile(.95):.1f} C "
           f"(two independent sources agree)\nexample:\n{table(tr[['day', 'mt_age', 'mt_tmpc', 'om_temperature_2m', 'mt_vsby' if 'mt_vsby' in tr else 'mt_age', 'om_deicing']].drop_duplicates('day'), 10)}")
    assert t_ok.all() and dtemp.median() < 3.0


def test_11_eurocontrol_consistent(fs, report):
    g = fs.groupby("day").agg(organiser_departures=(K, "size"), ast_departures=("ast_departures", "first"),
                              eum_to_mean_s=("eum_to_mean_s", "first"), label_mean=("label", "mean"))
    ratio = g.organiser_departures / g.ast_departures
    const = fs.groupby(fs.day.str[:7]).eum_to_mean_s.nunique(dropna=True).max()
    report("11 EUROCONTROL", f"published departures vs the organiser's count per day (ratio should be near 1):\n{g.assign(ratio=ratio.round(2)).to_string()}\n"
           f"monthly statistic constant within each month: {const <= 1}\n"
           "note: ast_ uses same-day published aggregates and eum_ the month's published taxi-out mean - open data, not our labels")
    assert ratio.between(0.8, 1.25).all() and const <= 1


def test_12_adsb_clock_usable(fs, clocks, report):
    s = sample(fs, 30, (fs.adsb_any > 0).to_numpy()).merge(clocks[[K, "ob_first_t", "status"]], on=K)
    assert np.allclose(s.cp_first, s.t - s.ob_first_t, equal_nan=True), "cp_first is not take-off minus first point"
    h = fs[fs.adsb_any > 0]
    early = np.mean(h.cp_first.dropna() < -60)
    rev_after = np.mean((h.cp_rev <= h.cp_first + 1).dropna()) if h.cp_rev.notna().any() else np.nan
    tr = fs[(fs["set"] == "train") & np.isfinite(fs.label) & (fs.label < 7200)].copy()
    tr["band"] = np.select([tr.cp_rev.notna(), tr.first_dist_m < 100, tr.first_dist_m < 1000, tr.first_dist_m < 1500, tr.adsb_any > 0],
                           ["pushback seen", "first point < 100 m from stand", "< 1 km", "1-1.5 km", "far / fast"], "no ADS-B")
    q = tr.groupby("band").apply(lambda g: pd.Series({"rows": len(g), "label - clock median s": np.nanmedian(g.label - g.cp_first) if g.cp_first.notna().any() else np.nan,
                                                      "clock RMSE vs label": np.sqrt(np.nanmean((g.label - g.cp_first) ** 2)) if g.cp_first.notna().any() else np.nan}))
    report("12 ADS-B clock", f"cp_first = take-off - first point on 30 checked rows; first point after take-off on {early:.2%} of heard rows; "
           f"pushback no later than first point on {rev_after:.1%}\nhow well the clock alone describes the label, by quality band (training rows <= 2 h):\n{q.round(0).to_string()}")
    assert early < 0.01


def test_12b_pushback_validity(fs, report):
    """Rejected pushbacks are gone and flagged; what remains is physically possible and close to the label."""
    ok = fs.cp_rev.notna()
    assert (fs.cp_rev[ok] >= 120).all() and (fs.rev_gs[ok].fillna(0) <= 15).all() and (fs.rev_minus_first[ok].fillna(0) >= -1800).all()
    tr = fs[ok & (fs["set"] == "train") & (fs.label < 7200)]
    e = tr.label - tr.cp_rev
    report("12b pushback validity", f"pushbacks kept {int(ok.sum()):,}, rejected {int(fs.rev_invalid.sum()):,}; kept pushback clock vs label (training <= 2 h): "
           f"median {e.median():.0f} s, RMSE {np.sqrt(np.mean(e ** 2)):.0f} s, |error| > 15 min on {np.mean(np.abs(e) > 900):.2%}")
    assert np.sqrt(np.mean(e ** 2)) < 400


def test_13_trajectory_plausible(fs, report):
    t = fs[fs.td_points.fillna(0) > 0]
    time_ok = np.mean(t.td_moving_s + t.td_stationary_s <= t.td_span_s + 60)
    gs_ok = np.mean(t.td_mean_gs_moving.between(1, 45))
    end_ok = np.mean((t.td_last_stop_end_to_mvt_s.dropna() >= -30))
    tr = t[(t["set"] == "train") & (t.label < 7200)]
    rho = tr[["td_span_s", "td_stationary_s", "td_path_m", "label"]].corr("spearman").label.drop("label")
    report("13 taxi trajectory", f"{len(t):,} rows with a trajectory; moving + stationary <= span on {time_ok:.1%}; mean moving speed 1-45 kt on {gs_ok:.1%}; "
           f"last stop ends before take-off on {end_ok:.1%}\nspearman with the label (training, <= 2 h): {rho.round(3).to_dict()}")
    assert time_ok > 0.98 and gs_ok > 0.95 and end_ok > 0.98


def test_14_opdi_plausible(fs, report):
    o = fs[fs.op_cp_taxiway.notna()]
    rng = np.mean(o.op_cp_taxiway.between(0, 3600)) if len(o) else 1.0
    tr = o[(o["set"] == "train") & o.label.notna()]
    lower = np.mean(tr.op_cp_taxiway <= tr.label + 60) if len(tr) else np.nan
    report("14 OPDI", f"taxiway entry on {len(o):,} slice rows; within 0-1 h of take-off on {rng:.1%}; "
           f"no later than the label allows (a lower bound on the taxi-out) on {lower:.1%} of training rows")
    assert rng > 0.95 and (np.isnan(lower) or lower > 0.9)


def test_15_gate_filters(fs, report):
    from taxiout.model import gate_rows
    for which in ("nm", "no_nm"):
        fit, app = gate_rows(fs, which)
        fast = (fs.first_gs.fillna(0) > 40).to_numpy() & fs.cp_rev.isna().to_numpy() & (fs.adsb_ok == 0).to_numpy()
        far = (fs.first_dist_m.fillna(0) > 1500).to_numpy() & fs.cp_rev.isna().to_numpy() & (fs.adsb_ok == 0).to_numpy()
        assert not (app & (fast | far)).any(), f"gate {which} applied to a far / fast first point"
        assert not (fit & (fs.adsb_any.to_numpy() == 0)).any(), f"gate {which} fitted on a row without ADS-B"
    fit, app = gate_rows(fs, "nm")
    report("15 gate filters", f"NM gate: fitted on {fit.sum():,}, applied to {app.sum():,} of {int((fs.has_nm > 0).sum()):,} NM slice rows; "
           "never on a row without ADS-B, never on a far (> 1.5 km, not seen at stand) or fast (> 40 kt) first point")


# ----------------------------------------------------------------------------------------------------- hygiene + signal
FAM = {"NM clocks": ("nm_", "csp_"), "schedule": ("sched",), "traffic": ("dep_", "arr_", "rwy_"), "queue (NM proxy)": ("q_", "qc_"),
       "queue detail (rb_)": ("rb_",), "arrival context (ac_)": ("ac_",), "post-ops medians": ("pm_",), "stand turnaround": ("stand_turn_age",),
       "ADS-B clock": ("cp_", "first_", "rev_", "hold_s", "gap_before_s", "n_ground", "stand_seen", "assigned_"), "taxi trajectory": ("td_",),
       "OPDI": ("op_cp", "op_taxiway"), "METAR": ("mt_",), "Open-Meteo": ("om_", "omh_"), "airport state": ("ast_",), "EU monthly": ("eum_",),
       "stamp evidence": ("svc_",), "agreement": ("agr_",), "seasonal": ("sea_",), "wake sequence": ("wk_",), "calendar": ("hour", "dow", "doy")}


def test_16_value_hygiene(fs, report):
    num = [c for c in fs.columns if pd.api.types.is_numeric_dtype(fs[c]) and c not in ("label", "t", "month", K)]
    inf = [c for c in num if np.isinf(fs[c].to_numpy(float)).any()]
    X = fs[num]
    dup, seen = [], {}
    for c in num:
        h = pd.util.hash_pandas_object(X[c].fillna(-9.99e99), index=False).sum()
        if h in seen and X[c].equals(X[seen[h]]):
            dup.append((seen[h], c))
        seen.setdefault(h, c)
    miss = pd.DataFrame({"2025": fs[fs["set"] == "train"][num].isna().mean(), "2026": fs[fs["set"] == "rank"][num].isna().mean()})
    miss["shift"] = (miss["2026"] - miss["2025"]).abs()
    fam_shift = {f: miss.loc[[c for c in num if c.startswith(p)], "shift"].max() for f, p in FAM.items() if any(c.startswith(p) for c in num)}
    report("16 value hygiene", f"{len(num)} numeric features; infinite values in: {inf or 'none'}\nidentical column pairs: {dup or 'none'}\n"
           "largest change in missing share 2025 -> 2026 per family (coverage drift the models must survive):\n"
           + "\n".join(f"  {k}: {v:.0%}" for k, v in sorted(fam_shift.items(), key=lambda x: -x[1])))
    assert not inf


def test_17_family_signal(fs, report):
    """Every family must relate to the label on ordinary rows (|spearman| of its best feature); a family with none is noise."""
    tr = fs[(fs["set"] == "train") & (fs.label < 7200)]
    num = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c]) and c not in ("label", "t", "month", K)]
    rho = tr[num].corrwith(tr.label, method="spearman").abs()
    rows = []
    for f, p in FAM.items():
        cs = [c for c in num if c.startswith(p)]
        if cs:
            best = rho[cs].sort_values(ascending=False)
            rows.append((f, len(cs), best.index[0], round(float(best.iloc[0]), 3)))
    t = pd.DataFrame(rows, columns=["family", "features", "best feature", "|spearman|"]).sort_values("|spearman|", ascending=False)
    weak = t[t["|spearman|"] < 0.05]
    report("17 family signal (ordinary training rows of the slice)", t.to_string(index=False) + f"\nfamilies with no single-feature signal: {list(weak.family) or 'none'}")
    assert len(weak) <= 3


# ----------------------------------------------------------------------------------------------------- stages end to end
@pytest.fixture(scope="module")
def mini_stack(fs):
    from taxiout.config import Settings
    from taxiout.model import CATS
    from taxiout.pipeline import cv_stack, fit_stack, predict_stack
    s = Settings(); s.trees, s.lgb_trees, s.spec_trees, s.gate_trees, s.device, s.verbose = 150, 150, 100, 100, "CPU", 0
    s.expert_trees, s.expert_min_rows = 100, 800             # small enough that the slice exercises the airport expert
    f0 = fs.drop(columns=["day"])                         # the test's helper column must not reach the models
    tr = f0[f0["set"] == "train"].reset_index(drop=True); rk = f0[f0["set"] == "rank"].reset_index(drop=True)
    dev = tr[tr.month != 1].reset_index(drop=True); test = tr[tr.month == 1].reset_index(drop=True)
    both = pd.concat([tr, rk]); cat_levels = {c: sorted(both[c].astype(str).unique()) for c in CATS if c in both}
    cat_levels["anchor_group"] = ["mid", "near", "none", "nonm_adsb", "nonm_none", "rev", "stand"]
    folds = [tuple(int(m) for m in b) for b in np.array_split(sorted(dev.month.unique()), s.inner_blocks)]
    log = lambda *a: None
    oof, ch = cv_stack(dev, folds, s, cat_levels, log)
    b = fit_stack(dev, oof, ch, s, cat_levels, log)
    return {"b": b, "oof": oof, "ch": ch, "dev": dev, "test": test, "rk": rk, "pt": predict_stack(b, test), "pr": predict_stack(b, rk)}


def test_18_stages_run_and_improve(mini_stack, report):
    from taxiout.pipeline import STAGES
    ms_ = mini_stack; y = ms_["test"].label.to_numpy(); ordn = y < 7200
    for which in ("pt", "pr"):
        for k in STAGES:
            v = np.asarray(ms_[which][k]); assert np.isfinite(v).all() and (v >= 0).all(), f"stage {k} has NaN / negative output"
    rm = {k: (np.sqrt(np.mean((np.asarray(ms_["pt"][k]) - y) ** 2)), np.sqrt(np.mean((np.asarray(ms_["pt"][k]) - y)[ordn] ** 2))) for k in STAGES}
    t = pd.DataFrame(rm, index=["all", "labels <= 2 h"]).T.round(1)
    report("18 stages end to end (tiny models, EHAM slice; January day predicted by a stack fitted on the other days)",
           f"{len(ms_['dev']):,} development rows, {len(ms_['test']):,} test rows, {len(ms_['rk']):,} ranking rows: every stage finite and >= 0\n{t.to_string()}\n"
           f"choices: blend weight {ms_['ch']['blend_weight_model_A']:.2f}; airport expert share {ms_['ch'].get('expert_weights')}; "
           f"gates {({k: v['airports'] for k, v in ms_['ch']['gates'].items()})}")
    assert "experts" in ms_["pt"] and np.isfinite(np.asarray(ms_["pt"]["experts"])).all()
    assert t.loc["model_A", "labels <= 2 h"] < t.loc["anchor", "labels <= 2 h"], "model A does not improve on the anchor"
    assert t.loc["final", "labels <= 2 h"] <= t.loc["anchor", "labels <= 2 h"]


def test_19_regime_probability(mini_stack, report):
    """The stamp classifier answers only on candidate rows (no NM record, or the schedule > 30 min from the NM clock):
    an on-time flight's taxi equals movement - schedule anyway, so 'label = schedule gap' says nothing there."""
    from taxiout.model import regime_candidates
    p = np.asarray(mini_stack["pt"]["p_stamp"]); t = mini_stack["test"]
    assert ((p >= 0) & (p <= 1)).all()
    cand = regime_candidates(t); assert (p[~cand] == 0).all(), "a non-candidate row got a stamp probability"
    c = cand & (t.sched >= 3600).to_numpy(); st = (np.abs(t.label - t.sched) <= 180).to_numpy()
    report("19 stamp probability", f"p in [0, 1]; non-candidates 0 ({int((~cand).sum()):,} rows); candidates with a gap >= 1 h: {int(c.sum())} "
           f"(stamped {int((c & st).sum())}): mean p stamped {p[c & st].mean() if (c & st).any() else float('nan'):.2f}, "
           f"not stamped {p[c & ~st].mean() if (c & ~st).any() else float('nan'):.2f}")
    if (c & st).sum() >= 3 and (c & ~st).sum() >= 3:
        assert p[c & st].mean() > p[c & ~st].mean()


def test_20_bundle_round_trip(mini_stack, tmp_path, report):
    from taxiout.pipeline import load_bundle, predict_stack, save_bundle
    save_bundle(mini_stack["b"], tmp_path, {"test": True})
    p2 = predict_stack(load_bundle(tmp_path), mini_stack["rk"])
    same = np.allclose(p2["final"], mini_stack["pr"]["final"])
    report("20 bundle round trip", f"saved, reloaded and re-predicted {len(mini_stack['rk']):,} ranking rows: identical {same}")
    assert same


def test_20b_airport_expert_path(mini_stack, tmp_path, report):
    """Force an expert share for the slice airport: the expert is fitted, applied, saved and reloaded identically."""
    from taxiout.config import Settings
    from taxiout.model import CATS
    from taxiout.pipeline import fit_stack, load_bundle, predict_stack, save_bundle
    s = Settings(); s.trees, s.lgb_trees, s.spec_trees, s.gate_trees, s.device, s.verbose = 150, 150, 100, 100, "CPU", 0
    s.expert_trees = 100
    ch = dict(mini_stack["ch"]); ch["expert_weights"] = {SLICE_AIRPORT: 0.5}
    dev, rk = mini_stack["dev"], mini_stack["rk"]
    both = pd.concat([dev, rk]); cat_levels = {c: sorted(both[c].astype(str).unique()) for c in CATS if c in both}
    cat_levels["anchor_group"] = ["mid", "near", "none", "nonm_adsb", "nonm_none", "rev", "stand"]
    b = fit_stack(dev, mini_stack["oof"], ch, s, cat_levels, lambda *a: None)
    p1 = predict_stack(b, rk)
    moved = float(np.mean(np.abs(np.asarray(p1["experts"]) - np.asarray(p1["blend"])) > 0.5))
    save_bundle(b, tmp_path, {"test": True}); p2 = predict_stack(load_bundle(tmp_path), rk)
    same = np.allclose(p1["final"], p2["final"])
    report("20b airport expert path", f"expert share 0.5 at {SLICE_AIRPORT}: expert fitted, {moved:.0%} of ranking rows moved by it; "
           f"saved, reloaded and re-predicted identically: {same}")
    assert SLICE_AIRPORT in b["experts"] and moved > 0.5 and same


def test_20c_top_refit_reproduces_stack(mini_stack, report):
    """Refit-only use of pipeline.cv_top / fit_top: stages 7-8 refitted from the base stages' out-of-fold predictions must give the
    full stack's out-of-fold and test predictions exactly, and the anchor group derived from a row's evidence alone must
    equal the stack's."""
    from taxiout import neighbours
    from taxiout.anchor import evidence
    from taxiout.config import Settings
    from taxiout.pipeline import cv_top, fit_top, predict_base, predict_top
    s = Settings(); s.trees, s.lgb_trees, s.spec_trees, s.gate_trees, s.device, s.verbose = 150, 150, 100, 100, "CPU", 0
    s.expert_trees, s.expert_min_rows = 100, 800
    ms_ = mini_stack; dev, test = ms_["dev"], ms_["test"]; log = lambda *a: None
    folds = [tuple(int(m) for m in b) for b in np.array_split(sorted(dev.month.unique()), s.inner_blocks)]
    base = {k: ms_["oof"][k] for k in ("anchor", "model_A", "model_A_neighbours", "lightgbm", "blend", "experts")}
    base["group"] = evidence(dev)[0]
    same_group = bool((base["group"] == ms_["oof"]["group"]).all())
    nb = neighbours.features(dev, base["model_A"] - base["anchor"], base["model_A"])
    base_ch = {k: ms_["ch"][k] for k in ("blend_weight_model_A", "blend_weight_by_airport", "expert_weights")}
    oof_top, ch = cv_top(dev, folds, base, nb, base_ch, s, log)
    same_oof = np.allclose(oof_top["final"], ms_["oof"]["final"], equal_nan=True)
    b_top = fit_top(dev, oof_top, nb, ch, s, log)
    out, a, g, nbt = predict_base(ms_["b"], test)
    p = predict_top({**ms_["b"], **b_top}, test, out, a, g, nbt)
    same_test = np.allclose(p["final"], ms_["pt"]["final"])
    report("20c refit-only mode", f"anchor group from evidence equals the stack's: {same_group}; top stages refitted from the base "
           f"out-of-fold predictions: out-of-fold final identical {same_oof}, test final identical {same_test}")
    assert same_group and same_oof and same_test


# ----------------------------------------------------------------------------------------------------- LIRF-specific stages
def test_21_lirf_stage_and_day_shift(features, report):
    from taxiout import lirf_regime
    from taxiout.config import Settings
    from taxiout.pipeline import day_shift
    L = features[features.airport == "LIRF"]
    tr = L[L["set"] == "train"].reset_index(drop=True); rk = L[L["set"] == "rank"].reset_index(drop=True)
    st = lirf_regime.fit(tr, Settings(), log=lambda *a: None)
    base = np.full(len(rk), 1000.0); new, p = lirf_regime.apply(st, rk, base)
    pop = lirf_regime.population(rk); g = rk.sched.to_numpy()
    assert (new[~pop] == base[~pop]).all(), "stage touched rows outside its population"
    assert np.allclose(new[pop & (g >= 8 * 3600) & (g < 14 * 3600)], g[pop & (g >= 8 * 3600) & (g < 14 * 3600)])
    assert (new[pop & (g >= 14 * 3600)] == base[pop & (g >= 14 * 3600)]).all()
    assert np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    fire = (features.has_nm > 0) & (features.nm_iobt.fillna(0) > 20 * 3600)
    ds_ = features[fire]; out = day_shift(ds_, np.zeros(len(ds_)))
    assert np.allclose(out, ds_.nm_iobt.to_numpy()), "the day-shift rule does not predict take-off - IOBT"
    tr_ = ds_["set"].to_numpy() == "train"
    report("21 LIRF stage and day-shift rule", f"LIRF no-NM >= 1 h 2026 rows {int(pop.sum())}: changed only inside the population; 8-14 h = gap; >= 14 h untouched\n"
           f"day-shift rule fires on {len(ds_)} rows ({int(tr_.sum())} in 2025) and predicts take-off - IOBT on all; "
           f"2025: prediction {out[tr_].round(0).tolist()} vs label {ds_.label[tr_].round(0).tolist()}")
    assert np.all(np.abs(out[tr_] - ds_.label.to_numpy()[tr_]) < 600)
