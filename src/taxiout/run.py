# SPDX-License-Identifier: GPL-3.0-only
"""One command from the raw inputs to an honest holdout score, a submission file and a fitted bundle.

    python -m taxiout.run --months 7          # development pass + submission pass, GPU (the shipped design, src/v77/v77.yaml)
    python -m taxiout.run --months 7 --dev-only   # development pass only (the honest July 2025 score)
    python -m taxiout.run --cpu               # every CatBoost fit on the CPU with fixed threads: byte-reproducible
    python -m taxiout.run --quick             # smoke run: 15% of the days, few trees

Stages (each learns what the stages below it still get wrong):
  2 regime P(schedule-stamped)   3 anchor (per-airport formula + ADS-B evidence + regime mix)
  4 model A (CatBoost on the anchor residual)     5 runway neighbours (neighbours' stage-4 residuals), model A refit
  6 model B (LightGBM) and an A/B blend   6b airport experts   7 no-NM specialist   7b Rome stage (lirf_regime.py)
  8 ADS-B gates (NM, no-NM)

Development pass (shipped design, --months 7): July 2025 is the outer test set. Cross-validation over the other eleven
months in four contiguous blocks (1-3, 4-6, 8-10, 11-12; Settings.inner_blocks = 4) produces every stage's out-of-fold
predictions; the blend weight, the expert shares and the airports each specialist/gate is kept at are chosen there and
frozen; the stack is fitted on the eleven months and predicts July once. That score is the honest one
(results/<run>/report_dev.json; reference: data/reference_v7f4/report_dev.json, 259.91 s). With --months 1,7 (the
Settings default) each listed month is an outer test in turn, predicted by a stack fitted on the other eleven.
Submission pass: the eleven months' out-of-fold predictions plus the July predictions are an out-of-fold set for all
twelve months; the stack is refitted on the twelve months with the frozen choices and predicts the ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import adsb, data, eurocontrol, features, weather
from .config import KEY, LABEL, REPO, Settings, paths
from .model import CATS, FAMILIES, mask_families, set_dropped
from .pipeline import STAGES, cv_stack, fit_stack, predict_stack, rmse, save_bundle

CODE_FILES = ["data.py", "fields.py", "adsb.py", "features.py", "queue_arrivals.py", "context_extra.py", "hotspots.py", "weather.py", "eurocontrol.py", "build_extra.py"]


MAX_LABEL_OVER_SCHED = 12 * 3600


def cell_matrix(f: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Error by airport x NM x ADS-B evidence (anchor group) on the plausible-label view: rows, RMSE, bias, share of squared error."""
    from .anchor import evidence
    e = pred - f.label.to_numpy(); ok = np.isfinite(e) & (f.label.to_numpy() - np.clip(f.sched.to_numpy(), 0, None) <= MAX_LABEL_OVER_SCHED)
    d = pd.DataFrame({"airport": f.airport.astype(str).to_numpy(), "group": evidence(f)[0], "month": f.month.to_numpy(), "e": e})[ok]
    sse = float((d.e ** 2).sum())
    g = d.groupby(["airport", "group"])
    t = pd.DataFrame({"rows": g.size(), "rmse": g.e.apply(lambda x: float(np.sqrt((x ** 2).mean()))), "bias": g.e.mean(),
                      "sse_share": g.e.apply(lambda x: float((x ** 2).sum()) / sse)})
    for mo in sorted(d.month.unique()):
        t[f"rmse_m{int(mo):02d}"] = d[d.month == mo].groupby(["airport", "group"]).e.apply(lambda x: float(np.sqrt((x ** 2).mean())))
    return t.sort_values("sse_share", ascending=False).reset_index()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scores(f: pd.DataFrame, pred: np.ndarray) -> dict:
    ok = np.isfinite(f.label.to_numpy()) & np.isfinite(pred)
    e = pred[ok] - f.label.to_numpy()[ok]; g = f[ok]
    nm = g.has_nm.to_numpy() > 0; lirf = (g.airport == "LIRF").to_numpy(); sub24 = g.label.to_numpy() <= 86400
    #: the plausible-label view (30 Sep): rows whose taxi exceeds the schedule gap by > 12 h are dropped - a label-only rule, so every
    #: model drops the same rows. On the 2025 holdout it drops exactly two LFPG no-NM rows (23.4 h and 16.2 h against
    #: 29-34 min schedule gaps) that alone hold ~35% of v4's holdout squared error; decisions are read on this view
    plausible = g.label.to_numpy() - np.clip(g.sched.to_numpy(), 0, None) <= MAX_LABEL_OVER_SCHED
    ord2 = g.label.to_numpy() <= 7200; se = np.sort(e ** 2)[::-1]
    #: the trimmed view (peer tip, 28 Sep; the old repos' concentration gate): a handful of rows can hold 30% of the
    #: squared error, so decisions are read on ordinary labels (<= 2 h) and the top-10 share is reported next to RMSE
    return {"all": rmse(e), "plausible": rmse(e[plausible]), "plausible_dropped_rows": int((~plausible).sum()),
            "plausible_by_month": {int(mo): rmse(e[plausible & (g.month == mo).to_numpy()]) for mo in sorted(g.month.unique())}, "labels_le_2h": rmse(e[ord2]), "nm_labels_le_2h": rmse(e[nm & ord2]),
            "top10_sse_share": float(se[:10].sum() / se.sum()) if se.sum() > 0 else 0.0,
            "all_labels_le_24h": rmse(e[sub24]), "nm_outside_lirf": rmse(e[nm & ~lirf]), "nm_lirf": rmse(e[nm & lirf]),
            "no_nm": rmse(e[~nm]), "no_nm_labels_le_24h": rmse(e[~nm & sub24]), "no_nm_lirf": rmse(e[~nm & lirf]),
            "by_month": {int(mo): rmse(e[(g.month == mo).to_numpy()]) for mo in sorted(g.month.unique())},
            "by_airport": {a: rmse(e[(g.airport == a).to_numpy()]) for a in sorted(g.airport.unique())}, "rows": int(ok.sum())}


def auc(p: np.ndarray, t: np.ndarray) -> float:
    if t.min() == t.max():
        return float("nan")
    r = pd.Series(p).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((REPO / "src" / "taxiout" / f).read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src/taxiout"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        return c + ("+local-edits" if dirty else "")
    except Exception:
        return "unknown"


def extra_inputs(dep: pd.DataFrame) -> pd.DataFrame | None:
    """Taxi dynamics (td_) and OPDI surface events (op_) built by `python -m taxiout.build_extra`; absent -> None."""
    from .build_extra import extra_dir
    d = extra_dir(); taxi = sorted(d.glob("taxi_*.parquet")); op = d / "opdi_events.parquet"
    if not taxi and not op.exists():
        return None
    out = dep[[KEY]].copy()
    if taxi:
        out = out.merge(pd.concat([pd.read_parquet(f) for f in taxi]).drop_duplicates(KEY), on=KEY, how="left")
    if op.exists():
        out = out.merge(pd.read_parquet(op), on=KEY, how="left")
    ib = d / "inbound.parquet"                                        # v4: the aircraft's previous arrival (build_extra inbound)
    if ib.exists():
        out = out.merge(pd.read_parquet(ib).drop_duplicates(KEY), on=KEY, how="left")
    for fn in ("hotspots.parquet", "hotspots2.parquet"):              # v4: waiting points and observed queue (hotspots.py)
        hs = d / fn
        if hs.exists():
            out = out.merge(pd.read_parquet(hs).drop_duplicates(KEY), on=KEY, how="left")
    for c in ("td_points", "op_matched", "op_cp_taxiway"):
        if c not in out:
            out[c] = np.nan
    out["op_matched"] = out["op_matched"].fillna(0.0)          # 1 matched per flight, 0 not (was 1 / missing)
    out.index = dep.index
    return out.drop(columns=[KEY])


def load_features() -> tuple[pd.DataFrame, dict]:
    p = paths(); cache = p["features"]; cache.mkdir(parents=True, exist_ok=True)
    fixed = os.environ.get("PRC_FEATURES")                       # v7: a fixed feature file (e.g. the v4 cache), no rebuild
    if fixed:
        log(f"features from the fixed file {fixed} (PRC_FEATURES)"); return pd.read_parquet(fixed), {"features_file": Path(fixed).name}
    m, hashes = data.load(p["organiser"]); clocks, clock_hash = adsb.load(p["adsb_clocks"])
    hashes["adsb_clocks"] = clock_hash; hashes["feature_code"] = code_hash()
    from .build_extra import extra_dir
    from .fields import fill_fields
    m, crep = fill_fields(m, extra_dir()); log(f"field filling (positional fields only): {crep}")
    hashes["extra"] = hashlib.sha256(str(sorted((x.name, x.stat().st_size) for x in extra_dir().glob("*.parquet"))).encode()).hexdigest()[:16]
    for k in ("metar", "open_meteo", "eurocontrol", "atfm"):
        if p.get(k) is not None:
            hashes[k] = hashlib.sha256(str(sorted(x.stat().st_size for x in ([p[k]] if p[k].is_file() else p[k].glob("*.parquet")))).encode()).hexdigest()[:16]
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    fp = cache / f"features_{key}.parquet"
    if fp.exists():
        log(f"features from cache {fp.name} (inputs and feature code unchanged)"); return pd.read_parquet(fp), hashes
    log(f"movements {len(m):,}; ADS-B clocks {len(clocks):,}; building features")
    dep = m[m.PHASE_mvt == "DEP"]; wx = []
    if p.get("metar") is not None:
        w, hashes["metar"] = weather.metar(dep, p["metar"]); wx.append(w)
        log(f"METAR: last report within 90 min found for {w.mt_tmpc.notna().mean():.1%} of departures")
    if p.get("open_meteo") is not None:
        w, hashes["open_meteo"] = weather.open_meteo(dep, p["open_meteo"]); wx.append(w)
        log(f"Open-Meteo: take-off hour found for {w.om_temperature_2m.notna().mean():.1%} of departures")
    if p.get("eurocontrol") is not None and p.get("atfm") is not None:
        day, atfm, hashes["eurocontrol"] = eurocontrol.daily(p["eurocontrol"], p["atfm"])
        w = eurocontrol.features(dep, day, atfm); wx.append(w)
        wm, hashes["eurocontrol_monthly"] = eurocontrol.monthly(p["eurocontrol"], dep); wx.append(wm)
        log(f"EUROCONTROL monthly statistics: taxi-out month found for {wm.eum_to_mean_s.notna().mean():.1%} of departures")
        log(f"EUROCONTROL daily tables: airport-day found for {1 - w.ast_missing.mean():.1%} of departures")
    ex = extra_inputs(dep)
    if ex is not None:
        wx.append(ex)
        log(f"taxi dynamics for {ex.td_points.notna().mean():.1%} of departures; OPDI flight-level match for {ex.op_matched.notna().mean():.1%} "
            f"(taxiway entry {ex.op_cp_taxiway.notna().mean():.1%})")
    f = features.build(m, clocks, wx); f.to_parquet(fp, index=False)
    log(f"features: {f.shape[0]:,} departures x {f.shape[1]} columns -> {fp.name}")
    return f, hashes


def combine_choices(chs: list[dict], s: Settings) -> dict:
    """The submission pass's choices from the outer runs' choices: gates at airports where their out-of-fold gains sum
    positive, a specialist where every run chose one, experts kept where every run gave a share > 0, mean blend weights."""
    from .pipeline import gate_airports
    gates = {}
    for which in chs[0]["gates"]:
        tot, rows = {}, {}
        for c in chs:
            for ap_, v in c["gates"][which]["oof_sse_removed_by_airport"].items():
                tot[ap_] = tot.get(ap_, 0.0) + v
            for ap_, v in c["gates"][which].get("evidence_rows_by_airport", {}).items():
                rows[ap_] = min(rows.get(ap_, 10 ** 9), v)
        gates[which] = {"airports": gate_airports(tot, rows, s), "oof_sse_removed_by_airport": tot, "evidence_rows_by_airport": rows}
    spec = {}
    for ap_ in set.intersection(*[set(c["specialist_airports"]) for c in chs]):     # every outer run chose one there
        gs = [c["specialist_oof_sse_removed"][ap_] for c in chs]
        spec[ap_] = max(gs, key=lambda g: g["sse_removed"])["variant"]
    # v4: the airport experts' shares (v3's submission dropped them: its choices carried no expert_weights, so the
    # 12-month refit fitted no expert although the holdout score included them). Mean share over the outer runs;
    # an airport keeps an expert only if every run gave it a share > 0
    ew = [c.get("expert_weights", {}) for c in chs]
    experts = {ap_: float(np.mean([e[ap_] for e in ew])) for ap_ in set.intersection(*[set(e) for e in ew]) if all(e[ap_] > 0 for e in ew)} if ew else {}
    w_ap = {k: float(np.mean([c.get("blend_weight_by_airport", {}).get(k, c["blend_weight_model_A"]) for c in chs]))
            for k in set().union(*[c.get("blend_weight_by_airport", {}) for c in chs])}
    ch = {"blend_weight_model_A": float(np.mean([c["blend_weight_model_A"] for c in chs])), "blend_weight_by_airport": w_ap, "specialist_airports": spec,
          "specialist_oof_sse_removed": {}, "gates": gates, "expert_weights": experts}
    return ch


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev-only", action="store_true"); ap.add_argument("--seed", type=int); ap.add_argument("--trees", type=int)
    ap.add_argument("--verbose", type=int); ap.add_argument("--no-gate-split", action="store_true"); ap.add_argument("--no-neighbours", action="store_true")
    ap.add_argument("--calibrate-regime", action="store_true"); ap.add_argument("--gate-genuine-only", action="store_true")
    ap.add_argument("--drop", default="", help=f"feature families to leave out, comma-separated: {', '.join(FAMILIES)}")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any Settings field (repeatable)")
    ap.add_argument("--months", default="", help="held-out months for the development pass, e.g. 7 (one-month benchmark)")
    ap.add_argument("--tag", default="", help="suffix for the run folder")
    ap.add_argument("--family-airports", default="", help="family=AP1,AP2;family2=AP3 - use a family only at these airports")
    a = ap.parse_args(); s = Settings()
    if a.cpu: s.device = "CPU"
    if a.seed: s.seed = a.seed
    if a.quick: s.sample_days, s.trees, s.gate_trees, s.lgb_trees, s.spec_trees, s.expert_trees = 0.15, 400, 300, 300, 300, 300
    if a.trees: s.trees = a.trees
    if a.verbose is not None: s.verbose = a.verbose
    if a.no_gate_split: s.gate_split = False
    if a.no_neighbours: s.neighbours = False
    if a.calibrate_regime: s.calibrate_regime = True
    if a.gate_genuine_only: s.gate_genuine_only = True
    if a.drop: s.drop = tuple(x.strip() for x in a.drop.split(",") if x.strip())
    if a.months: s.holdout_months = tuple(int(m) for m in a.months.split(","))
    if a.family_airports:
        s.family_airports = {k: tuple(v.split(",")) for k, v in (x.split("=", 1) for x in a.family_airports.split(";") if x)}
        unknown = set(s.family_airports) - set(FAMILIES)
        if unknown:
            raise SystemExit(f"--family-airports: unknown families {sorted(unknown)}")
    for kv in a.set:
        k, v = kv.split("=", 1)
        if not hasattr(s, k):
            raise SystemExit(f"--set: unknown setting {k}")
        cur = getattr(s, k)
        setattr(s, k, (v.lower() in ("1", "true", "yes", "on")) if isinstance(cur, bool) else type(cur)(v) if cur is not None else v)
    set_dropped(s.drop)
    from . import neighbours as _nb
    _nb.SKIP_AIRPORTS = tuple(x.strip() for x in s.nb_skip_airports.split(",") if x.strip())
    t0 = time.time(); run_id = (time.strftime("%Y%m%dT%H%M%S") + ("_quick" if a.quick else "") + ("_cpu" if a.cpu else "")
                                + ("_drop-" + "-".join(s.drop) if s.drop else "") + ("_genuine" if s.gate_genuine_only else "")
                                + (f"_{a.tag}" if a.tag else ""))
    out = paths()["out"] / run_id; out.mkdir(parents=True, exist_ok=True)
    f, hashes = load_features()
    if s.disruption:
        from . import disruption
        f = disruption.add(f); log("disruption features added (dis_*)")
    f = mask_families(f, s.family_airports)
    tr = f[f["set"] == "train"].reset_index(drop=True); rk = f[f["set"] == "rank"].reset_index(drop=True)
    if s.sample_days < 1:
        day = (tr.t // 86400).astype(int); keep = day.isin(day.drop_duplicates().sample(frac=s.sample_days, random_state=s.seed))
        tr = tr[keep.to_numpy()].reset_index(drop=True)
    both = pd.concat([tr, rk], ignore_index=True)
    cat_levels = {c: sorted(both[c].astype(str).unique()) for c in CATS if c in both}
    cat_levels["anchor_group"] = ["mid", "near", "none", "nonm_adsb", "nonm_none", "rev", "stand"]
    meta = {"run": run_id, "git_commit": git_commit(), "settings": asdict(s), "inputs": hashes,
            "libraries": {"python": platform.python_version(),
                          **{k: __import__(k).__version__ for k in ("numpy", "pandas", "pyarrow", "catboost", "lightgbm")}},
            "feature_columns": list(f.columns)}

    # ---- development pass: January <-> July cross. For each held-out month, the stack is cross-validated on the
    # other eleven 2025 months (month blocks), every choice is made there, and the held-out month is predicted once.
    hold_m = tr.month.isin(s.holdout_months).to_numpy(); hold = tr[hold_m].reset_index(drop=True)
    hp = {k: np.full(len(hold), np.nan) for k in STAGES + ["p_stamp"]}; grp_hold = np.empty(len(hold), dtype=object)
    outer = {}
    for tm in s.holdout_months:
        dev = tr[tr.month != tm].reset_index(drop=True); te = (hold.month == tm).to_numpy(); test = hold[te].reset_index(drop=True)
        folds = [tuple(int(m) for m in blk) for blk in np.array_split(sorted(dev.month.unique()), s.inner_blocks)]
        log(f"DEVELOPMENT (test month {tm}): cross-validate on {len(dev):,} departures in month blocks {folds}; test on {len(test):,}")
        oof_dev, ch = cv_stack(dev, folds, s, cat_levels, log)
        b_dev = fit_stack(dev, oof_dev, ch, s, cat_levels, log)
        p_ = predict_stack(b_dev, test)
        for k in hp:
            hp[k][te] = p_[k]
        grp_hold[te] = b_dev["anchor"].predict(test, p_["p_stamp"])[1]
        outer[tm] = {"choices": ch, "test": {k: scores(test, p_[k]) for k in STAGES}, "dev_oof": {k: scores(dev, oof_dev[k]) for k in STAGES},
                     "oof": oof_dev, "dev_keys": dev[KEY].to_numpy()}
        save_bundle(b_dev, out / f"bundle_test{tm:02d}", {**meta, "fitted_on": f"2025 months except {tm}"})
        v_ = outer[tm]["test"]["final"]
        log(f"  test month {tm}: final {v_['all']:.2f} | labels <= 2 h {v_['labels_le_2h']:.2f} | NM outside LIRF {v_['nm_outside_lirf']:.2f} | top-10 rows {v_['top10_sse_share']:.1%} of SSE")
    rep = {**meta, "evaluation": "January <-> July cross: each held-out month predicted by a stack fitted and tuned on the other eleven months",
           "holdout": {k: scores(hold, hp[k]) for k in STAGES},
           "by_test_month": {int(tm): {"test": o["test"], "dev_oof": o["dev_oof"], "choices": o["choices"]} for tm, o in outer.items()}}
    lf = ((hold.airport == "LIRF") & (hold.has_nm == 0)).to_numpy(); st = (np.abs(hold.label - hold.sched) <= 180).to_numpy().astype(int)
    cand = hp["p_stamp"] > 0
    rep["regime_check"] = {"auc_all_candidates": auc(hp["p_stamp"][cand], st[cand]), "auc_lirf_no_nm": auc(hp["p_stamp"][lf], st[lf]),
                           "lirf_no_nm_mean_p": float(hp["p_stamp"][lf].mean()) if lf.any() else None,
                           "lirf_no_nm_stamped_rate": float(st[lf].mean()) if lf.any() else None}
    log("HOLDOUT Jan + Jul 2025 (each month predicted by a stack that never saw it): all | labels <= 2 h | NM <= 2 h | NM outside LIRF | NM LIRF | no-NM | no-NM LIRF | top-10 share")
    for k in STAGES:
        v = rep["holdout"][k]
        log(f"  {k:20s} {v['all']:8.2f} | {v['labels_le_2h']:7.2f} | {v['nm_labels_le_2h']:7.2f} | {v['nm_outside_lirf']:7.2f} | {v['nm_lirf']:7.2f} | "
            f"{v['no_nm']:8.1f} | {v['no_nm_lirf']:8.1f} | {v['top10_sse_share']:5.1%}")
    rc = rep["regime_check"]
    log(f"regime at LIRF no-NM: AUC {rc['auc_lirf_no_nm']:.3f}, mean P {rc['lirf_no_nm_mean_p']}, stamped rate {rc['lirf_no_nm_stamped_rate']}")
    ref = paths().get("reference")
    if ref is not None and ref.exists():                     # evaluation only: an earlier model's holdout predictions, never trained on
        r = pd.read_parquet(ref).set_index(KEY).iloc[:, 0].reindex(hold[KEY]).to_numpy(); y = hold.label.to_numpy()
        bb = np.isfinite(r); nmb = hold.has_nm.to_numpy()[bb] > 0; lfb = hold.airport.to_numpy()[bb] == "LIRF"
        e_new, e_ref = hp["final"][bb] - y[bb], r[bb] - y[bb]
        rep["vs_reference"] = {"file": ref.name, "rows": int(bb.sum()), "this_run": rmse(e_new), "reference": rmse(e_ref),
                               "this_run_nm": rmse(e_new[nmb]), "reference_nm": rmse(e_ref[nmb]), "this_run_no_nm": rmse(e_new[~nmb]),
                               "reference_no_nm": rmse(e_ref[~nmb]), "this_run_no_nm_lirf": rmse(e_new[~nmb & lfb]), "reference_no_nm_lirf": rmse(e_ref[~nmb & lfb])}
        v = rep["vs_reference"]
        log(f"vs {ref.name} on {v['rows']:,} identical holdout rows: this run {v['this_run']:.2f} vs {v['reference']:.2f} | NM {v['this_run_nm']:.2f} vs "
            f"{v['reference_nm']:.2f} | no-NM {v['this_run_no_nm']:.1f} vs {v['reference_no_nm']:.1f} (LIRF {v['this_run_no_nm_lirf']:.1f} vs {v['reference_no_nm_lirf']:.1f})")
    (out / "report_dev.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    pd.DataFrame({KEY: hold[KEY], "month": hold.month, "label": hold.label, **{k: hp[k] for k in STAGES + ["p_stamp"]}}).to_parquet(out / "holdout_predictions.parquet", index=False)
    cell_matrix(hold, hp["final"]).to_csv(out / "holdout_cells.csv", index=False)
    v = rep["holdout"]["final"]; log(f"plausible-label view (label <= schedule gap + 12 h, {v['plausible_dropped_rows']} rows dropped): {v['plausible']:.2f} by month {v['plausible_by_month']}")
    t1 = s.holdout_months[0]
    pd.DataFrame({KEY: outer[t1]["dev_keys"], **{k: outer[t1]["oof"][k] for k in STAGES}}).to_parquet(out / "dev_oof.parquet", index=False)
    for tm, o in outer.items():                               # every outer run's out-of-fold set (refitting stages 7-8 alone needs it)
        pd.DataFrame({KEY: o["dev_keys"], "group": o["oof"]["group"], **{k: o["oof"][k] for k in STAGES}}).to_parquet(out / f"dev_oof_m{tm:02d}.parquet", index=False)
    log(f"development pass done in {time.time() - t0:.0f} s: report_dev.json, holdout_predictions.parquet, dev_oof.parquet, bundle_test*/")
    if a.dev_only:
        return

    # ---- submission pass. Choices: combined over the two outer runs (gates at airports where their out-of-fold gains
    # sum positive, a specialist where both runs chose one, the mean blend weight). Out-of-fold set for all twelve
    # months: the first outer run's eleven-month out-of-fold predictions plus its test month's predictions.
    chs = [o["choices"] for o in outer.values()]
    ch = combine_choices(chs, s); spec, experts, gates = ch["specialist_airports"], ch["expert_weights"], ch["gates"]
    o1 = outer[t1]; oof12 = {k: np.full(len(tr), np.nan) for k in STAGES}; grp = np.empty(len(tr), dtype=object)
    pos = pd.Series(np.arange(len(tr)), index=tr[KEY].to_numpy())
    idx_dev = pos.reindex(o1["dev_keys"]).to_numpy(); idx_t = pos.reindex(hold[KEY][(hold.month == t1).to_numpy()].to_numpy()).to_numpy()
    te1 = (hold.month == t1).to_numpy()
    for k in STAGES:
        oof12[k][idx_dev] = o1["oof"][k]; oof12[k][idx_t] = hp[k][te1]
    grp[idx_dev] = o1["oof"]["group"]; grp[idx_t] = grp_hold[te1]; oof12["group"] = grp
    log(f"SUBMISSION: refit on all {len(tr):,} departures; choices {json.dumps({'w': round(ch['blend_weight_model_A'], 3), 'spec': spec, 'experts': experts, 'gates': {k: v['airports'] for k, v in gates.items()}})}")
    if s.experts and not experts:
        log("  WARNING: experts are on but no airport kept an expert share")
    b_all = fit_stack(tr, oof12, ch, s, cat_levels, log)
    rp = predict_stack(b_all, rk)
    tmpl = pd.read_parquet(paths()["organiser"] / "submitting.parquet")
    sub = tmpl[[KEY]].merge(pd.DataFrame({KEY: rk[KEY], LABEL: rp["final"]}), on=KEY, how="left")
    assert sub[LABEL].notna().all() and len(sub) == len(tmpl)
    sub[LABEL] = np.round(sub[LABEL]).clip(lower=0).astype(tmpl[LABEL].dtype)
    sub.to_parquet(out / "submission.parquet", index=False)
    save_bundle(b_all, out / "bundle", {**meta, "fitted_on": "all twelve 2025 months"})
    rep_s = {**meta, "choices": ch, "submission_sha256": hashlib.sha256((out / "submission.parquet").read_bytes()).hexdigest(),
             "elapsed_s": round(time.time() - t0), "ranking_stage_means": {k: float(np.mean(rp[k])) for k in STAGES}}
    (out / "report_submission.json").write_text(json.dumps(rep_s, indent=1, default=float), encoding="utf-8")
    log(f"wrote submission.parquet ({len(sub):,} rows), bundle/ and report_submission.json in {rep_s['elapsed_s']} s")

if __name__ == "__main__":
    main()
