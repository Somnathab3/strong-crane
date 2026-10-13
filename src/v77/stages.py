# SPDX-License-Identifier: GPL-3.0-only
"""v77: the V7f4 pipeline as separate steps. Thin wrappers around taxiout - no modelling logic lives here.

Every function mirrors one block of taxiout.run.main (development pass, report, submission pass), so the notebook can
run the chain stage by stage and look at each stage's numbers before the next one. The settings, the inputs and the
reference results come from v77/v77.yaml.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from taxiout import neighbours
from taxiout.config import KEY, LABEL, REPO, Settings, paths
from taxiout.model import CATS, DROP as MODEL_DROP, FAMILIES, kept, mask_families, set_dropped
from taxiout.pipeline import STAGES, cv_base, cv_top, fit_stack, predict_stack, rmse, save_bundle
from taxiout.run import auc, cell_matrix, combine_choices, git_commit, load_features, scores

V77 = Path(__file__).resolve().parent
CFG_PATH = V77 / "v77.yaml"
H = 3600.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------ configuration
def load_config(path: Path = CFG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_settings(cfg: dict, smoke: bool = False) -> Settings:
    """The Settings object the V7f4 run used (cfg['settings']), with the smoke overrides on top when asked.
    Also applies the two module-level switches run.main sets (dropped families, neighbour-skip airports) and pins
    the feature file through PRC_FEATURES, exactly as the run script did."""
    s = Settings()
    overrides = dict(cfg["settings"])
    if os.environ.get("V77_HEDGE", "0") == "1":               # the v77 Rome hedge options (hedge_options in the YAML)
        overrides.update(cfg.get("hedge_options", {}))
    if smoke:
        overrides.update(cfg["smoke"]["settings"])
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise KeyError(f"unknown Settings field {k}")
        cur = getattr(s, k)
        if isinstance(cur, tuple):
            v = tuple(v)
        elif isinstance(cur, dict):
            v = {a: tuple(b) for a, b in v.items()}
        setattr(s, k, v)
    dev = os.environ.get("V77_DEVICE", "").upper()              # CPU / GPU, overrides the YAML (e.g. a machine without a GPU)
    if dev:
        s.device = dev
        if hasattr(s, "regime_device"):
            s.regime_device = dev
    unknown = set(s.family_airports) - set(FAMILIES)
    if unknown:
        raise KeyError(f"family_airports: unknown families {sorted(unknown)}")
    set_dropped(s.drop)
    neighbours.SKIP_AIRPORTS = tuple(x.strip() for x in s.nb_skip_airports.split(",") if x.strip())
    if features_mode(cfg) == "pinned":                       # the run script's PRC_FEATURES; unset -> run.load_features builds
        os.environ["PRC_FEATURES"] = str(REPO / cfg["inputs"]["features_file"])
    else:
        os.environ.pop("PRC_FEATURES", None)
    return s


def features_mode(cfg: dict) -> str:
    """'pinned' (the fixed table of the reference run) or 'build' (from the raw inputs); V77_FEATURES overrides the YAML."""
    m = os.environ.get("V77_FEATURES", cfg["inputs"].get("features_mode", "pinned")).lower()
    if m not in ("pinned", "build"):
        raise ValueError(f"features_mode must be pinned or build, not {m}")
    return m


# ------------------------------------------------------------------------------------------------ stage 0: raw inputs
def raw_inventory(cfg: dict, hash_organiser: bool = True) -> dict:
    """What stage 1 starts from: the organiser files (name, rows are not read here; size and SHA-256) and every external
    input location of configs/paths.json with its existence and entry count."""
    from taxiout.data import file_hash
    p = paths(); org = p["organiser"]
    files = sorted(org.glob("training_2025-*.parquet")) + [org / "ranking.parquet", org / "submitting.parquet"]
    organiser = pd.DataFrame([{"file": f.name, "bytes": f.stat().st_size if f.exists() else None,
                               "sha256": (file_hash(f)[:16] if hash_organiser and f.exists() else "")} for f in files])
    rows = []
    for k, v in p.items():
        if v is None or k in ("out", "features", "extra"):
            continue
        rows.append({"input": k, "path": str(v), "exists": v.exists(),
                     "entries": (sum(1 for _ in v.glob("*")) if v.is_dir() else 1) if v.exists() else 0})
    from taxiout.build_extra import extra_dir
    ed = extra_dir()
    rows.append({"input": "extra (build_extra outputs)", "path": str(ed), "exists": ed.exists(), "entries": sum(1 for _ in ed.glob("*.parquet")) if ed.exists() else 0})
    return {"organiser_files": organiser, "external_inputs": pd.DataFrame(rows), "features_mode": features_mode(cfg)}


# every input: in which feature mode it is needed ("both", "pinned" or "build"), how to get it, and a fetch command
# when this package can download or build it ({path} = the configured location, {parent} = its folder)
INPUT_GUIDE = {
    "organiser": ("both", "the challenge files (training_2025-*.parquet, ranking.parquet, submitting.parquet) from "
                  "https://ansperformance.eu/study/data-challenge/dc2026/data.html", None),
    "features_file": ("pinned", "the pinned feature table of the reference run (features_f391f7893a3cde99.parquet, 615 MB); "
                      "without it set V77_FEATURES=build to build the table from the raw inputs below", None),
    "adsb_clocks": ("build", "clock_<day>.parquet per day, built from the adsb.lol daily trace archive (SOURCES.md); "
                    "not rebuilt by this package", None),
    "metar": ("build", "hourly METAR from the IEM ASOS archive", ["scripts/download_metar.py", "--destination", "{path}"]),
    "open_meteo": ("build", "Open-Meteo historical forecast per airport-hour", ["scripts/download_open_meteo.py", "--root", "{parent}"]),
    "eurocontrol": ("build", "EUROCONTROL airport statistics, https://ansperformance.eu/data/ (files in SOURCES.md)", None),
    "atfm": ("build", "EUROCONTROL airport arrival ATFM delay, https://ansperformance.eu/data/", None),
    "opdi": ("build", "OPDI v002 flight_list/ and flight_events/ (URL pattern in SOURCES.md)", None),
    "extra": ("build", "python -m taxiout.build_extra opdi | types (from OPDI and the clocks); taxi, stops, hotspots and "
              "inbound need the raw adsb.lol traces", None),
}


def input_check(cfg: dict) -> pd.DataFrame:
    """Every input: needed in this feature mode, found, entries, and how to get it (INPUT_GUIDE)."""
    from taxiout.build_extra import extra_dir
    mode = features_mode(cfg); p = paths()
    loc = {k: p.get(k) for k in INPUT_GUIDE}; loc["features_file"] = REPO / cfg["inputs"]["features_file"]; loc["extra"] = extra_dir()
    rows = []
    for k, (need, how, cmd) in INPUT_GUIDE.items():
        v = loc.get(k); found = v is not None and Path(v).exists()
        n = (sum(1 for _ in Path(v).glob("*")) if Path(v).is_dir() else 1) if found else 0
        found = found and n > 0
        rows.append({"input": k, "needed": need in ("both", mode), "found": found, "entries": n, "path": str(v),
                     "fetch": "yes" if cmd else "", "how to get it": how})
    return pd.DataFrame(rows)


def fetch_missing(cfg: dict, chk: pd.DataFrame) -> pd.DataFrame:
    """Download / build every needed and missing input that has a fetch command (V77_FETCH=1), then check again."""
    import subprocess, sys
    for r in chk[chk.needed & ~chk.found & (chk.fetch == "yes")].itertuples():
        v = Path(r.path); cmd = [sys.executable] + [a.format(path=v, parent=v.parent) for a in INPUT_GUIDE[r.input][2]]
        print("fetching", r.input, ":", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, cwd=REPO, check=False)
    return input_check(cfg)


def non_default_settings(s: Settings) -> dict:
    d0, d1 = asdict(Settings()), asdict(s)
    return {k: d1[k] for k in d1 if json.dumps(d1[k], sort_keys=True, default=str) != json.dumps(d0[k], sort_keys=True, default=str)}


def settings_vs_reference(s: Settings, cfg: dict) -> pd.DataFrame:
    """Field by field against the reference run's recorded settings (its report_dev.json), where that run exists."""
    ref_dir = REPO / cfg["inputs"]["reference_run"]
    rp = ref_dir / "report_dev.json"
    if not rp.exists():
        return pd.DataFrame(columns=["field", "this", "reference"])
    ref = json.loads(rp.read_text(encoding="utf-8"))["settings"]
    rows = []
    for k, v in asdict(s).items():
        if k not in ref:                                       # a field added after the reference run (the v77 hedge options)
            continue
        a, b = json.dumps(v, sort_keys=True, default=str), json.dumps(ref.get(k), sort_keys=True, default=str)
        if a != b:
            rows.append({"field": k, "this": v, "reference": ref.get(k)})
    return pd.DataFrame(rows, columns=["field", "this", "reference"])


def run_dir(cfg: dict, smoke: bool) -> Path:
    d = REPO / "results" / cfg["name"] / (time.strftime("%Y%m%dT%H%M%S") + ("_smoke" if smoke else "_full"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------- stage 1: features
def verify_feature_file(cfg: dict) -> dict:
    p = REPO / cfg["inputs"]["features_file"]
    info = {"file": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None,
            "bytes_ok": p.exists() and p.stat().st_size == cfg["inputs"]["features_bytes"]}
    if p.exists() and cfg["inputs"].get("verify_sha256", True):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        info["sha256"] = h.hexdigest(); info["sha256_ok"] = info["sha256"] == cfg["inputs"]["features_sha256"]
    return info


def feature_table(cfg: dict, s: Settings) -> dict:
    """run.main's stage-1 block: the feature table (pinned file, or built from the raw inputs and cached), the
    family-airport masking, the train / rank split, the smoke-run day sample, the categorical levels and the run metadata."""
    f, hashes = load_features()
    f = mask_families(f, s.family_airports)
    tr = f[f["set"] == "train"].reset_index(drop=True); rk = f[f["set"] == "rank"].reset_index(drop=True)
    if s.sample_days < 1:
        day = (tr.t // 86400).astype(int); keep = day.isin(day.drop_duplicates().sample(frac=s.sample_days, random_state=s.seed))
        tr = tr[keep.to_numpy()].reset_index(drop=True)
    both = pd.concat([tr, rk], ignore_index=True)
    cat_levels = {c: sorted(both[c].astype(str).unique()) for c in CATS if c in both}
    cat_levels["anchor_group"] = ["mid", "near", "none", "nonm_adsb", "nonm_none", "rev", "stand"]
    meta = {"run": "v77", "git_commit": git_commit(), "settings": asdict(s), "inputs": hashes,
            "libraries": {"python": platform.python_version(),
                          **{k: __import__(k).__version__ for k in ("numpy", "pandas", "pyarrow", "catboost", "lightgbm")}},
            "feature_columns": list(f.columns)}
    return {"f": f, "tr": tr, "rk": rk, "cat_levels": cat_levels, "meta": meta, "hashes": hashes}


META_COLS = {KEY, "set", "month", "label", "t"}
RAW_COLS = {"actype_raw", "market_raw", "flight_type_raw", "wake_raw"}


def feature_inventory(f: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Every cached column: its family (model.FAMILIES prefixes, else core), whether model A reads it under these
    settings, and its missing share on 2025 NM / no-NM rows."""
    tr = f[f["set"] == "train"]; nm = tr.has_nm.to_numpy() > 0
    keep = set(kept([c for c in f.columns if c not in MODEL_DROP]))
    rows = []
    for c in f.columns:
        hits = [(len(p), k) for k, pre in FAMILIES.items() for p in pre if c.startswith(p)]     # longest prefix wins: om_visibility is om_drift, not weather
        fam = max(hits)[1] if hits else "core"
        role = ("key / split / label / time" if c in META_COLS else "raw organiser field (stage 2 only)" if c in RAW_COLS
                else "dropped family" if fam in s.drop else "restricted to airports" if fam in s.family_airports else "model A input")
        col = tr[c]
        miss_nm = float(col[nm].isna().mean()) if pd.api.types.is_numeric_dtype(col) else float((col[nm].astype(str) == "nan").mean())
        miss_nn = float(col[~nm].isna().mean()) if pd.api.types.is_numeric_dtype(col) else float((col[~nm].astype(str) == "nan").mean())
        rows.append({"column": c, "family": fam, "role": role, "model_A_input": c in keep and c not in META_COLS and c not in RAW_COLS,
                     "dtype": str(col.dtype), "missing_nm": round(miss_nm, 4), "missing_no_nm": round(miss_nn, 4),
                     "restricted_airports": ",".join(s.family_airports.get(fam, ())) or ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- basic EDA
MAX_LABEL_OVER_SCHED = 12 * 3600


def eda_tables(tr: pd.DataFrame, rk: pd.DataFrame) -> dict:
    """The few tables that frame the problem: rows by airport and set, NM share, label quantiles, and Rome's
    no-NM departures by schedule gap (stamped = |label - gap| <= 180 s)."""
    out = {}
    both = pd.concat([tr.assign(_set="train 2025"), rk.assign(_set="rank 2026")], ignore_index=True)
    out["rows_by_airport_set"] = pd.crosstab(both.airport.astype(str), both._set)
    out["no_nm_share_by_airport"] = (pd.crosstab(both.airport.astype(str), both._set, values=(both.has_nm == 0).astype(float), aggfunc="mean")
                                     .round(4))
    y = tr.label.to_numpy(); g = np.clip(tr.sched.to_numpy(), 0, None); plausible = (y - g) <= MAX_LABEL_OVER_SCHED
    q = tr[plausible].assign(pop=np.where(tr.has_nm[plausible] > 0, "NM", "no-NM"))
    out["label_quantiles_s"] = (q.groupby(["pop", q.airport.astype(str)]).label.quantile([0.1, 0.5, 0.9, 0.99]).unstack().round(0))
    out["plausible_view_dropped_rows"] = int((~plausible).sum())
    r = tr[(tr.airport.astype(str) == "LIRF") & (tr.has_nm == 0) & (tr.sched >= 1800)].copy()
    st = (r.label - r.sched).abs() <= 180; ds = r.label > 20 * H
    r["class"] = np.where(st, "stamped", np.where(ds, "day-shift", "ordinary / other"))
    r["gap band"] = pd.cut(r.sched / H, [0.5, 1, 2, 4, 8, 14, 20, 100], right=False).astype(str)
    out["lirf_no_nm_by_gap_band"] = pd.crosstab(r["gap band"], r["class"])
    out["month_rows_train"] = tr.groupby("month").size()
    return out


def eda_plots(tr: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    y = tr.label.to_numpy(); nm = tr.has_nm.to_numpy() > 0; ok = np.isfinite(y) & (y <= 7200)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y[ok & nm] / 60, bins=120, alpha=0.7, label="NM record")
    ax.hist(y[ok & ~nm] / 60, bins=120, alpha=0.7, label="no NM record")
    ax.set_xlabel("taxi-out, minutes (labels <= 2 h)"); ax.set_ylabel("departures"); ax.set_yscale("log"); ax.legend(); ax.set_title("2025 taxi-out labels")
    p = out / "eda_labels.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); files.append(p)
    r = tr[(tr.airport.astype(str) == "LIRF") & (tr.has_nm == 0) & (tr.sched >= 1800)]
    if len(r):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(r.sched / H, r.label / H, s=6, alpha=0.5)
        lim = max(1, float(np.nanmax(r.sched / H))); ax.plot([0, lim], [0, lim], lw=0.8, color="grey")
        ax.set_xlabel("schedule gap, h (take-off - scheduled off-block)"); ax.set_ylabel("label, h"); ax.set_title("LIRF departures without an NM record: label vs gap")
        p = out / "eda_lirf_label_vs_gap.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); files.append(p)
    return files


# ------------------------------------------------------------------------------------- development pass (stages 2-8)
def stage_table(frame: pd.DataFrame, pred: dict, stages: list[str] = STAGES) -> pd.DataFrame:
    """RMSE per stage on `frame` (all rows, labels <= 2 h, NM outside LIRF, NM LIRF, no-NM, no-NM LIRF, top-10 share)."""
    rows = []
    for k in stages:
        if k not in pred or not np.isfinite(pred[k]).any():
            continue
        v = scores(frame, pred[k])
        rows.append({"stage": k, "all": v["all"], "labels_le_2h": v["labels_le_2h"], "nm_outside_lirf": v["nm_outside_lirf"],
                     "nm_lirf": v["nm_lirf"], "no_nm": v["no_nm"], "no_nm_lirf": v["no_nm_lirf"], "top10_sse_share": v["top10_sse_share"], "rows": v["rows"]})
    return pd.DataFrame(rows).set_index("stage").round(2)


def dev_split(tr: pd.DataFrame, s: Settings, tm: int) -> dict:
    """For the held-out month tm: the development rows (the other months), the test rows and the inner month blocks."""
    dev = tr[tr.month != tm].reset_index(drop=True); test = tr[tr.month == tm].reset_index(drop=True)
    folds = [tuple(int(m) for m in blk) for blk in np.array_split(sorted(dev.month.unique()), s.inner_blocks)]
    return {"dev": dev, "test": test, "folds": folds}


def cv_stages_2_to_6(dev: pd.DataFrame, folds: list, s: Settings, cat_levels: dict) -> dict:
    """Out-of-fold over the inner month blocks: regime, anchor, model A, runway neighbours, LightGBM, blend, experts."""
    oof, nb, base_ch = cv_base(dev, folds, s, cat_levels, log)
    return {"oof": oof, "nb": nb, "base_choices": base_ch}


def cv_stages_7_to_8(dev: pd.DataFrame, folds: list, cv: dict, s: Settings) -> dict:
    """Out-of-fold on top of the base stages: no-NM specialists, ADS-B gates; the choices made on these rows only."""
    oof, ch = cv_top(dev, folds, cv["oof"], cv["nb"], cv["base_choices"], s, log)
    return {"oof": oof, "choices": ch}


def fit_and_predict_month(dev: pd.DataFrame, test: pd.DataFrame, oof_dev: dict, ch: dict, s: Settings, cat_levels: dict, meta: dict,
                          out: Path, tm: int) -> dict:
    """The stack fitted on the development months with the frozen choices, predicting the held-out month once."""
    b_dev = fit_stack(dev, oof_dev, ch, s, cat_levels, log)
    p_ = predict_stack(b_dev, test)
    grp = b_dev["anchor"].predict(test, p_["p_stamp"])[1]
    save_bundle(b_dev, out / f"bundle_test{tm:02d}", {**meta, "fitted_on": f"2025 months except {tm}"})
    return {"pred": p_, "group": grp, "bundle": b_dev,
            "outer": {"choices": ch, "test": {k: scores(test, p_[k]) for k in STAGES}, "dev_oof": {k: scores(dev, oof_dev[k]) for k in STAGES},
                      "oof": oof_dev, "dev_keys": dev[KEY].to_numpy()}}


def assemble_holdout(tr: pd.DataFrame, s: Settings, month_results: dict) -> dict:
    """hold / hp / grp_hold exactly as run.main keeps them (the held-out months in tr order)."""
    hold = tr[tr.month.isin(s.holdout_months)].reset_index(drop=True)
    hp = {k: np.full(len(hold), np.nan) for k in STAGES + ["p_stamp"]}; grp_hold = np.empty(len(hold), dtype=object)
    for tm, r in month_results.items():
        te = (hold.month == tm).to_numpy()
        for k in hp:
            hp[k][te] = r["pred"][k]
        grp_hold[te] = r["group"]
    return {"hold": hold, "hp": hp, "grp_hold": grp_hold, "outer": {tm: r["outer"] for tm, r in month_results.items()}}


def holdout_report(hd: dict, s: Settings, meta: dict, out: Path, cfg: dict) -> dict:
    """report_dev.json, holdout_predictions.parquet, holdout_cells.csv, dev_oof*.parquet - as run.main writes them."""
    hold, hp, outer = hd["hold"], hd["hp"], hd["outer"]
    rep = {**meta, "evaluation": "held-out month(s) predicted by a stack fitted and tuned on the other eleven months",
           "holdout": {k: scores(hold, hp[k]) for k in STAGES},
           "by_test_month": {int(tm): {"test": o["test"], "dev_oof": o["dev_oof"], "choices": o["choices"]} for tm, o in outer.items()}}
    lf = ((hold.airport == "LIRF") & (hold.has_nm == 0)).to_numpy(); st = (np.abs(hold.label - hold.sched) <= 180).to_numpy().astype(int)
    cand = hp["p_stamp"] > 0
    rep["regime_check"] = {"auc_all_candidates": auc(hp["p_stamp"][cand], st[cand]) if cand.any() else None,
                           "auc_lirf_no_nm": auc(hp["p_stamp"][lf], st[lf]) if lf.any() else None,
                           "lirf_no_nm_mean_p": float(hp["p_stamp"][lf].mean()) if lf.any() else None,
                           "lirf_no_nm_stamped_rate": float(st[lf].mean()) if lf.any() else None}
    ref = paths().get("reference")
    if ref is not None and ref.exists():                      # evaluation only: an earlier model's holdout predictions
        r = pd.read_parquet(ref).set_index(KEY).iloc[:, 0].reindex(hold[KEY]).to_numpy(); y = hold.label.to_numpy()
        bb = np.isfinite(r); nmb = hold.has_nm.to_numpy()[bb] > 0; lfb = hold.airport.to_numpy()[bb] == "LIRF"
        if bb.any():
            e_new, e_ref = hp["final"][bb] - y[bb], r[bb] - y[bb]
            rep["vs_reference"] = {"file": ref.name, "rows": int(bb.sum()), "this_run": rmse(e_new), "reference": rmse(e_ref),
                                   "this_run_nm": rmse(e_new[nmb]), "reference_nm": rmse(e_ref[nmb]), "this_run_no_nm": rmse(e_new[~nmb]),
                                   "reference_no_nm": rmse(e_ref[~nmb]), "this_run_no_nm_lirf": rmse(e_new[~nmb & lfb]), "reference_no_nm_lirf": rmse(e_ref[~nmb & lfb])}
    (out / "report_dev.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    pd.DataFrame({KEY: hold[KEY], "month": hold.month, "label": hold.label, **{k: hp[k] for k in STAGES + ["p_stamp"]}}).to_parquet(out / "holdout_predictions.parquet", index=False)
    cell_matrix(hold, hp["final"]).to_csv(out / "holdout_cells.csv", index=False)
    t1 = s.holdout_months[0]
    pd.DataFrame({KEY: outer[t1]["dev_keys"], **{k: outer[t1]["oof"][k] for k in STAGES}}).to_parquet(out / "dev_oof.parquet", index=False)
    for tm, o in outer.items():
        pd.DataFrame({KEY: o["dev_keys"], "group": o["oof"]["group"], **{k: o["oof"][k] for k in STAGES}}).to_parquet(out / f"dev_oof_m{tm:02d}.parquet", index=False)
    return rep


# ------------------------------------------------------------------------------------------ submission pass (12 months)
def submission_pass(tr: pd.DataFrame, rk: pd.DataFrame, hd: dict, s: Settings, cat_levels: dict, meta: dict, out: Path) -> dict:
    """run.main's submission block: choices combined over the outer runs, the twelve-month out-of-fold set assembled
    from the first outer run's out-of-fold rows plus its test month's predictions, the stack refitted on all of 2025
    with the frozen choices, the ranking rows predicted, the file written in the template's order."""
    hold, hp, grp_hold, outer = hd["hold"], hd["hp"], hd["grp_hold"], hd["outer"]
    chs = [o["choices"] for o in outer.values()]
    ch = combine_choices(chs, s)
    t1 = s.holdout_months[0]; o1 = outer[t1]
    oof12 = {k: np.full(len(tr), np.nan) for k in STAGES}; grp = np.empty(len(tr), dtype=object)
    pos = pd.Series(np.arange(len(tr)), index=tr[KEY].to_numpy())
    idx_dev = pos.reindex(o1["dev_keys"]).to_numpy(); te1 = (hold.month == t1).to_numpy()
    idx_t = pos.reindex(hold[KEY][te1].to_numpy()).to_numpy()
    for k in STAGES:
        oof12[k][idx_dev] = o1["oof"][k]; oof12[k][idx_t] = hp[k][te1]
    grp[idx_dev] = o1["oof"]["group"]; grp[idx_t] = grp_hold[te1]; oof12["group"] = grp
    log(f"SUBMISSION: refit on all {len(tr):,} departures; choices {json.dumps({'w': round(ch['blend_weight_model_A'], 3), 'spec': ch['specialist_airports'], 'experts': ch['expert_weights'], 'gates': {k: v['airports'] for k, v in ch['gates'].items()}})}")
    if s.experts and not ch["expert_weights"]:
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
             "ranking_stage_means": {k: float(np.mean(rp[k])) for k in STAGES}}
    (out / "report_submission.json").write_text(json.dumps(rep_s, indent=1, default=float), encoding="utf-8")
    return {"sub": sub, "rp": rp, "choices": ch, "report": rep_s, "template_dtype": tmpl[LABEL].dtype}


# --------------------------------------------------------------------------------------------- LOBT window (range fix)
def _raw_departures(cols: list[str]) -> pd.DataFrame:
    org = paths()["organiser"]
    files = sorted(org.glob("training_2025-*.parquet")) + [org / "ranking.parquet"]
    d = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt"] + cols).assign(_file=p.name) for p in files])
    return d[d.PHASE_mvt == "DEP"].drop_duplicates(KEY)


def _secs(s: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def lobt_property_2025(cfg: dict) -> dict:
    """The data property behind the window: |BLOCK - LOBT| on every 2025 departure with a LOBT. Read from the raw
    training files only here, as a check; it is never a feature."""
    hw = cfg["lobt_window"]["half_width_s"]
    d = _raw_departures([KEY, "BLOCK_TIME_UTC_mvt", "LOBT_flt"]); d = d[d._file != "ranking.parquet"]
    x = _secs(d.BLOCK_TIME_UTC_mvt) - _secs(d.LOBT_flt); ok = np.isfinite(x)
    return {"departures_2025_with_lobt": int(ok.sum()), "max_abs_block_minus_lobt_s": float(np.nanmax(np.abs(x[ok]))),
            "exceptions_over_half_width": int((np.abs(x[ok]) > hw).sum()), "half_width_s": hw}


def lobt_window(sub: pd.DataFrame, cfg: dict, template_dtype) -> tuple[pd.DataFrame, dict]:
    """Move predictions outside [MVT - LOBT - hw, MVT - LOBT + hw] to the nearest edge (ranking rows with a LOBT)."""
    hw = cfg["lobt_window"]["half_width_s"]
    d = _raw_departures([KEY, "MVT_TIME_UTC_mvt", "LOBT_flt"]); d = d[d._file == "ranking.parquet"]
    c = pd.Series(_secs(d.MVT_TIME_UTC_mvt) - _secs(d.LOBT_flt), index=d[KEY].to_numpy()).reindex(sub[KEY].to_numpy()).to_numpy()
    p = sub[LABEL].to_numpy(dtype=float); has = np.isfinite(c)
    w = np.where(has, np.clip(p, c - hw, c + hw), p)
    out = sub.copy(); out[LABEL] = np.round(np.clip(w, 0, None)).astype(template_dtype)
    moved = has & (out[LABEL].to_numpy(dtype=float) != p)
    return out, {"ranking_rows_with_lobt": int(has.sum()), "rows_moved": int(moved.sum()),
                 "mean_abs_move_s": float(np.abs(out[LABEL].to_numpy(dtype=float) - p)[moved].mean()) if moved.any() else 0.0}


# ------------------------------------------------------------------------------------------------------- comparisons
def compare_holdout(rep: dict, cfg: dict) -> pd.DataFrame:
    """This run's held-out stage RMSE next to the reference run's (its report_dev.json if present, else the YAML)."""
    ref_dir = REPO / cfg["inputs"]["reference_run"]; rp = ref_dir / "report_dev.json"
    if rp.exists():
        ref = {k: v["all"] for k, v in json.loads(rp.read_text(encoding="utf-8"))["holdout"].items()}
    else:
        ref = cfg["reference"]["holdout_july_2025"]
    t = pd.DataFrame({"this run": {k: rep["holdout"][k]["all"] for k in STAGES}, "reference run": {k: ref.get(k) for k in STAGES}})
    t["difference"] = t["this run"] - t["reference run"]; t["within_tolerance"] = t.difference.abs() <= cfg["reference"]["tolerance_rmse_s"]
    return t.round(2)


def compare_submission(sub: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """RMS and max difference of this file against the reference run's submission and our uploaded v111 / v117
    (only the files are read: what the excluded steps changed, no scores involved)."""
    ref_dir = REPO / cfg["inputs"]["reference_run"]; sd = REPO / cfg["inputs"]["submissions_dir"]
    cands = {"V7f4 run submission.parquet": ref_dir / "submission.parquet", "v111 (V7f4 + excluded steps)": sd / "strong-crane_v111.parquet",
             "v117 (v111 + LOBT window)": sd / "strong-crane_v117.parquet"}
    me = sub.set_index(KEY)[LABEL].astype(float); rows = []
    for name, p in cands.items():
        if not p.exists():
            rows.append({"file": name, "available": False}); continue
        o = pd.read_parquet(p); o = o.set_index(o[KEY].astype(me.index.dtype))[LABEL].astype(float).reindex(me.index)
        d = me.to_numpy() - o.to_numpy()
        rows.append({"file": name, "available": True, "rows_differing": int((np.abs(d) > 0.5).sum()), "rms_diff_s": float(np.sqrt(np.mean(d ** 2))),
                     "max_abs_diff_s": float(np.max(np.abs(d))), "mean_diff_s": float(np.mean(d))})
    return pd.DataFrame(rows).round(2)


def write_results(out: Path, results: dict) -> Path:
    p = out / "results.yaml"
    p.write_text(yaml.safe_dump(json.loads(json.dumps(results, default=_plain)), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def _plain(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="index"))
    if isinstance(o, Path):
        return str(o)
    return str(o)
