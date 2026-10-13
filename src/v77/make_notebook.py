# SPDX-License-Identifier: GPL-3.0-only
"""Writes notebooks/v77_pipeline.ipynb from the cell list below (nbformat), so the notebook is generated, not hand-edited JSON.
    python src/v77/make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

V77 = Path(__file__).resolve().parent; REPO = V77.parents[1]

CELLS = [
("md", """# Taxi-out time: from the inputs to the submission file

One run, top to bottom: check the inputs, build (or read) the feature table, train the stages with month-block
cross-validation, predict one held-out month of 2025 as a check, refit on all of 2025, predict the ranking set and write
the submission file. The code is in `src/taxiout/` (the pipeline) and `src/v77/stages.py` (one function per step);
settings and inputs are in `src/v77/v77.yaml` and `configs/paths.json`. Every output goes to `results/v77/<run>/`.

Switches, read from the environment:

| switch | values | meaning |
|---|---|---|
| `V77_SMOKE` | `1` (default) / `0` | 5% of the days with small models, a quick check that every step runs / **the full run** (~12.5 h on a GPU) |
| `V77_FEATURES` | `pinned` (default) / `build` | read the pinned feature table / **build it from the raw inputs** |
| `V77_DEVICE` | unset / `CPU` / `GPU` | the device in the YAML (GPU) / CatBoost on the CPU (no GPU needed, slower) |
| `V77_FETCH` | `0` (default) / `1` | stop if a needed input is missing / **download the missing METAR and Open-Meteo inputs first** |
| `V77_HEDGE` | `0` (default) / `1` | Rome stage as configured / the optional Rome hedge settings (`hedge_options` in the YAML) |"""),
("code", """import os, json, time
import numpy as np, pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path.cwd() / "src"))  # run from the repository root
from v77 import stages as S
from taxiout.pipeline import STAGES

SMOKE = os.environ.get("V77_SMOKE", "1") == "1"            # V77_SMOKE=0 trains everything
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
cfg = S.load_config(); FEATURES = S.features_mode(cfg)
print("mode:", "SMOKE" if SMOKE else "FULL", "| features:", FEATURES)"""),
("md", """## 1. Inputs

Every input from `configs/paths.json`: whether this mode needs it (`pinned`: the organiser files and the pinned feature
table; `build`: every raw input), whether it was found, and how to get it. The run stops here if something is missing."""),
("code", """chk = S.input_check(cfg)
if os.environ.get("V77_FETCH", "0") == "1" and (chk.needed & ~chk.found).any():
    chk = S.fetch_missing(cfg, chk)
display(chk[["input", "needed", "found", "entries", "path", "fetch"]])
missing = chk[chk.needed & ~chk.found]
for _, r in missing.iterrows():
    print(f"MISSING {r['input']} ({r['path']}): {r['how to get it']}")
assert missing.empty, f"inputs needed in '{FEATURES}' mode are missing: {missing.input.tolist()} (see above; V77_FETCH=1 downloads METAR / Open-Meteo)"
raw = S.raw_inventory(cfg, hash_organiser=not SMOKE)
display(raw["organiser_files"])"""),
("md", """## 2. Settings

The model settings from the YAML, compared field by field with the settings of the reference run (in smoke mode only
the smoke settings differ)."""),
("code", """s = S.make_settings(cfg, smoke=SMOKE)
OUT = S.run_dir(cfg, SMOKE); print("run folder:", OUT)
print("non-default settings:", json.dumps(S.non_default_settings(s), default=str))
diff = S.settings_vs_reference(s, cfg)
print("fields differing from the reference run:" if len(diff) else "every setting equals the reference run's")
diff"""),
("md", """## 3. Features

`pinned`: the fixed feature table (size and SHA-256 checked). `build`: the table is built from the raw inputs
(movements, field filling, ADS-B clocks, weather, EUROCONTROL statistics, ADS-B derived inputs) and cached. Then the
2025 training rows and the 2026 ranking rows are split."""),
("code", """if FEATURES == "pinned":
    info = S.verify_feature_file(cfg); print(json.dumps(info, indent=1))
    assert info["exists"] and info["bytes_ok"] and info.get("sha256_ok", True), "the feature file is not the pinned one"
else:
    info = {"mode": "build"}; print("building the feature table from the raw inputs (cached afterwards under data/features/)")
ft = S.feature_table(cfg, s); f, tr, rk, cat_levels, meta = ft["f"], ft["tr"], ft["rk"], ft["cat_levels"], ft["meta"]
info["features_file"] = ft["hashes"].get("features_file", "built"); info["columns"] = f.shape[1]
print(f"features: {f.shape[0]:,} departures x {f.shape[1]} columns; train rows used {len(tr):,} (sample_days {s.sample_days}); ranking rows {len(rk):,}")"""),
("md", """## 4. What the model reads

Every feature column by family, whether the main model uses it, and its missing share for departures with and without
an NM flight plan."""),
("code", """inv = S.feature_inventory(f, s); inv.to_csv(OUT / "feature_inventory.csv", index=False)
print(f"{len(inv)} columns; model A inputs: {int(inv.model_A_input.sum())}")
summary = inv.groupby(["family", "role"]).agg(columns=("column", "size"), missing_nm=("missing_nm", "mean"), missing_no_nm=("missing_no_nm", "mean")).round(3)
summary"""),
("code", """inv[inv.role == "model A input"].groupby("family").size().sort_values(ascending=False).to_frame("model A columns")"""),
("md", """## 5. Data overview

Departures per airport, the share without an NM flight plan, label quantiles, and Rome's departures without an NM
flight plan by their gap to the schedule."""),
("code", """eda = S.eda_tables(tr, rk)
display(eda["rows_by_airport_set"]); display(eda["no_nm_share_by_airport"])
print("plausible-label view drops", eda["plausible_view_dropped_rows"], "training rows (label more than 12 h over the schedule gap)")
display(eda["label_quantiles_s"])
display(eda["lirf_no_nm_by_gap_band"])
for p in S.eda_plots(tr, OUT): print("plot:", p)"""),
("code", """from IPython.display import Image, display as _d
for p in sorted(OUT.glob("eda_*.png")): _d(Image(filename=str(p), width=700))"""),
("md", """## 6. Training, out of fold: regime, anchor, main model, neighbours, LightGBM, experts

One month of 2025 is held out. The other eleven months are cross-validated in contiguous month blocks: the regime
classifier, the anchor, the main CatBoost model on the anchor residual, the runway-neighbour refit, LightGBM and the
blend, and the airport experts. Every number is out of fold."""),
("code", """month_results = {}
for tm in s.holdout_months:
    sp = S.dev_split(tr, s, tm); dev, test, folds = sp["dev"], sp["test"], sp["folds"]
    print(f"test month {tm}: development rows {len(dev):,} in month blocks {folds}; test rows {len(test):,}")
    cv = S.cv_stages_2_to_6(dev, folds, s, cat_levels)
    display(S.stage_table(dev, cv["oof"], ["anchor", "model_A", "model_A_neighbours", "lightgbm", "blend", "experts"]))
    print("blend weight model A:", round(cv["base_choices"]["blend_weight_model_A"], 4), "| expert shares:", cv["base_choices"]["expert_weights"])
    month_results[tm] = {"split": sp, "cv": cv}"""),
("md", """## 7. Training, out of fold: specialists and ADS-B gates

Specialists for departures without an NM flight plan (kept per airport where they win out of fold) and the ADS-B
gates. These choices are then fixed."""),
("code", """for tm, r in month_results.items():
    top = S.cv_stages_7_to_8(r["split"]["dev"], r["split"]["folds"], r["cv"], s)
    r["oof"], r["choices"] = top["oof"], top["choices"]
    display(S.stage_table(r["split"]["dev"], r["oof"]))
    ch = r["choices"]; print("specialists:", ch["specialist_airports"], "| gates:", {k: v["airports"] for k, v in ch["gates"].items()})"""),
("md", """## 8. Held-out month

The stack trained on the other eleven months predicts the held-out month once: the honest check of the model, stage by
stage, next to the reference run's numbers."""),
("code", """for tm, r in month_results.items():
    res = S.fit_and_predict_month(r["split"]["dev"], r["split"]["test"], r["oof"], r["choices"], s, cat_levels, meta, OUT, tm)
    r.update(res)
    print(f"held-out month {tm}:"); display(S.stage_table(r["split"]["test"], r["pred"]))
hd = S.assemble_holdout(tr, s, month_results)
rep = S.holdout_report(hd, s, meta, OUT, cfg)
print("regime check:", json.dumps(rep["regime_check"], default=float))
cmp_hold = S.compare_holdout(rep, cfg); display(cmp_hold)
if SMOKE: print("smoke run: small models on 5% of the days, the numbers only check that every step runs")"""),
("md", """## 9. Final fit and predictions

The stack is refitted on all of 2025 with the fixed choices and predicts the ranking set; the file is written in the
template's order."""),
("code", """subm = S.submission_pass(tr, rk, hd, s, cat_levels, meta, OUT)
print("ranking stage means (s):", {k: round(v, 1) for k, v in subm["report"]["ranking_stage_means"].items()})
print("choices:", json.dumps({k: subm["choices"][k] for k in ("blend_weight_model_A", "specialist_airports", "expert_weights")}, default=float))"""),
("md", """## 10. LOBT window and the output file

On 2025, every departure with a LOBT has its off-block within a fixed window of it; predictions outside that window
are moved to its nearest edge."""),
("code", """prop = S.lobt_property_2025(cfg); print(json.dumps(prop, indent=1))
assert prop["exceptions_over_half_width"] == 0, "the LOBT property does not hold on 2025 - do not apply the window"
final, win = S.lobt_window(subm["sub"], cfg, subm["template_dtype"]); print(json.dumps(win, indent=1))
final.to_parquet(OUT / "submission_lobt_window.parquet", index=False)
print("final file:", OUT / "submission_lobt_window.parquet", "| rows", len(final), "| dtype", final.iloc[:, 1].dtype)"""),
("md", """## 11. Published daily totals

EUROCONTROL publishes, per airport and day, the total pre-departure delay of the day's departures; on 2025 it equals the
sum over the day of max(0, take-off - schedule - taxi-out). Each day's predictions are reconciled with that total
(an equal shift of the day's delayed departures), every departure of the day then responds to the day's shift with a
loading fitted on this run's out-of-fold predictions, and a Rome / Istanbul departure without a flight plan whose
flight has a plan at the destination is kept outside the window around that plan's off-block. Skipped when the daily
table is not among the inputs."""),
("code", """from v77 import postprocess as PP
ppfit = PP.fit(pd.read_parquet(OUT / "dev_oof.parquet", columns=["MVT_ID_mvt", "final"]), cfg)
print("fit:", json.dumps({k: v for k, v in ppfit.items() if k != "loadings"}, indent=1))
final, pprep = PP.apply(final, ppfit, cfg); print("apply:", json.dumps(pprep, indent=1))
final.to_parquet(OUT / "submission_postprocessed.parquet", index=False); print("final file:", OUT / "submission_postprocessed.parquet")"""),
("md", """## 12. Results

The main numbers of this run into `results.yaml` in the run folder."""),
("code", """results = {"mode": "smoke" if SMOKE else "full", "features_mode": FEATURES, "run_folder": str(OUT), "settings_non_default": S.non_default_settings(s),
           "feature_file": info, "holdout_stage_rmse": {k: rep["holdout"][k]["all"] for k in STAGES},
           "holdout_final": {k: rep["holdout"]["final"][k] for k in ("all", "plausible", "labels_le_2h", "nm_outside_lirf", "nm_lirf", "no_nm", "no_nm_lirf", "top10_sse_share", "rows")},
           "regime_check": rep["regime_check"], "vs_reference_holdout": cmp_hold, "submission_choices": subm["choices"],
           "ranking_stage_means": subm["report"]["ranking_stage_means"], "lobt_property_2025": prop, "lobt_window": win,
           "postprocess_fit": {k: v for k, v in ppfit.items() if k != "loadings"}, "postprocess": pprep}
p = S.write_results(OUT, results); print("wrote", p)
import yaml; back = yaml.safe_load(p.read_text(encoding="utf-8")); print(yaml.safe_dump({k: back[k] for k in ("mode", "holdout_stage_rmse", "lobt_window")}, sort_keys=False))"""),
]


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    p = REPO / "notebooks" / "v77_pipeline.ipynb"; p.parent.mkdir(exist_ok=True)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print("wrote", build())
