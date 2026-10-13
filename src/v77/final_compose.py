# SPDX-License-Identifier: GPL-3.0-only
"""Rebuild the submitted final file (11 Oct 2026; 670,790 rows, blind, receipt Succeeded) and check it.

    python src/v77/final_compose.py --organiser <final organiser folder>          # or PRC_ORGANISER=<...>; make final

The final organiser folder holds the 2025 training files, final_ranking.parquet as ranking.parquet and
final_submitting.parquet as submitting.parquet (January, February, June, July 2026; 670,790 departures).

  February/June rows (325,949)  option B = round(0.5 x v3part + 0.5 x blend_dis75), then the LOBT window, where
                                  v3part      = the earlier stack (prc2026-clean tag v3, commit facec86, unchanged)
                                                rerun on the final organiser files (July 2025 holdout 263.67)
                                  blend_dis75 = 0.75 x final_v7f4_dis_s8 + 0.25 x final_v7f4, LOBT window (src/v77/v77.yaml
                                                final_phase)
                                then stage 12 (src/v77/postprocess.py: daily reconciliation with EUROCONTROL's published
                                pre-departure delay, loaded day shift, orphan window; loadings data/final_parts/postprocess_fit.json)
  January/July rows (344,841)   a score-weighted ensemble of our scored submission files (weights fitted to their official
                                scores, i.e. calibrated on the ranking set; paper Section 6.5), post-processed the same way.
                                The member files are not included, so these rows are taken from the shipped final file (--janjul).

Shipped part files (data/final_parts/, each ~4 MB): v3_rerun_final.parquet (670,790 rows), blend_dis75_final.parquet,
option_B_final.parquet (the built option B). Writes results/final_compose/option_B_final.parquet and the final file,
checks the February/June rows against the shipped option B and the SHA-256 of the final file; a different
pandas/pyarrow may change the bytes but not the values (also checked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from taxiout.config import KEY, LABEL, paths   # noqa: E402
from v77 import postprocess as PP               # noqa: E402
from v77.stages import load_config               # noqa: E402

N_FINAL, N_JANJUL = 670790, 344841
PARTS = REPO / "data" / "final_parts"; FINAL = REPO / "results" / "strong-crane_final.parquet"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def col(p: Path, idx: np.ndarray) -> np.ndarray:
    return pd.read_parquet(p).set_index(KEY)[LABEL].astype(float).reindex(idx).to_numpy()


def option_b(tmpl: pd.DataFrame, v3: Path, blend: Path) -> pd.DataFrame:
    """round(0.5 x v3part + 0.5 x blend_dis75), LOBT window, template order (as src/v77/final_postprocess.py writes it)."""
    from v77 import stages as S
    idx = tmpl[KEY].to_numpy(); v3part = col(v3, idx); b = col(blend, idx)
    assert np.isfinite(v3part).all() and np.isfinite(b).all(), "a part file does not cover the final rows"
    pre = pd.DataFrame({KEY: idx, LABEL: np.clip(np.round(0.5 * v3part + 0.5 * b), 0, None)})
    win, rep = S.lobt_window(pre, S.load_config(), tmpl[LABEL].dtype)
    out = tmpl[[KEY]].merge(win, on=KEY, how="left"); out[LABEL] = out[LABEL].astype(tmpl[LABEL].dtype)
    print(f"option B: LOBT window moved {rep['rows_moved']} rows")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organiser", default=None, help="final organiser folder (else PRC_ORGANISER / configs/paths.json)")
    ap.add_argument("--janjul", default=str(FINAL), help="file whose January/July rows are used (default: the shipped final file)")
    ap.add_argument("--v3-rerun", default=str(PARTS / "v3_rerun_final.parquet"))
    ap.add_argument("--blend", default=str(PARTS / "blend_dis75_final.parquet"))
    ap.add_argument("--febjun", default=None, help="use this option-B file instead of building it (e.g. data/final_parts/option_B_final.parquet)")
    ap.add_argument("--out", default=str(REPO / "results" / "strong-crane_final.parquet"))
    a = ap.parse_args()
    if a.organiser:
        os.environ["PRC_ORGANISER"] = a.organiser
    tmpl = pd.read_parquet(paths()["organiser"] / "submitting.parquet")
    assert len(tmpl) == N_FINAL and tmpl[KEY].is_unique, f"submitting.parquet has {len(tmpl):,} rows: point --organiser at the final organiser folder"
    work = REPO / "results" / "final_compose"; work.mkdir(parents=True, exist_ok=True)
    if a.febjun:
        pb = Path(a.febjun); b = pd.read_parquet(pb)
    else:
        b = option_b(tmpl, Path(a.v3_rerun), Path(a.blend)); pb = work / "option_B_final.parquet"; b.to_parquet(pb, index=False)
    assert len(b) == N_FINAL and (b[KEY].to_numpy() == tmpl[KEY].to_numpy()).all(), "option B is not in final_submitting order"
    cfg = load_config(); SHA_FINAL = cfg["final_phase"]["final_file"]["sha256"]
    fitted = json.loads((PARTS / "postprocess_fit.json").read_text(encoding="utf-8"))
    rk = pd.read_parquet(paths()["organiser"] / "ranking.parquet", columns=[KEY, "MVT_TIME_UTC_mvt"]).drop_duplicates(KEY).set_index(KEY)
    month = pd.to_datetime(rk["MVT_TIME_UTC_mvt"], utc=True).dt.month.reindex(tmpl[KEY].to_numpy()).to_numpy(); m = np.isin(month, (1, 7))
    assert m.sum() == N_JANJUL, f"{m.sum():,} January/July rows, expected {N_JANJUL:,}"
    ens = PARTS / "janjul_ensemble.parquet"                                 # the ensemble's January/July rows before stage 12
    src0 = pd.read_parquet(ens if ens.exists() else a.janjul).set_index(KEY)[LABEL]
    b.loc[m, LABEL] = src0.reindex(b.loc[m, KEY]).to_numpy(float)         # they take part in stage 12 (a 1 July departure can
    steps = tuple(cfg["final_phase"]["final_file"].get("postprocess_steps", ["reconcile", "loaded", "orphan"]))
    b, ppr = PP.apply(b, fitted, cfg, steps=steps); print("stage 12", steps, ":", json.dumps(ppr))  # belong to the 30 June group)
    fin = PARTS / "janjul_final.parquet"                                   # the ensemble recomputed with the post-processed files
    if fin.exists():
        a.janjul = str(fin)
    elif ens.exists():                                                      # else every row of the file comes out of stage 12
        a.janjul = str(work / "janjul_from_stage12.parquet"); b[m].to_parquet(a.janjul, index=False)
    assert ppr.get("status") == "applied", "stage 12 needs the EUROCONTROL daily table (paths eurocontrol)"
    ref_b = PARTS / "option_B_final.parquet"
    if not a.febjun and ref_b.exists():                         # February/June rows only: the shipped option B's January/July rows are not used
        pre = pd.read_parquet(pb)[LABEL].to_numpy(float)[~m]; shipped = pd.read_parquet(ref_b)[LABEL].to_numpy(float)[~m]
        same_b = bool(np.array_equal(pre, shipped)); print("option B February/June values equal the shipped option B:", same_b)
    src = pd.read_parquet(a.janjul).set_index(KEY)[LABEL]
    jj = src.reindex(tmpl[KEY].to_numpy()[m]); assert jj.notna().all(), "the January/July file does not cover the January/July rows"
    final = b.copy()
    final.loc[m, LABEL] = jj.reindex(final.loc[m, KEY]).to_numpy(float)
    final[LABEL] = final[LABEL].astype(tmpl[LABEL].dtype)
    assert final[LABEL].notna().all() and (final[LABEL] >= 0).all() and (final[KEY].to_numpy() == tmpl[KEY].to_numpy()).all()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); final.to_parquet(out, index=False)
    rep = {"rows": len(final), "janjul_rows_from": Path(a.janjul).name, "janjul_rows": int(m.sum()), "febjun_rows": int((~m).sum()),
           "option_B": pb.relative_to(REPO).as_posix() if pb.is_relative_to(REPO) else str(pb), "febjun_values_equal_shipped_option_B": same_b,
           "final": out.relative_to(REPO).as_posix() if out.is_relative_to(REPO) else str(out), "final_sha256": sha(out), "final_sha256_expected": SHA_FINAL, "final_sha256_match": sha(out) == SHA_FINAL,
           "mean_by_part_s": {"janjul": float(final[LABEL][m].mean()), "febjun": float(final[LABEL][~m].mean())}}
    (work / "final_compose.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
