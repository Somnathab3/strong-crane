# SPDX-License-Identifier: GPL-3.0-only
"""Post-process a run on the final dataset into the two submission files (8 Oct 2026).

    PRC_ORGANISER=<final organiser folder> python src/v77/final_postprocess.py <run folder> <tag>

Reads <run>/submission.parquet (predictions for every final_ranking departure), applies the LOBT window (a data property,
read from the final ranking file) and writes, under results/final_runs/<tag>/:
  strong-crane_final_<tag>.parquet   all 670,790 departures, aligned to final_submitting.parquet (the single final submission)
  strong-crane_janjul_<tag>.parquet  the January + July 2026 departures only, aligned to the original submitting.parquet
                                     (344,841 rows), so the result can be compared with earlier submissions on the ranking set;
                                     that template is paths()["ranking_template_jan_jul"] (configs/paths.json key
                                     ranking_template_jan_jul or PRC_RANKING_TEMPLATE_JAN_JUL); not set -> this file is skipped
  postprocess.json                   row counts, LOBT-window moves, SHA-256, per-month means
Nothing is uploaded."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from taxiout.config import KEY, LABEL, paths   # noqa: E402
from v77 import stages as S                    # noqa: E402

def main(run: Path, tag: str) -> None:
    out = REPO / "results" / "final_runs" / tag; out.mkdir(parents=True, exist_ok=True)
    tmpl = pd.read_parquet(paths()["organiser"] / "submitting.parquet")
    sub = pd.read_parquet(run / "submission.parquet")
    assert len(sub) == len(tmpl) == 670790, f"expected 670,790 final rows, got {len(sub):,} / {len(tmpl):,}"
    final, win = S.lobt_window(sub, S.load_config(), tmpl[LABEL].dtype)
    final = tmpl[[KEY]].merge(final, on=KEY, how="left")
    assert final[LABEL].notna().all() and (final[LABEL] >= 0).all()
    final[LABEL] = final[LABEL].astype(tmpl[LABEL].dtype)
    p_final = out / f"strong-crane_final_{tag}.parquet"; final.to_parquet(p_final, index=False)
    p_jj, old_tmpl = None, paths()["ranking_template_jan_jul"]
    if old_tmpl is not None:
        old = pd.read_parquet(old_tmpl)
        jj = old[[KEY]].merge(final, on=KEY, how="left")
        assert len(jj) == 344841 and jj[LABEL].notna().all(), "January/July rows missing from the final predictions"
        jj[LABEL] = jj[LABEL].astype(old[LABEL].dtype)
        p_jj = out / f"strong-crane_janjul_{tag}.parquet"; jj.to_parquet(p_jj, index=False)
    raw = pd.read_parquet(paths()["organiser"] / "ranking.parquet", columns=[KEY, "MVT_TIME_UTC_mvt"])
    month = final.merge(raw.drop_duplicates(KEY), on=KEY).assign(m=lambda d: pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.month)
    rep = {"run": str(run), "tag": tag, "lobt_window": win, "final_rows": len(final), "janjul_rows": len(jj) if p_jj else 0,
           "final_sha256": hashlib.sha256(p_final.read_bytes()).hexdigest(), "janjul_sha256": hashlib.sha256(p_jj.read_bytes()).hexdigest() if p_jj else None,
           "mean_by_month_s": {int(k): round(float(v), 1) for k, v in month.groupby("m")[LABEL].mean().items()},
           "median_by_month_s": {int(k): float(v) for k, v in month.groupby("m")[LABEL].median().items()}}
    (out / "postprocess.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
