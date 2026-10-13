# results/

Runs write to `results/<run>/` (not in git). Kept in git: `strong-crane_final.parquet`, the submitted final file
(670,790 departures; SHA-256 `8776bb6cf4bbe9dfe0e539672cf762698dc0bf4c77f5b0c4339aabb703a8856b`; uploaded 14 October 2026).

- **February and June** (325,949 rows): round(0.5 x the earlier stack's predictions + 0.5 x this pipeline's
  predictions), the LOBT window, then step 6 of the README (daily reconciliation, loaded day shift, orphan window).
  The parts are in `data/final_parts/`; `make final FINAL_ORGANISER=<folder>` rebuilds these rows and checks the file.
- **January and July** (344,841 rows): a score-weighted ensemble of our scored submission files, calibrated on the
  ranking set (paper, Section 6.5), recomputed with the post-processed scored files (211.73 official); its output is `data/final_parts/janjul_final.parquet`, the member files are not included.
