# SPDX-License-Identifier: GPL-3.0-only
"""Execute notebooks/v77_pipeline.ipynb top to bottom (nbclient) and save the executed copy under results/v77/.
    python src/v77/run_notebook.py --smoke             # 5% of the days, small models: does every stage run?
    python src/v77/run_notebook.py                     # the full reproduction on the pinned feature table (~12.5 h, GPU)
    python src/v77/run_notebook.py --features build    # the same, with stage 1 built from the raw inputs first
    python src/v77/run_notebook.py --hedge             # with the v77 Rome hedge options
Exit code 0 only if every cell ran. The kernel is the environment this script runs in.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

V77 = Path(__file__).resolve().parent; REPO = V77.parents[1]                      # src/v77 -> repository root


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--features", choices=("pinned", "build"), help="feature table: the pinned reference file, or built from the raw inputs (default: the YAML)")
    ap.add_argument("--hedge", action="store_true", help="enable the v77 Rome hedge options (hedge_options in the YAML)")
    a = ap.parse_args()
    os.environ["V77_SMOKE"] = "1" if a.smoke else "0"
    if a.features:
        os.environ["V77_FEATURES"] = a.features
    if a.hedge:
        os.environ["V77_HEDGE"] = "1"
    nb = nbformat.read(REPO / "notebooks" / "v77_pipeline.ipynb", as_version=4)
    out = REPO / "results" / "v77"; out.mkdir(parents=True, exist_ok=True)
    dest = out / f"executed_{'smoke' if a.smoke else 'full'}_{time.strftime('%Y%m%dT%H%M%S')}.ipynb"
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(REPO)}},
                            allow_errors=False, record_timing=True)
    t0 = time.time()
    try:
        client.execute()
        ok = True
    except Exception as e:                                    # the executed copy keeps the traceback of the failing cell
        print(f"FAILED: {type(e).__name__}: {str(e)[:2000]}", flush=True); ok = False
    nbformat.write(nb, dest)
    print(f"{'ok' if ok else 'failed'} in {time.time() - t0:.0f} s -> {dest}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
