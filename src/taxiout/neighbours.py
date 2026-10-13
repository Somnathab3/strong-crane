# SPDX-License-Identifier: GPL-3.0-only
"""Stage 5: the runway neighbourhood. If the flights around me on my runway are running late, I probably am too.

For each departure: the mean *predicted residual* (stage-4 prediction minus anchor, out-of-fold on 2025 rows) of the
other departures on the same runway within +-w minutes (w = 5, 10, 20, 30, 60), before-only and after-only means at
30 min, the neighbour counts, and each mean minus the row's own predicted residual. Predicted, not true, residuals:
true labels of neighbours are not available on ranking rows, so only predictions can be used on both sides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
#: v7: airports whose neighbour features are set missing (set from Settings.nb_skip_airports by run / refit / predict).
#: At LIRF the neighbours' predicted residuals carry schedule-stamped flights, which pulled ordinary taxis long
#: (quick split July + December 2025, two seeds: ordinary LIRF RMSE +12 / +27 with them, hubs -2.4 / -1.0).
SKIP_AIRPORTS: tuple = ()


def features(f: pd.DataFrame, resid: np.ndarray, pred: np.ndarray | None = None) -> pd.DataFrame:
    """pred (optional): the stage-4 predicted taxi time itself; adds the neighbours' mean *level* (nb_level_pm30/60,
    self excluded). A flight without an NM record gets an airport-median anchor, so its residual says little about
    the moment; its neighbours' predicted taxi time does (EHAM 2-7 Jan 2026 snow disruption: NM flights around it were
    predicted 1,800-2,700 s while the no-NM ones were pushed a further ~1,000 s by pooled no-NM patterns)."""
    out = _features(f, resid)
    if pred is not None:
        lv = _features(f, pred, windows=(30, 60), sides=False)
        out = pd.concat([out, lv.rename(columns=lambda c: c.replace("nb_pm", "nb_level_pm")).filter(like="_mean")], axis=1)
    if SKIP_AIRPORTS:
        out.loc[f.airport.astype(str).isin(SKIP_AIRPORTS).to_numpy(), :] = np.nan
    return out


def _features(f: pd.DataFrame, resid: np.ndarray, windows=WINDOWS, sides=True) -> pd.DataFrame:
    key = pd.factorize(f.airport.astype(str) + ":" + f.runway.astype(str))[0].astype(np.int64)
    t = np.round(f.t.to_numpy()).astype(np.int64)
    comp = key * 10 ** 11 + t
    order = np.argsort(comp, kind="stable"); cs_sorted = comp[order]
    r = np.nan_to_num(resid[order]); ok = np.isfinite(resid[order]).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(r)]); ccnt = np.concatenate([[0.0], np.cumsum(ok)])
    own_r = np.nan_to_num(resid); own_ok = np.isfinite(resid).astype(float)
    out = {}

    def window(lo_s, hi_s, name, exclude_self):
        a = np.searchsorted(cs_sorted, comp + lo_s, "left"); b = np.searchsorted(cs_sorted, comp + hi_s, "right")
        s = csum[b] - csum[a]; n = ccnt[b] - ccnt[a]
        if exclude_self:
            s = s - own_r; n = n - own_ok
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out[f"nb_{name}_mean"] = mean; out[f"nb_{name}_n"] = n; out[f"nb_{name}_minus_own"] = mean - resid

    for w in windows:
        window(-w * 60, w * 60, f"pm{w}", True)
    if sides:
        window(-30 * 60, -1, "before30", False)
        window(1, 30 * 60, "after30", False)
    return pd.DataFrame(out, index=f.index)
