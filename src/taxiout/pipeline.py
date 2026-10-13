# SPDX-License-Identifier: GPL-3.0-only
"""The stack as three functions, so the same code serves validation, submission and prediction.

    cv_stack(rows, folds, s)          out-of-fold predictions of every stage on `rows`, and the choices made on them
                                      (blend weight, specialist airports, gate airports)
    fit_stack(rows, oof, choices, s)  every stage fitted on all of `rows`; stacked stages (5, 7, 8) learn from `oof`
    predict_stack(bundle, new)        every stage's prediction on new rows (2025 holdout months or the 2026 ranking)

Honest evaluation (run.py): the shipped design (src/v77/v77.yaml, `--months 7`) holds out July 2025. `cv_stack` runs
on the other eleven months only, in four contiguous month blocks (1-3, 4-6, 8-10, 11-12); every choice is made there and
frozen; `fit_stack` on the eleven months predicts July once. (`--months 1,7` crosses two outer tests the same way.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import lirf_regime, neighbours
from .anchor import Anchor
from .config import Settings
from .model import (AvgModel, CATS, GATES, TRIM_S, fit_expert, blend_weight, fit_gate, fit_lgb, fit_main, fit_specialist, gate_rows, main_frame,
                    predict_gate, predict_main, regime_fit, regime_p, regime_predict, to_lgb)

STAGES = ["anchor", "model_A", "model_A_neighbours", "lightgbm", "blend", "experts", "specialist", "final"]
#: stage 6b mixing weights tried per airport (expert share); chosen out of fold on labels <= 2 h
EXPERT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def rmse(e) -> float:
    e = np.asarray(e, float); return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def spec_x(frame, a_, g_, ex_, pre_):
    """Stage-7 inputs: the main inputs plus the blend's prediction."""
    extra = pd.DataFrame({"pre": pre_}, index=frame.index)
    if ex_ is not None:
        extra = pd.concat([ex_.set_axis(frame.index), extra], axis=1)
    return main_frame(frame, a_, g_, extra)


#: a gate (or a fold of it) with fewer ADS-B evidence rows than this is not fitted (tests/test_pretrain.py found the
#: crash on a single-airport sample: CatBoost refuses a target with one value)
MIN_GATE_ROWS = 100
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")


def blend_by_airport(a: np.ndarray, b: np.ndarray, airport: np.ndarray, w: float, w_ap: dict) -> np.ndarray:
    ww = np.array([w_ap.get(x, w) for x in airport])
    return ww * a + (1 - ww) * b


def gate_airports(gain: dict, rows: dict, s: Settings) -> list:
    """Where a gate is applied. Default ON (the gate is pooled over airports, with the airport as an input), OFF only
    where the out-of-fold error shows it hurts on >= gate_min_evidence rows, and never at the airports in gate_skip
    (LFPG: its ADS-B first points are the inbound landing). Why default ON: ADS-B coverage flipped between 2025 and 2026
    at several airports (EHAM no-NM 1% -> 62% in January, LEMD NM 1% -> 96%, LIRF July 96% -> 23%), so an airport
    with little 2025 evidence says nothing about 2026; the earlier stack's no-NM gate, applied everywhere outside
    LIRF/LFPG, gained most at EHAM and LEBL."""
    out = []
    for ap_ in AIRPORTS:
        if ap_ in s.gate_skip:
            continue
        if rows.get(ap_, 0) < s.gate_min_evidence or gain.get(ap_, 0.0) >= 0:
            out.append(ap_)
    return out


def _pick(airport: pd.Series, mask: np.ndarray, before: np.ndarray, after: np.ndarray, y: np.ndarray) -> tuple[list, dict]:
    g = pd.DataFrame({"ap": airport.to_numpy(), "a": np.where(mask, (before - y) ** 2, 0), "b": np.where(mask, (after - y) ** 2, 0)}).groupby("ap").sum()
    return sorted(g.index[g.b < g.a]), {k: float(g.a[k] - g.b[k]) for k in g.index}


#: stage-7 variants: (population, target). "resid": learn label - blend. "delay": learn the off-block delay
#: (off-block - schedule = schedule gap - label; zero when the block was stamped at the schedule) and predict
#: label = schedule gap - delay: the target an earlier LIRF rule used, bimodal rather than spread over 0-30 h.
SPECIALISTS = {"all_resid": (lambda f: np.ones(len(f), bool), "resid"),
               "lirf_resid": (lambda f: (f.airport == "LIRF").to_numpy(), "resid"),
               "lirf_delay": (lambda f: (f.airport == "LIRF").to_numpy(), "delay")}
#: "lirf_mix": the average of lirf_delay and lirf_resid (built from the two fitted variants, chosen on the development
#: months like the others): less confident than the delay target alone, which over-predicted LIRF's high-gap no-NM
#: rows (the stamped share there moves with the season: 33-76% by month in 2025)
MIXES = {"lirf_mix": ("lirf_delay", "lirf_resid")}


def _spec_target(kind, y, pre, sched):
    return y - pre if kind == "resid" else sched - y


def _spec_apply(kind, raw, pre, sched):
    return np.clip(pre + raw, 0, None) if kind == "resid" else np.clip(sched - raw, 0, None)


def _choose_specialist(airport, nn, base, cands, y, fm=None, min_folds: int = 0, max_top3: float = 1.0) -> tuple[dict, dict]:
    """Per airport, the variant with the lowest out-of-fold SSE on its no-NM rows, if it beats no specialist.
    v4 (after v3's LTFM pick hurt on the holdout): the variant must also lower the SSE in at least `min_folds` of the
    inner folds, and its top 3 rows may carry at most `max_top3` of its gain; otherwise the airport gets none."""
    choice, gain = {}, {}
    for ap_ in sorted(airport[nn].unique()):
        r = nn & (airport == ap_).to_numpy() & np.isfinite(y)
        sse0 = float(((base[r] - y[r]) ** 2).sum()); best = (sse0, None)
        for name, c in cands.items():
            v = float(((c[r] - y[r]) ** 2).sum())
            if v < best[0]:
                best = (v, name)
        if not best[1]:
            continue
        c = cands[best[1]]; d = (base - y) ** 2 - (c - y) ** 2
        folds_won = sum(1 for _, te in (fm or []) if (r & te).any() and d[r & te].sum() > 0)
        top3 = float(np.sort(d[r])[::-1][:3].sum() / max(d[r].sum(), 1e-9))
        ok = folds_won >= min_folds and top3 <= max_top3
        gain[ap_] = {"variant": best[1], "sse_removed": sse0 - best[0], "folds_won": folds_won, "top3_share": top3, "used": ok}
        if ok:
            choice[ap_] = best[1]
    return choice, gain


def cv_stack(tr: pd.DataFrame, folds: list[tuple], s: Settings, cat_levels: dict, log) -> tuple[dict, dict]:
    oof, nb, base_ch = cv_base(tr, folds, s, cat_levels, log)
    return cv_top(tr, folds, oof, nb, base_ch, s, log)


def cv_base(tr: pd.DataFrame, folds: list[tuple], s: Settings, cat_levels: dict, log) -> tuple[dict, pd.DataFrame | None, dict]:
    """Stages 2-6b out of fold: regime, anchor, model A, neighbours, LightGBM, blend, airport experts."""
    n = len(tr); y = tr.label.to_numpy()
    fm = [(fold, tr.month.isin(fold).to_numpy()) for fold in folds]
    oof = {k: np.full(n, np.nan) for k in STAGES}; grp = np.empty(n, dtype=object); ANC = {}; anchor_tables = {}
    for fold, te in fm:
        fit = ~te
        log(f"  fold {fold}: stage 2 regime, stage 3 anchor")
        p = regime_p(tr, s, fit)
        A = Anchor().fit(tr[fit]); aa, gg = A.predict(tr, p); ANC[fold] = (aa, gg); anchor_tables[str(fold)] = A.table()
        oof["anchor"][te] = aa[te]; grp[te] = gg[te]
        log(f"  fold {fold}: stage 4 model A on {int(fit.sum()):,} rows")
        m = fit_main(tr[fit], aa[fit], gg[fit], s); oof["model_A"][te] = predict_main(m, tr[te], aa[te], gg[te])
        log(f"  fold {fold}: anchor {rmse(aa[te] - y[te]):.2f} -> model A {rmse(oof['model_A'][te] - y[te]):.2f}")
    nb = neighbours.features(tr, oof["model_A"] - oof["anchor"], oof["model_A"]) if s.neighbours else None
    ex_pred = np.full(n, np.nan)
    for fold, te in fm:
        aa, gg = ANC[fold]; fit = ~te
        ex_f, ex_t = (nb[fit], nb[te]) if nb is not None else (None, None)
        log(f"  fold {fold}: stage 5 model A + runway neighbours, stage 6 LightGBM")
        m = fit_main(tr[fit], aa[fit], gg[fit], s, ex_f); oof["model_A_neighbours"][te] = predict_main(m, tr[te], aa[te], gg[te], ex_t)
        L = fit_lgb(main_frame(tr[fit], aa[fit], gg[fit], ex_f), np.clip(y[fit] - aa[fit], -s.residual_clip, s.residual_clip), s, cat_levels)
        oof["lightgbm"][te] = np.clip(aa[te] + L.predict(to_lgb(main_frame(tr[te], aa[te], gg[te], ex_t), cat_levels)), 0, None)
        log(f"  fold {fold}: + neighbours {rmse(oof['model_A_neighbours'][te] - y[te]):.2f}; LightGBM {rmse(oof['lightgbm'][te] - y[te]):.2f}")
        if s.experts:
            for ap_ in AIRPORTS:
                ra = (tr.airport == ap_).to_numpy(); rf = fit & ra & np.isfinite(y); rt = te & ra
                if rf.sum() < s.expert_min_rows or not rt.any():
                    continue
                E = fit_expert(tr[rf], aa[rf], gg[rf], s, nb[rf] if nb is not None else None)
                ex_pred[rt] = predict_main(E, tr[rt], aa[rt], gg[rt], nb[rt] if nb is not None else None)
            log(f"  fold {fold}: stage 6b airport experts on {int(np.isfinite(ex_pred[te]).sum()):,} rows")
    w = blend_weight(oof["model_A_neighbours"], oof["lightgbm"], y)
    #: per-airport weights (the first repository measured +0.50 s out of sample for per-airport blend weights), shrunk
    #: toward the global weight: w_ap = (n w_raw + 20,000 w) / (n + 20,000)
    w_ap = {}
    for ap_ in AIRPORTS:
        r = (tr.airport == ap_).to_numpy() & np.isfinite(y)
        if r.sum() >= 1000:
            n_ = r.sum(); w_ap[ap_] = float((n_ * blend_weight(oof["model_A_neighbours"][r], oof["lightgbm"][r], y[r]) + 20000 * w) / (n_ + 20000))
    oof["blend"] = blend_by_airport(oof["model_A_neighbours"], oof["lightgbm"], tr.airport.to_numpy(), w, w_ap)
    #: stage 6b: per airport, the share of the airport expert that lowers the out-of-fold error on ordinary labels
    #: (<= 2 h, so a few extreme rows cannot decide it); 0 keeps the stack (an airport where the expert does not help)
    oof["experts"] = oof["blend"].copy(); expert_w = {}
    for ap_ in (AIRPORTS if s.experts else ()):
        ra = (tr.airport == ap_).to_numpy() & np.isfinite(ex_pred)
        r = ra & np.isfinite(y) & (np.nan_to_num(y, nan=np.inf) <= TRIM_S)
        if r.sum() < 1000:
            continue
        sse = {l: float(np.sum(((1 - l) * oof["blend"][r] + l * ex_pred[r] - y[r]) ** 2)) for l in EXPERT_WEIGHTS}
        lam = min(sse, key=sse.get)
        if lam > 0:
            expert_w[ap_] = lam; oof["experts"][ra] = (1 - lam) * oof["blend"][ra] + lam * ex_pred[ra]
    log(f"  stage 6b expert share by airport: {expert_w or 'none'}")
    oof["group"] = grp
    return oof, nb, {"blend_weight_model_A": w, "blend_weight_by_airport": w_ap, "expert_weights": expert_w, "anchor_tables_by_fold": anchor_tables}


def cv_top(tr: pd.DataFrame, folds: list[tuple], oof: dict, nb: pd.DataFrame | None, base_ch: dict, s: Settings, log) -> tuple[dict, dict]:
    """Stages 7-8 out of fold on top of the base stages' out-of-fold predictions (oof: anchor, experts, group):
    no-NM specialists, ADS-B gates, the rest corrector. Also callable on its own from saved base out-of-fold predictions (tests/test_pretrain.py test_20c)."""
    n = len(tr); y = tr.label.to_numpy(); nn = tr.has_nm.to_numpy() == 0; grp = oof["group"]
    fm = [(fold, tr.month.isin(fold).to_numpy()) for fold in folds]
    oof = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in oof.items()}
    spec_cats = [c for c in CATS if c in tr] + ["anchor_group"]
    cands = {}
    for name, (popf, kind) in SPECIALISTS.items():
        pop = nn & popf(tr); sd = np.full(n, np.nan)
        for fold, te in fm:
            fr = ~te & pop & np.isfinite(y); r = te & pop
            if fr.sum() < 100 or not r.any():
                continue
            log(f"  fold {fold}: stage 7 specialist '{name}' on {int(fr.sum()):,} rows")
            X = spec_x(tr[fr], oof["anchor"][fr], grp[fr], nb[fr] if nb is not None else None, oof["experts"][fr])
            S = fit_specialist(X, _spec_target(kind, y[fr], oof["experts"][fr], tr.sched.to_numpy()[fr]), s, spec_cats)
            raw = S.predict(spec_x(tr[r], oof["anchor"][r], grp[r], nb[r] if nb is not None else None, oof["experts"][r]))
            sd[r] = _spec_apply(kind, raw, oof["experts"][r], tr.sched.to_numpy()[r])
        cands[name] = np.where(np.isfinite(sd), sd, oof["experts"])
    for mname, (v1_, v2_) in MIXES.items():
        if v1_ in cands and v2_ in cands:
            cands[mname] = 0.5 * cands[v1_] + 0.5 * cands[v2_]
    spec_choice, spec_gain = _choose_specialist(tr.airport, nn, oof["experts"], cands, y, fm, s.spec_min_folds, s.spec_max_top3)
    oof["specialist"] = oof["experts"].copy()
    for ap_, name in spec_choice.items():
        r = nn & (tr.airport == ap_).to_numpy(); oof["specialist"][r] = cands[name][r]
    spec_ap = spec_choice
    pre = oof["specialist"]; oof["final"] = pre.copy(); gates = {}
    for which in (("nm", "no_nm") if s.gate_split else ("all",)):
        fit_r, app_r = gate_rows(tr, which, s.gate_genuine_only); delta = np.zeros(n); floor = GATES[which]["floor"]
        if fit_r.sum() < MIN_GATE_ROWS:                     # too little ADS-B evidence to learn a gate at all
            log(f"  gate {which}: {int(fit_r.sum())} rows with evidence, not fitted"); continue
        for fold, te in fm:
            if (fit_r & ~te).sum() < MIN_GATE_ROWS:
                continue
            log(f"  gate {which} fold {fold}: fit on {int((fit_r & ~te).sum()):,} rows")
            g = fit_gate(tr[fit_r & ~te], pre[fit_r & ~te], s, which)
            r = app_r & te; delta[r] = predict_gate(g, tr[r], pre[r])
        new = np.clip(pre + delta, floor, None)
        _, gain = _pick(tr.airport, app_r, pre, new, y)
        rows_ap = pd.Series(app_r & np.isfinite(y)).groupby(tr.airport.to_numpy()).sum().to_dict()
        aps = gate_airports(gain, rows_ap, s)
        use = app_r & tr.airport.isin(aps).to_numpy(); oof["final"][use] = new[use]
        gates[which] = {"airports": aps, "oof_sse_removed_by_airport": gain, "evidence_rows_by_airport": {k: int(v) for k, v in rows_ap.items()}}
    if s.rest_corrector:
        #: the NM rows no ADS-B gate reaches. Unlike the gates it is OFF by default: ON at an airport only where it lowers
        #: the out-of-fold SSE on ordinary labels (<= TRIM_S) with >= gate_min_evidence rows, so a few extreme rows can
        #: neither switch it on nor off
        fit_r, app_r = gate_rows(tr, "rest"); delta = np.zeros(n); pre_r = oof["final"].copy()
        for fold, te in fm:
            if (fit_r & ~te).sum() < MIN_GATE_ROWS:
                continue
            log(f"  corrector rest fold {fold}: fit on {int((fit_r & ~te).sum()):,} rows")
            g = fit_gate(tr[fit_r & ~te], pre_r[fit_r & ~te], s, "rest")
            r = app_r & te; delta[r] = predict_gate(g, tr[r], pre_r[r])
        new = np.clip(pre_r + delta, 0.0, None)
        ordinary = app_r & (np.nan_to_num(y, nan=np.inf) <= TRIM_S)
        _, gain = _pick(tr.airport, ordinary, pre_r, new, y)
        _, gain_all = _pick(tr.airport, app_r & np.isfinite(y), pre_r, new, y)
        rows_ap = pd.Series(ordinary).groupby(tr.airport.to_numpy()).sum().to_dict()
        aps = sorted(a for a, v in gain.items() if v > 0 and rows_ap.get(a, 0) >= s.gate_min_evidence)
        use = app_r & tr.airport.isin(aps).to_numpy(); oof["final"][use] = new[use]
        gates["rest"] = {"airports": aps, "oof_sse_removed_by_airport": gain, "oof_sse_removed_all_labels": gain_all,
                         "evidence_rows_by_airport": {k: int(v) for k, v in rows_ap.items()}}
        if not aps:
            gates.pop("rest")
            log("  corrector rest: no airport improves on ordinary labels; not used")
    choices = {**base_ch, "specialist_airports": spec_ap, "specialist_oof_sse_removed": spec_gain, "gates": gates}
    log(f"  choices (made on these rows only): blend weight {base_ch['blend_weight_model_A']:.3f}; specialist at {spec_ap}; "
        + "; ".join(f"gate {k} at {v['airports']}" for k, v in gates.items()))
    return oof, choices


def fit_stack(tr: pd.DataFrame, oof: dict, ch: dict, s: Settings, cat_levels: dict, log) -> dict:
    """Every stage on all of `tr`. Stages 5, 7 and 8 learn from `oof` (out-of-fold predictions for exactly these rows)."""
    b = fit_base(tr, oof, ch, s, cat_levels, log); nb = b.pop("_nb")
    return {**b, **fit_top(tr, oof, nb, ch, s, log)}


def fit_base(tr: pd.DataFrame, oof: dict, ch: dict, s: Settings, cat_levels: dict, log) -> dict:
    """Stages 2-6b on all of `tr`."""
    y = tr.label.to_numpy()
    log(f"fit on {len(tr):,} rows: regime, anchor")
    p, R = regime_fit(tr, s, np.ones(len(tr), bool))
    A = Anchor().fit(tr); a, g = A.predict(tr, p)
    log("  model A"); m4 = fit_main(tr, a, g, s)
    nb = neighbours.features(tr, oof["model_A"] - oof["anchor"], oof["model_A"]) if s.neighbours else None
    log("  model A + neighbours"); m5 = fit_main(tr, a, g, s, nb)
    log("  LightGBM"); L = fit_lgb(main_frame(tr, a, g, nb), np.clip(y - a, -s.residual_clip, s.residual_clip), s, cat_levels)
    EX = {}
    for ap_ in ch.get("expert_weights", {}):
        r = (tr.airport == ap_).to_numpy() & np.isfinite(y); log(f"  airport expert {ap_} on {int(r.sum()):,} rows")
        EX[ap_] = fit_expert(tr[r], a[r], g[r], s, nb[r] if nb is not None else None)
    return {"regime": R, "anchor": A, "model_A": m4, "model_A_neighbours": m5, "lightgbm": L, "choices": ch, "cat_levels": cat_levels,
            "neighbours": s.neighbours, "experts": EX, "_nb": nb}


def fit_top(tr: pd.DataFrame, oof: dict, nb: pd.DataFrame | None, ch: dict, s: Settings, log) -> dict:
    """Stages 7-8 (specialists, gates, LIRF regime) on all of `tr`, from the base stages' out-of-fold predictions."""
    y = tr.label.to_numpy(); nn = tr.has_nm.to_numpy() == 0
    spec_cats = [c for c in CATS if c in tr] + ["anchor_group"]; SP = {}
    needed = set()
    for n_ in set(ch["specialist_airports"].values()):
        needed |= set(MIXES.get(n_, (n_,)))
    for name in sorted(needed):
        popf, kind = SPECIALISTS[name]; fr = nn & popf(tr) & np.isfinite(y)
        log(f"  specialist '{name}' on {int(fr.sum()):,} rows")
        X = spec_x(tr[fr], oof["anchor"][fr], oof["group"][fr], nb[fr] if nb is not None else None, oof["experts"][fr])
        SP[name] = fit_specialist(X, _spec_target(kind, y[fr], oof["experts"][fr], tr.sched.to_numpy()[fr]), s, spec_cats)
    G = {}
    for which in ch["gates"]:
        fit_r, _ = gate_rows(tr, which, s.gate_genuine_only); log(f"  gate {which}")
        base = oof["final"] if which == "rest" else oof["specialist"]
        G[which] = fit_gate(tr[fit_r], base[fit_r], s, which)
    LR = lirf_regime.fit(tr, s, log, oof.get("final")) if s.lirf_regime else None
    return {"specialists": SP, "gates": G, "lirf_regime": LR, "choices": ch}


def predict_stack(b: dict, new: pd.DataFrame) -> dict:
    out, a, g, nb = predict_base(b, new)
    return predict_top(b, new, out, a, g, nb)


def predict_base(b: dict, new: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Stages 2-6b on new rows; also returns the anchor, its group and the neighbour features the top stages use."""
    ch = b["choices"]; out = {}
    p = regime_predict(b["regime"], new)
    a, g = b["anchor"].predict(new, p); out["p_stamp"] = p; out["anchor"] = a
    out["model_A"] = predict_main(b["model_A"], new, a, g)
    nb = neighbours.features(new, out["model_A"] - a, out["model_A"]) if b["neighbours"] else None
    out["model_A_neighbours"] = predict_main(b["model_A_neighbours"], new, a, g, nb)
    out["lightgbm"] = np.clip(a + b["lightgbm"].predict(to_lgb(main_frame(new, a, g, nb), b["cat_levels"])), 0, None)
    out["blend"] = blend_by_airport(out["model_A_neighbours"], out["lightgbm"], new.airport.to_numpy(), ch["blend_weight_model_A"],
                                    ch.get("blend_weight_by_airport", {}))
    out["experts"] = out["blend"].copy()
    for ap_, lam in ch.get("expert_weights", {}).items():
        r = (new.airport == ap_).to_numpy()
        if r.any() and ap_ in b.get("experts", {}):
            e = predict_main(b["experts"][ap_], new[r], a[r], g[r], nb[r] if nb is not None else None)
            out["experts"][r] = (1 - lam) * out["blend"][r] + lam * e
    return out, a, g, nb


def predict_top(b: dict, new: pd.DataFrame, out: dict, a: np.ndarray, g: np.ndarray, nb: pd.DataFrame | None) -> dict:
    """Stages 7-8 on new rows, from the base stages' predictions in `out` (experts) and the anchor a / group g."""
    ch = b["choices"]; out = dict(out)
    pre = out["experts"].copy(); nnew = new.has_nm.to_numpy() == 0
    for ap_, name in ch["specialist_airports"].items():
        r = nnew & (new.airport == ap_).to_numpy()
        if r.any():
            parts = []
            for nm_ in MIXES.get(name, (name,)):
                raw = b["specialists"][nm_].predict(spec_x(new[r], a[r], g[r], nb[r] if nb is not None else None, out["experts"][r]))
                parts.append(_spec_apply(SPECIALISTS[nm_][1], raw, out["experts"][r], new.sched.to_numpy()[r]))
            pre[r] = np.mean(parts, axis=0)
    out["specialist"] = pre.copy(); fin = pre.copy()
    for which, gm in b["gates"].items():
        _, app = gate_rows(new, which); r = app & new.airport.isin(ch["gates"][which]["airports"]).to_numpy()
        if r.any():
            base = fin if which == "rest" else pre
            fin[r] = np.clip(base[r] + predict_gate(gm, new[r], base[r]), GATES[which]["floor"], None)
    fin, out["p_lirf_regime"] = lirf_regime.apply(b.get("lirf_regime"), new, fin)
    out["final"] = day_shift(new, fin)
    return out


DAY_SHIFT_S = 20 * 3600


def day_shift(new: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """An NM departure whose movement is logged more than 20 h after its IOBT had its movement logged the day after it
    left. 2025 has one such NM departure (LIRF, July: label 87,002 s; movement - IOBT 87,012 s, movement - schedule
    87,001 s), and twelve LIRF no-NM departures labelled 24 h + a taxi (87,168-88,392 s, median 87,420 s), which is what
    the NM clock describes: the aircraft left around its IOBT. The stamp classifier cannot learn this from one row
    (the stack gave such a row p_stamp 0.0004 and predicted 23,021 s), so it is a rule: predict movement - IOBT."""
    iobt = new.nm_iobt.to_numpy() if "nm_iobt" in new else np.full(len(new), np.nan)
    r = (new.has_nm.to_numpy() > 0) & (np.nan_to_num(iobt) > DAY_SHIFT_S)
    out = pred.copy(); out[r] = np.clip(iobt[r], 0, None)
    return out


# ---- the fitted bundle: everything needed to predict without retraining
def save_bundle(b: dict, d: Path, meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    models = b["regime"].get("model")
    models = [] if models is None else (models if isinstance(models, list) else [models])
    for k, m in enumerate(models):
        m.save_model(str(d / f"regime_{k}.cbm"))
    (d / "regime.json").write_text(json.dumps({"op_table": b["regime"]["op_table"], "prior": b["regime"]["prior"], "cal": b["regime"]["cal"],
                                               "const": b["regime"].get("const"), "n_models": len(models),
                                               "raw_fields": b["regime"].get("raw_fields", False)}),
                                   encoding="utf-8")
    for k in ("model_A", "model_A_neighbours"):
        ms = getattr(b[k], "models", None)                                     # v7: seed-averaged -> one file per seed
        if ms is None:
            b[k].save_model(str(d / f"{k}.cbm"))
        else:
            for i, mm in enumerate(ms):
                mm.save_model(str(d / f"{k}_{i}.cbm"))
    for name, m in b["specialists"].items():
        for k, mm in enumerate(getattr(m, "models", [m])):                   # v4.1: averaged specialists save one file per seed
            mm.save_model(str(d / f"specialist_{name}{'' if not hasattr(m, 'models') else f'_{k}'}.cbm"))
    b["lightgbm"].save_model(str(d / "lightgbm.txt"))
    for which, (m, spm) in b["gates"].items():
        m.save_model(str(d / f"gate_{which}.cbm")); (d / f"gate_{which}_spm.json").write_text(json.dumps(spm), encoding="utf-8")
    A = b["anchor"]
    (d / "anchor.json").write_text(json.dumps({"coef": {"|".join(k): list(map(float, c)) for k, c in A.coef.items()},
                                               "median": A.median, "glob": A.glob, "rich": getattr(A, "rich", False),
                                               "fit_rows": getattr(A, "fit_rows", "robust"), "clock_ridge": getattr(A, "clock_ridge", False),
                                               "clock_ridge_models": {k: list(map(float, v)) for k, v in getattr(getattr(A, "cr", None), "m", {}).items()}}),
                                   encoding="utf-8")
    lirf_regime.save(b.get("lirf_regime"), d)
    for ap_, m in b.get("experts", {}).items():
        m.save_model(str(d / f"expert_{ap_}.cbm"))
    ch = {k: v for k, v in b["choices"].items() if k != "anchor_tables_by_fold"}
    (d / "bundle.json").write_text(json.dumps({"choices": ch, "cat_levels": b["cat_levels"], "neighbours": b["neighbours"], **meta},
                                              indent=1, default=float), encoding="utf-8")


def load_bundle(d: Path) -> dict:
    import lightgbm as lgb
    from catboost import CatBoostClassifier, CatBoostRegressor
    j = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    b = {"choices": j["choices"], "cat_levels": j["cat_levels"], "neighbours": j["neighbours"]}
    rj = json.loads((d / "regime.json").read_text(encoding="utf-8"))
    rm_ = [CatBoostClassifier().load_model(str(d / f"regime_{k}.cbm")) for k in range(rj.get("n_models", 0))]
    if not rm_ and (d / "regime.cbm").exists():                            # bundles written before v4
        rm_ = [CatBoostClassifier().load_model(str(d / "regime.cbm"))]
    b["regime"] = {"model": rm_ or None, "raw_fields": rj.get("raw_fields", False),
                   "const": rj.get("const"), "op_table": {k: tuple(v) for k, v in rj["op_table"].items()},
                   "prior": rj["prior"], "cal": {k: tuple(v) for k, v in rj["cal"].items()}}
    for k in ("model_A", "model_A_neighbours"):
        parts = sorted(d.glob(f"{k}_[0-9]*.cbm"))
        b[k] = AvgModel([CatBoostRegressor().load_model(str(q)) for q in parts]) if parts else CatBoostRegressor().load_model(str(d / f"{k}.cbm"))
    names = set()
    for n in set(j["choices"]["specialist_airports"].values()):
        names |= set(MIXES.get(n, (n,)))
    def _spec(n):
        parts = sorted(d.glob(f"specialist_{n}_*.cbm"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
        if parts:
            return AvgModel([CatBoostRegressor().load_model(str(q)) for q in parts])
        return CatBoostRegressor().load_model(str(d / f"specialist_{n}.cbm"))
    b["specialists"] = {n: _spec(n) for n in names}
    b["lightgbm"] = lgb.Booster(model_file=str(d / "lightgbm.txt"))
    b["gates"] = {w: (CatBoostRegressor().load_model(str(d / f"gate_{w}.cbm")), json.loads((d / f"gate_{w}_spm.json").read_text(encoding="utf-8")))
                  for w in j["choices"]["gates"]}
    aj = json.loads((d / "anchor.json").read_text(encoding="utf-8"))
    A = Anchor(rich=aj.get("rich", False), fit_rows=aj.get("fit_rows", "robust"), clock_ridge=aj.get("clock_ridge", False))
    if A.clock_ridge:
        from .anchor import ClockRidge
        A.cr = ClockRidge(); A.cr.m = {k: np.array(v) for k, v in aj["clock_ridge_models"].items()}
    A.coef = {tuple(k.split("|")): np.array(v) for k, v in aj["coef"].items()}; A.median = aj["median"]; A.glob = aj["glob"]
    b["anchor"] = A
    b["lirf_regime"] = lirf_regime.load(d)
    b["experts"] = {ap_: CatBoostRegressor().load_model(str(d / f"expert_{ap_}.cbm")) for ap_ in j["choices"].get("expert_weights", {})}
    return b
