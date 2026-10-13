# SPDX-License-Identifier: GPL-3.0-only
"""Paths and settings. Paths come from configs/paths.json, overridable by environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]            # src/taxiout/config.py -> the repository root
KEY, LABEL = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIRPORT_TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
              "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
              "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}


def config() -> dict:
    """configs/paths.json (copy configs/paths.example.json and edit). Relative paths are relative to the repository root."""
    p = REPO / "configs" / "paths.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: copy configs/paths.example.json to configs/paths.json and set the input locations")
    return json.loads(p.read_text(encoding="utf-8"))


def _path(value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO / p


def paths() -> dict[str, Path]:
    cfg = config()
    opt = lambda key, env: _path(os.environ.get(env, cfg.get(key))) if (cfg.get(key) or os.environ.get(env)) else None
    return {"organiser": _path(os.environ.get("PRC_ORGANISER", cfg["organiser"])),
            "adsb_clocks": _path(os.environ.get("PRC_ADSB_CLOCKS", cfg["adsb_clocks"])),
            "metar": opt("metar", "PRC_METAR"),
            "open_meteo": opt("open_meteo", "PRC_OPEN_METEO"),
            "eurocontrol": opt("eurocontrol", "PRC_EUROCONTROL"),
            "atfm": opt("atfm", "PRC_ATFM"),
            "opdi": opt("opdi", "PRC_OPDI"),
            "extra": _path(cfg.get("extra", "data/extra")),
            "out": REPO / "results",                                                 # run outputs
            "features": _path(os.environ.get("PRC_FEATURES_DIR", "data/features")),  # the feature-table cache
            "reference": opt("reference", "PRC_REFERENCE"),
            # the January/July 2026 submitting.parquet (344,841 rows) while PRC_ORGANISER points at the final organiser folder
            "ranking_template_jan_jul": opt("ranking_template_jan_jul", "PRC_RANKING_TEMPLATE_JAN_JUL")}


@dataclass
class Settings:
    #: month-blocked folds over 2025: every row's out-of-fold prediction comes from a model that never saw its month
    #: January and July 2025 (the ranking set's calendar months) are the outer tests, crossed: each is predicted by a
    #: stack fitted and tuned on the other eleven months, which include the other held-out month. Inside, cross-validation
    #: uses contiguous month blocks (never day splits: neighbouring days share flights and airlines).
    #: The shipped run (src/v77/v77.yaml, --months 7) holds out July only: four blocks 1-3, 4-6, 8-10, 11-12.
    holdout_months: tuple = (1, 7)
    inner_blocks: int = 4
    trees: int = 4100                                 # model A (stages 4-5); Optuna on held-out months: 242.87 at 4,070 trees
                                                      # (vs 242.64 at 7,992 trees and twice the time)
    depth: int = 8
    learning_rate: float = 0.08                       # gates and specialists
    l2: float = 3.0                                   # gates and specialists
    a_learning_rate: float = 0.096                    # model A only (Optuna)
    a_l2: float = 1.0
    a_border_count: int = 128
    a_bagging_temperature: float = 0.092
    experts: bool = True                              # stage 6b: one model A per airport, mixed with the stack per airport
    expert_trees: int = 2000
    expert_min_rows: int = 5000                       # an airport needs this many fitting rows for an expert
    residual_clip: float = 7200.0
    lgb_trees: int = 1650                             # stage 6 second learner (Optuna: 256.84 -> 250.68, early stop at 1,637)
    spec_trees: int = 1500                            # stage 7 no-NM specialist
    spec_seeds: int = 5                               # v4.1: each no-NM specialist averaged over this many seeds
    spec_small_rows: int = 5000                       # v4.1: below this many fitting rows a specialist uses shallower,
    spec_small_depth: int = 4                         #       more regularised trees (depth 4, L2 10)
    spec_min_folds: int = 3                           # v4: a specialist must win in >= this many inner folds at the airport
    spec_max_top3: float = 0.9                        # v4: ... and its top 3 rows may carry at most this share of its gain (LIRF: 51%, LTFM: 172% in v3)
    neighbours: bool = True                           # stage 5
    nb_skip_airports: str = ""                        # v7: comma-separated airports whose runway-neighbour features are
                                                      #     left missing (LIRF: stamped neighbours pull ordinary taxis long)
    main_seeds: int = 1                               # v7: model A and model A + neighbours averaged over this many seeds
    drop: tuple = ()                                  # feature families left out (ablation), see model.FAMILIES
    family_airports: dict = field(default_factory=dict)  # v4: {family: airports}; the family's columns are missing elsewhere
    gate_min_evidence: int = 200                      # below this many rows at an airport the gate stays ON (see pipeline.gate_airports)
    gate_skip: tuple = ("LFPG",)                      # never gated: LFPG's ADS-B first points are inbound landings
    gate_genuine_only: bool = False                   # gates fitted without schedule-stamped rows (A/B)
    calibrate_regime: bool = False                    # isotonic calibration of P(stamped) (off: see model.regime_fit)
    gate_trees: int = 1500
    gate_depth: int = 7
    gate_clip: float = 1800.0
    lirf_regime: bool = True                         # stage 7b: LIRF no-NM >= 1 h (lirf_regime.py)
    lirf_regime_weight: float = 0.5                  # classifier prediction vs the stack's, below 8 h
    lirf_regime_weight_auto: bool = True             # v4: that weight chosen (0..1 by 0.1) on the stack's out-of-fold predictions
    lirf_regime_seeds: int = 3                       # v4: the stage's classifier averaged over this many seeds
    lirf_regime_weighted: bool = False               # v77: classifier rows weighted by (gap - ordinary)^2, what a wrong call costs
                                                     #      (July 2025 holdout 259.91 -> 258.27, January 326.24 -> 326.05)
    lirf_dayshift_ev: bool = False                   # v77: LIRF no-NM rows >= 14 h late whose take-off is on a later local date than
                                                     #      the schedule get the expected value over day-shift / stamped / ordinary,
                                                     #      shares by gap band from the fitting rows (July 259.91 -> 255.36, January
                                                     #      326.24 -> 322.78; both months' gains rest on a handful of day-shift rows)
    disruption: bool = False                         # 9 Oct: dis_ family (disruption.py) - airport-hour / airport-day share of
                                                     #      departures >= 1 h behind schedule, from the schedule only; residual screen
                                                     #      on 2025 OOF -0.92 against its control, the only new family with a signal
    rest_corrector: bool = False                     # corrector on NM rows no ADS-B gate reaches: cross-month it loses at every airport (28 Sep, out/rest_check.py)
    gate_split: bool = True                           # separate NM and no-NM gates (False: one gate over both)
    seed: int = 20261011
    device: str = "GPU"                               # every CatBoost fit; "CPU" (--cpu) for byte-reproducible runs
    regime_device: str = "CPU"                        # the stamp classifier: CPU gives repeatable probabilities (v4)
    regime_seeds: int = 3                             # stamp classifier averaged over this many seeds (v4)
    regime_raw_fields: bool = True                    # stamp classifier on the organiser's raw type / market / flight type / wake
    verbose: int = 250                                # CatBoost progress every N trees (0: silent)
    threads: int = 16
    sample_days: float = 1.0                          # < 1: smoke run on a share of the days
    notes: dict = field(default_factory=dict)
