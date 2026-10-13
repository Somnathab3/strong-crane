# Features: what each column is, and how and where it is made

The pinned feature table (`features_f391f7893a3cde99.parquet`, 338 columns, one row per movement) is built by
`taxiout.run.load_features` (notebook step 3 with `V77_FEATURES=build`, or `make features`). This page traces every
column back to the function that creates it and the input it is computed from, and every input back to the script
that produces it. Paths are `configs/paths.json` keys; file paths are relative to the repository.

## From raw data to the table

```
organiser files ─────────────────────────────────────────────┐
OPDI flight list ─┐                                          │
adsb.lol archive ─┼─ scripts/build_adsb_clocks.py ─ adsb_clocks/clock_<day>, rev_<day>, raw/<day>/ ─┐
X-Plane, OSM ─────┘   (scripts/download_xplane_stands.py, download_osm_stands.py)                   │
                       python -m taxiout.build_extra taxi|opdi|types|stops|stops2|hotspots|hotspots2|inbound ─ extra/
METAR ───────── scripts/download_metar.py ─ metar/                                                   │
Open-Meteo ──── scripts/download_open_meteo.py ─ open_meteo/                                         │
EUROCONTROL ─── ansperformance.eu (SOURCES.md) ─ eurocontrol/, atfm                                  │
                                                     taxiout.run.load_features ─ feature table ◄─────┘
```

Build order inside `taxiout.run.load_features`:

1. `data.load`: the organiser files. Times become epoch seconds, `airport` = ADEP for departures and ADES for arrivals,
   `label` = TAXITIME_SEC on 2025 departures only, `taxi_in` = TAXITIME_SEC on arrival rows. The departures'
   TAXITIME_SEC and BLOCK_TIME are dropped here, before any feature is made.
2. `adsb.load`: `clock_<day>.parquet`, joined with `rev_<day>.parquet`.
3. `fields.fill_fields`: aircraft type, wake, market and flight type where the organiser leaves them blank (from
   `extra/opdi_types`, `extra/trace_types`).
4. The table is cached by a hash of the code and the inputs; a cached table is read instead of rebuilt.
5. `weather.metar`, `weather.open_meteo`, `eurocontrol.daily` + `eurocontrol.features`, `eurocontrol.monthly`,
   `run.extra_inputs` (the `extra/` files).
6. `features.build`: identity, NM clocks, stamp evidence, agreement, seasonal mix, wake sequence, calendar, traffic,
   `queue_arrivals.build`, ADS-B, then the weather / EUROCONTROL / extra frames, then `context_extra.build`. Columns
   with at most one distinct value are dropped (`op_`, `td_` exempt).
7. After the table (in `run.main`, when `Settings.disruption` is on): `disruption.add` → `dis_`.

## Columns by family

Label use: no column reads the departure's own TAXITIME or BLOCK time. "Arrival taxi-in" marks columns that use the
taxi-in times of arrival rows, which the organiser files give in both years.

| Columns | n | Created in | Input (paths key / file) | Input made by | What it measures | Label use |
|---|---|---|---|---|---|---|
| `MVT_ID_mvt`, `set`, `month`, `t`, `label` | 5 | `features.build` (from `data.load`) | organiser | challenge files | key, 2025 / 2026, month, take-off time, **the target** (not a model input) | `label` is the target |
| `airport`, `runway`, `stand`, `stand_runway`, `actype`, `ap_actype`, `wake`, `operator`, `market`, `flight_type`, `rule`, `ades`, `*_raw`, `type_src`, `market_src` | 18 | `features.identity` (`fields.fill_fields`) | organiser; `extra/opdi_types`, `extra/trace_types` | `build_extra types` | the flight's airport, runway, stand, aircraft, operator, market; which source filled a blank field | none |
| `nm_*`, `has_nm` | 14 | `features.nm_clocks` | organiser (AOBT_3, LOBT, EOBT_1, IOBT) | challenge files | take-off minus each NM off-block clock, the best clock `nm_b` and its source, spread, missing count | none |
| `csp_` | 6 | `features.nm_clocks` | organiser | challenge files | pairwise differences between the NM clocks | none |
| `sched`, `sched_minus_b`, `sched_abs_log`, `sched_band` | 4 | `features.nm_clocks` | organiser (SCHED_TIME) | challenge files | take-off minus schedule, against `nm_b`, log, band | none |
| `flight_dur_plan` | 1 | `features.nm_clocks` | organiser (ARVT_1 − EOBT_1) | challenge files | planned block time | none |
| `svc_`, `sched_prev_day` | 6 | `features.stamp_evidence` | organiser (FLIGHT, SCHED_TIME) | challenge files | doubled or malformed flight number, service counts per month and day, scheduled on the previous day | none |
| `agr_` | 6 | `features.agreement` | organiser (movement vs flight-plan copies) | challenge files | whether the airport and the NM agree on type, destination, rule, flight number | none |
| `sea_` | 4 | `features.seasonal_mix` | organiser | challenge files | share of departures without an NM record per airport-day and month, departures per month | none |
| `wk_` | 6 | `features.wake_sequence` | organiser (wake, type, runway, take-off times) | challenge files | own and leading wake class, required separation, actual minus expected gap, wake load in 30 min | none |
| `hour_`, `dow`, `doy_`, `month_num` | 7 | `features.calendar` | organiser `t` | challenge files | local hour, weekday, day of year, month | none |
| `dep_`, `arr_before/after_*`, `rwy_` | 20 | `features.traffic` | organiser (all take-off and landing times) | challenge files | departures and arrivals within ±15/30/60 min, on the runway, gaps to the previous and next take-off | none |
| `q_`, `qc_` | 6 | `features.traffic` | organiser (AOBT_3, take-off times) | challenge files | departures off-block but not yet airborne (NM clock), their mean wait, queue per departure | none |
| `arr_done_*`, `arr_taxiin_mean_*` | 4 | `features.traffic` | organiser arrivals | challenge files | arrivals in-block in the last 30 / 60 min, their mean taxi-in | arrival taxi-in |
| `stand_turn_age` | 1 | `features.traffic` | organiser arrivals | challenge files | time since the last arrival at this stand | arrival taxi-in |
| `pm_` | 2 | `features.traffic` | organiser arrivals | challenge files | monthly median taxi-in per stand and per runway | arrival taxi-in (whole month) |
| `rb_` | 16 | `queue_arrivals._queue` | organiser (AOBT_3, take-off, runway) | challenge files | runway queue: members, their age, own rank, headways, take-offs within ±5/10/20 min | none |
| `ac_` | 16 | `queue_arrivals._arrivals` | organiser arrivals | challenge files | arrivals completed at the airport and runway, their taxi-in median / p90, last taxi-in, landings within ±10 min | arrival taxi-in |
| `adsb_`, `cp_`, `first_`, `gap_before_s`, `n_ground`, `hold_s`, `stand_seen_dist_m`, `assigned_dist_m` | 16 | `features.adsb` | `adsb_clocks/clock_<day>` | `scripts/build_adsb_clocks.py` (adsb.lol, OPDI link, X-Plane and OSM stands) | take-off minus each ADS-B event (first ground point, leaving the stand, last point at the stand, hold start); where the first point was heard; validity flags | none |
| `rev_` | 5 | `features.adsb` | `adsb_clocks/rev_<day>` | `scripts/build_adsb_clocks.py` (`adsb/pushback.py`) | the last run moving backwards before take-off (the pushback): start, end, steps, speed | none |
| `mt_` | 24 | `weather.metar` | metar | `scripts/download_metar.py` | last METAR within 90 min: temperature, dew point, wind, gusts, visibility, pressure, weather flags, report age; 1 h and 3 h changes | none |
| `om_` | 35 | `weather.open_meteo` | open_meteo | `scripts/download_open_meteo.py` | forecast for the take-off hour; de-icing, snow, fog and freezing-precipitation flags | none |
| `omh_` | 6 | `weather.open_meteo` | open_meteo | `scripts/download_open_meteo.py` | precipitation, snowfall and peak gust over the 3 h before and after | none |
| `ast_` | 45 | `eurocontrol.daily`, `eurocontrol.features` | eurocontrol (airport traffic, pre-departure delay, slot adherence), atfm | ansperformance.eu (SOURCES.md) | same-day arrival ATFM delay and causes, traffic, pre-departure delay, slot adherence; previous day, 3- and 7-day means; z-scores; ATFM delay at the destination | none (published airport aggregates) |
| `eum_` | 9 | `eurocontrol.monthly` | eurocontrol (taxi-out, taxi-in, ASMA additional time) | ansperformance.eu (SOURCES.md) | EUROCONTROL's monthly mean and additional taxi-out, taxi-in and ASMA time per airport | none from the organiser; EUROCONTROL's own monthly taxi-out average (dropped from the final model, see `v77.yaml` settings) |
| `td_` | 15 | `build_extra._dynamics` (`build_extra taxi`), joined in `run.extra_inputs` | `extra/taxi_<day>` | `python -m taxiout.build_extra taxi` (raw traces under `adsb_clocks/raw/`) | how the aircraft moved from its start to take-off: points, span, gaps, path, stopped and moving time, stops, turns, speeds | none |
| `op_` | 3 | `build_extra.build_opdi`, joined in `run.extra_inputs` | `extra/opdi_events` | `python -m taxiout.build_extra opdi` (OPDI flight events and list) | OPDI match, take-off minus the first taxiway entry, number of taxiway entries | none |
| `ib_` | 6 | `build_extra.build_inbound` | `extra/inbound` | `python -m taxiout.build_extra inbound` (OPDI list, clocks, organiser arrivals) | the aircraft's previous arrival: found, ground time, slack to schedule, arrival delay, taxi-in, same stand | arrival taxi-in (dropped from the final model) |
| `hs_` | 9 | `hotspots.build` (`build_extra hotspots`) | `extra/hotspots` (from `extra/stops`) | `build_extra stops`, then `build_extra hotspots` | waits at holding points, junctions and long stops on the own track; queue at the runway holding point; share of long waits in the last hour | none (the holding-point map uses 2025 tracks only) |
| `hs2_` | 13 | `hotspots.build2` (`build_extra hotspots2`) | `extra/hotspots2` (from `extra/stops2`) | `build_extra stops2`, then `build_extra hotspots2` | aircraft in the runway-entry area at take-off − 2/5/10 min, the last 3 entry waits, own hold, taxi and entry times | none |
| `cx_` | 10 | `context_extra.wind`, `.configuration`, `.plan_relative` (called from `features.build`) | `mt_` columns, organiser runway and `t`, `flight_dur_plan` | as above | runway head- and crosswind, low-visibility procedures, runway configuration changes, planned block time against the 2025 route median | none (dropped from the final model) |
| `dis_` (not in the cached table) | 4 | `disruption.add` | the table's `sched`, `airport`, `t`, `set` | – | share of departures ≥ 1 h behind schedule in the airport-hour and day, departures in the hour, own schedule-gap percentile | none |

Families switched off in the final model (`drop` in `src/v77/v77.yaml`): `cx_` (context), `ib_` (inbound), `eum_`
(eu_monthly), `sea_` (seasonal) and four Open-Meteo fields whose 2026 values moved for reasons other than weather
(`om_drift`). Hotspot families are used at EDDF, EHAM and LEBL only (`family_airports`).

## Inputs without a script in this repository

The organiser files (challenge website), the OPDI flight list and events, and the EUROCONTROL tables are downloaded as
files; [SOURCES.md](SOURCES.md) gives each URL pattern, licence and download date.
