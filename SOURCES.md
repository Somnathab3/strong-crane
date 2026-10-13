# Data sources

Every input the pipeline reads: what it is, where it was downloaded from, its licence, the period covered, and where it
goes in `configs/paths.json`. The challenge requires every additional dataset to be openly available under an open
licence.

## Inputs used by the pipeline

| input | `paths.json` key | source (URL) | licence | coverage | files |
|---|---|---|---|---|---|
| Challenge movements and ranking set | `organiser` | PRC Data Challenge 2026 — https://ansperformance.eu/study/data-challenge/dc2026/data.html | challenge terms | 2025-01-01 to 2025-12-31 (training), January and July 2026 (ranking), January, February, June and July 2026 (final phase, released 8 October 2026) | `training_2025-MM-01_….parquet` × 12, `ranking.parquet`, `submitting.parquet`; final phase `final_ranking.parquet`, `final_submitting.parquet` (used as `ranking.parquet` / `submitting.parquet` through `PRC_ORGANISER`) |
| ADS-B traces (adsb.lol) | `adsb_clocks`, `extra` | https://github.com/adsblol/globe_history_2025 and https://github.com/adsblol/globe_history_2026 — daily releases `v<YYYY.MM.DD>-planes-readsb-prod-0` (`trace_full` per aircraft, ~3 GB a day) (`scripts/build_adsb_clocks.py`: OPDI link, one pass per day over the archive, stand clock) | ODbL 1.0 | every day of 2025 and of January, February, June and July 2026 | processed into `clock_<YYYY-MM-DD>.parquet` (2025-01-01 to 2026-07-31; February and June 2026 built 8 October 2026; 11 June 2026 is incomplete in the adsb.lol archive itself) and the `data/extra` files (taxi dynamics, stops, hotspots, inbound aircraft); the raw traces are not kept |
| Stand positions (for the ADS-B stand clock) | (built into the clocks) | X-Plane Scenery Gateway, recommended airport scenery (`apt.dat` rows 1300/1301) — https://gateway.x-plane.com/airports/page (files through the API, https://gateway.x-plane.com/api, endpoint `/apiv1/scenery/<id>`); scenery IDs EDDF 103368, EDDM 111124, EGLL 111341, EHAM 111943, LEBL 102255, LEMD 99851, LFPG 111022, LIRF 110482, LSZH 106102, LTFM 105504 (fetched 24 Sep 2026; `scripts/download_xplane_stands.py`); then (`scripts/download_osm_stands.py`) OpenStreetMap (`aeroway=parking_position`, `gate`) — https://www.openstreetmap.org | Gateway terms (X-Plane); ODbL 1.0 (OSM) | static | used when the clocks were built (each stand's source is recorded as `xplane_norm` or `osm`) |
| OPDI flight list | `opdi` | EUROCONTROL Open Performance Data Initiative v002 — `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_<YYYYMM>.parquet` ; code: https://github.com/euctrl-pru/opdi | EUROCONTROL open data | 2025-01 to 2026-07 (monthly) | `flight_list/flight_list_202501.parquet` … `flight_list_202607.parquet` |
| OPDI flight events and measurements | `opdi` | `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_<start>_<end>.parquet`, same pattern for `measurements/` | EUROCONTROL open data | 2024-12-26 to 2026-08-08 (10-day files) | `flight_events/`, `measurements/` |
| METAR | `metar` | Iowa Environmental Mesonet ASOS archive — https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py (`scripts/download_metar.py`) | freely provided by IEM; source observations public | 2025-01-01 to 2026-08-01, the ten airports and every far-end airport of a movement | one parquet per station (1,441 stations) |
| Open-Meteo historical forecast | `open_meteo` | https://historical-forecast-api.open-meteo.com/v1/forecast (`scripts/download_open_meteo.py`) | CC-BY 4.0 (Open-Meteo) | 2025-01-01 to 2026-08-31, hourly, the ten airports | `open_meteo_hourly.parquet` |
| EUROCONTROL airport statistics (incl. the daily pre-departure delay table, used by stage 12 to reconcile each day's predictions) | `eurocontrol` | EUROCONTROL Aviation Intelligence bulk CSV — https://ansperformance.eu/csv/ (and https://ansperformance.eu/data/) | EUROCONTROL open data | 2025 and 2026 (monthly tables published to June 2026) | `airport_traffic`, `all_pre_departure_delays`, `atc_pre_departure_delays`, `atfm_slot_adherence`, `taxi_out_additional_time`, `taxi_in_additional_time`, `asma_additional_time` (`_2025.csv`, `_2026.csv`, combined `.parquet`) |
| Airport arrival ATFM delay | `atfm` | EUROCONTROL — https://ansperformance.eu/data/ (Airport Arrival ATFM Delay) | EUROCONTROL open data | 2025–2026, per airport and day | `airport_arrival_atfm_delay.parquet` (from `Airport_Arrival_ATFM_Delay.xlsx`) |

Download dates (files on disk): challenge files 1–5 September 2026; OPDI 4 September; METAR and the ATFM delay table
13 September; Open-Meteo 14 September; EUROCONTROL airport statistics 13–22 September; ADS-B clocks built 24–26
September. Everything was archived as downloaded: the IEM and EUROCONTROL archives are revised over time (a METAR
station's position changed within two weeks), so a fresh download can differ slightly.

`data/osm_layout/<ICAO>.json` (shipped): runways, taxiways, aprons, buildings and stands of the ten airports, extracted from OpenStreetMap on 12 August 2026. © OpenStreetMap contributors, available under the Open Database License (ODbL 1.0), https://www.openstreetmap.org/copyright.

## Used only in submitted files, not in this pipeline

| source | licence | what it added |
|---|---|---|
| ADS-B Exchange sample data (operations + `trace_full`, first day of each month) — https://www.adsbexchange.com/products/historical-data/ | free for testing and evaluation; **not open-licensed** | a correction layer on the 1 January and 1 July 2026 ranking rows of v108 to v117 (7,440 rows). It is not part of `src/` and the notebook does not read it. Those files are members of the score-weighted ensembles v118, v122, v127, v129 and v132, and v132 supplies the January and July rows of the final submission, so the final file depends on it through the ensemble weights |

## Used for checks and figures only (not model inputs)

| source | licence | use |
|---|---|---|
| EUROCONTROL daily punctuality and daily traffic variation dashboards — https://www.eurocontrol.int/Economics/DailyPunctuality-Airports.html (`pun_apt_data.json`, `Airports_Punctuality.xlsx`, `tfc_ao_data.json`, `Aircraft_Operators.xlsx`) | EUROCONTROL open data | daily airport punctuality and departure delay (off-block based) to check the 2026 predictions; screened as features (no gain beyond the disruption features); downloaded 10 October 2026 |
| one full adsb.lol day of raw points (2 April 2025, `data_ext/adsb_core/2025-04-02.parquet`) | ODbL 1.0 | the example-track figure of the paper (SWR758, Zurich) |

## Checked and not used

| source | why not |
|---|---|
| adsb.lol `mlatonly-0` and `staging-0` releases (ODbL) | no ground points at the ten airports / a mirror of `prod-0` |
| adsb.lol aircraft datalink archive, ACARS / VDL2 (CC0) | receivers mostly in the US; no off-block (OUT) messages for European airports |
| ADSBiq (ODbL) | starts 28 June 2026; no 2025; almost no ground data |
| OpenSky REST API / Trino | historical access needs approval; academic-only terms |
| adsb.fi, airplanes.live, ADSB One, ADSBHub, TheAirTraffic, Plane.watch, sdrmap | live feeds only, no historical archive (or non-commercial terms) |
| airframes.io (ACARS OOOI) | not open; history by authorisation |
| Schiphol Flight API | a few days of history only |
| FlightAware, Flightradar24, RadarBox, PlaneFinder | commercial; no open 2025 history |
| IEM TAF archive | US stations only |
| European NOTAM archives | commercial only |
| A-CDM TOBT / TSAT / AOBT records | not published |
