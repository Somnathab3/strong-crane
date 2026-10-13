# SPDX-License-Identifier: GPL-3.0-only
"""Build the daily ADS-B clock files (paths adsb_clocks) from the public adsb.lol archive.

    python scripts/build_adsb_clocks.py --days 2025-04-02 2025-04-03
    python scripts/build_adsb_clocks.py --from 2025-01-01 --to 2025-12-31 [--delete-raw]
    PRC_ORGANISER=<final organiser folder> python scripts/build_adsb_clocks.py --from 2026-01-01 --to 2026-07-31

Per UTC day: the departures at the ten airports whose take-off is on that day (2025: the training file of the month;
2026: ranking.parquet of the organiser folder), linked to their aircraft through the OPDI flight list (paths opdi),
one pass over the adsb.lol day archive keeping only those aircraft (<adsb_clocks>/raw/<day>/trace_full_<icao24>.json.gz,
the input of `python -m taxiout.build_extra`), then the stand clock of each departure -> <adsb_clocks>/clock_<day>.parquet.
Stands: X-Plane positions (scripts/download_xplane_stands.py); an assigned stand the X-Plane names do not match exactly is
placed with normalised names, then OpenStreetMap (scripts/download_osm_stands.py, data/xplane/osm_stands.parquet), and
the row's `stand_src` says which. Then the pushback signature (adsb/pushback.py) -> <adsb_clocks>/rev_<day>.parquet. Days already built are skipped, so the job can
be stopped and restarted. The archive is ~3-4 GB a day; the kept traces ~300 MB a day (--delete-raw removes them once
the clock file is written, but build_extra needs them).

Data: adsb.lol globe_history_<year> (ODbL 1.0, feeder contributions CC0); OPDI flight list (EUROCONTROL open data);
X-Plane Scenery Gateway apt.dat stand positions.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "src"))
from adsb.fetch import fetch_day                                     # noqa: E402
from adsb.link import AIRPORTS, RAW_COLUMNS, attach_icao24           # noqa: E402
from adsb.pushback import pushbacks                                 # noqa: E402
from adsb.stand_clock import airport_centres, clock_for, load_stands, lookup, norm_stand, resolve  # noqa: E402
from taxiout.config import paths                                     # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def departures_on(day: str, organiser: Path) -> pd.DataFrame:
    ts = pd.Timestamp(day)
    src = organiser / "ranking.parquet" if ts.year == 2026 else sorted(organiser.glob(f"training_{ts:%Y-%m}-01_*.parquet"))[0]
    f = pd.read_parquet(src, columns=["PHASE_mvt"] + RAW_COLUMNS)
    f = f[f.PHASE_mvt.eq("DEP")].drop(columns=["PHASE_mvt"])
    f = f[pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").eq(day) & f.ADEP_mvt.isin(AIRPORTS)].copy()
    f["has_nm"] = f.IOBT_flt.notna()
    f["FLIGHT_mvt"] = f["FLIGHT_mvt"].fillna("").astype(object)      # the callsign parser expects str
    return f.reset_index(drop=True)


def build_day(day: str, P: dict, stands: pd.DataFrame, centres: dict, xp: dict, osm: dict, delete_raw: bool = False) -> None:
    out_dir = Path(P["adsb_clocks"]); target = out_dir / f"clock_{day}.parquet"; raw_root = out_dir / "raw"
    rev_target = out_dir / f"rev_{day}.parquet"
    if target.exists():
        if not rev_target.exists() and (raw_root / day).exists():
            pushbacks(pd.read_parquet(target, columns=["MVT_ID_mvt", "icao24", "mvt"]), raw_root / day).to_parquet(rev_target, index=False)
            log(f"{day}: clock already built; pushback signature added")
        else:
            log(f"{day}: already built")
        return
    man = attach_icao24(departures_on(day, Path(P["organiser"])), day, Path(P["opdi"]))
    wanted = sorted(set(man.icao24.dropna().astype(str)))
    log(f"{day}: departures {len(man):,}, aircraft linked {man.icao24.notna().mean():.1%}")
    try:
        fetch_day(day, wanted, raw_root, pod="planes-readsb-prod-0")
    except urllib.error.HTTPError as e:                               # 28 May - 10 Jun 2025 exist only as "...-prod-0tmp" releases
        if e.code != 404:
            raise
        log(f"{day}: prod-0 release missing, using prod-0tmp"); fetch_day(day, wanted, raw_root, pod="planes-readsb-prod-0tmp")
    raw = raw_root / day; rows = []; known = set(zip(stands.airport, stands.key))
    for _, dep in man.iterrows():
        ap = dep.ADEP_mvt; icao = dep.get("icao24"); st = stands[stands.airport == ap].reset_index(drop=True); src = None
        path = raw / f"trace_full_{str(icao).lower()}.json.gz" if isinstance(icao, str) and icao else Path("__none__")
        stand = str(dep.STAND_mvt) if pd.notna(dep.STAND_mvt) else ""
        if stand.strip() and (ap, norm_stand(stand)) not in known and path.exists():
            la, lo, how = resolve(ap, stand, xp, osm)                   # a stand the X-Plane names miss: placed, then the clock
            if la == la:
                st = pd.concat([st, pd.DataFrame([{"airport": ap, "key": norm_stand(stand), "lat": la, "lon": lo, "name": stand}])], ignore_index=True); src = how
        r = clock_for(dep, path, st, centres[ap])
        r.update({"stand_src": src})
        r.update({"date": day, "airport": ap, "has_nm": bool(dep.has_nm), "icao24": icao, "mvt": pd.Timestamp(dep.MVT_TIME_UTC_mvt).timestamp(),
                  "aobt3": pd.Timestamp(dep.AOBT_3_flt).timestamp() if pd.notna(dep.AOBT_3_flt) else float("nan"), "stand_assigned": str(dep.STAND_mvt)})
        rows.append(r)
    res = pd.DataFrame(rows); out_dir.mkdir(parents=True, exist_ok=True); res.to_parquet(target, index=False)
    ok = res.clock_b_ok.astype("boolean").fillna(False).astype(bool) if "clock_b_ok" in res else pd.Series(False, index=res.index)
    log(f"{day}: pushback signature on {int(ok.sum()):,} rows ({ok.mean():.1%}), stand clock on {int(res.status.eq('clock').sum()):,}; saved {target.name}")
    rev = pushbacks(res[["MVT_ID_mvt", "icao24", "mvt"]], raw); rev.to_parquet(rev_target, index=False)
    log(f"{day}: backward-moving pushback run on {len(rev):,} departures; saved {rev_target.name}")
    if delete_raw:
        shutil.rmtree(raw, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=[]); ap.add_argument("--from", dest="start"); ap.add_argument("--to", dest="end")
    ap.add_argument("--xplane-dir", type=Path, default=REPO / "data" / "xplane", help="stands.parquet from scripts/download_xplane_stands.py")
    ap.add_argument("--delete-raw", action="store_true", help="delete each day's traces after its clock is saved")
    a = ap.parse_args()
    days = list(a.days) + ([d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end)] if a.start and a.end else [])
    if not days:
        ap.error("give --days or --from/--to")
    P = paths(); stands = load_stands(a.xplane_dir); centres = airport_centres(stands)
    xp = lookup(stands[["airport", "key", "lat", "lon"]]); f = a.xplane_dir / "osm_stands.parquet"
    osm = lookup(pd.read_parquet(f)) if f.exists() else {}
    if not osm:
        log(f"no {f}: stands missing from the X-Plane names are placed by normalised names only")
    for day in days:
        try:
            build_day(day, P, stands, centres, xp, osm, delete_raw=a.delete_raw)
        except Exception as e:                                        # keep going; a failed day is rebuilt on the next run
            log(f"{day}: FAILED {type(e).__name__}: {str(e)[:200]}")
    log("done")


if __name__ == "__main__":
    main()
