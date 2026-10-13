# SPDX-License-Identifier: GPL-3.0-only
"""Stand and gate positions from OpenStreetMap, the fallback for stands the X-Plane files do not name.

    python scripts/download_osm_stands.py [--out data/xplane]                      # live Overpass query
    python scripts/download_osm_stands.py --from-aeroways <osm folder>             # from saved aeroway tables

Every aeroway=parking_position or aeroway=gate with a `ref` within 6 km of the airport (centre: median of its X-Plane
stands) becomes one row per ref (refs like "A1;A2" are split): airport, key (normalised like the X-Plane names), lat,
lon (the node, or the centre of a way) -> <out>/osm_stands.parquet. OpenStreetMap data (c) OpenStreetMap contributors,
ODbL 1.0. The clock files were built from a snapshot taken on 12 August 2026 (<osm folder>/<ICAO>/aeroways_raw.parquet,
WKB geometries); a live query today can differ where OSM was edited since.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from adsb.stand_clock import norm_stand  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")


def wkb_centre(b: bytes) -> tuple[float, float]:
    """(lat, lon) of a WKB Point / LineString / Polygon (mean of the outer ring)."""
    e = "<" if b[0] == 1 else ">"; t = struct.unpack(e + "I", b[1:5])[0] & 0xFF
    if t == 1:
        x, y = struct.unpack(e + "dd", b[5:21]); return y, x
    if t == 2:
        n = struct.unpack(e + "I", b[5:9])[0]; xy = np.frombuffer(b[9:9 + 16 * n], dtype=e + "f8").reshape(-1, 2)
    elif t == 3:
        n = struct.unpack(e + "I", b[9:13])[0]; xy = np.frombuffer(b[13:13 + 16 * n], dtype=e + "f8").reshape(-1, 2)
    else:
        return np.nan, np.nan
    return float(xy[:, 1].mean()), float(xy[:, 0].mean())


def rows_of(airport: str, ref, lat: float, lon: float) -> list[dict]:
    return [{"airport": airport, "key": norm_stand(part), "lat": lat, "lon": lon} for part in str(ref).split(";")]


def from_aeroways(folder: Path) -> pd.DataFrame:
    rows = []
    for ap_dir in sorted(Path(folder).iterdir()):
        f = ap_dir / "aeroways_raw.parquet"
        if ap_dir.name not in AIRPORTS or not f.exists():
            continue
        t = pd.read_parquet(f, columns=["geometry", "aeroway", "ref"]); t = t[t.aeroway.isin(["parking_position", "gate"]) & t.ref.notna()]
        for g, ref in zip(t.geometry, t.ref):
            rows += rows_of(ap_dir.name, ref, *wkb_centre(bytes(g)))
    return pd.DataFrame(rows).dropna()


def from_overpass(centres: dict[str, tuple[float, float]], radius_m: int = 6000) -> pd.DataFrame:
    rows = []
    for ap in AIRPORTS:
        lat, lon = centres[ap]
        q = (f'[out:json][timeout:180];(node["aeroway"~"^(parking_position|gate)$"]["ref"](around:{radius_m},{lat},{lon});'
             f'way["aeroway"~"^(parking_position|gate)$"]["ref"](around:{radius_m},{lat},{lon}););out center tags;')
        req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "strong-crane (research)"})
        els = json.load(urllib.request.urlopen(req, timeout=240)).get("elements", [])
        for el in els:
            c = el if el.get("type") == "node" else el.get("center") or {}
            if "lat" in c:
                rows += rows_of(ap, el["tags"]["ref"], float(c["lat"]), float(c["lon"]))
        print(f"{ap}: {len(els)} OSM stands/gates", flush=True); time.sleep(2)
    return pd.DataFrame(rows).dropna()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "xplane")
    ap.add_argument("--from-aeroways", type=Path, default=None, help="folder with <ICAO>/aeroways_raw.parquet (saved OSM aeroway tables)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.from_aeroways:
        t = from_aeroways(a.from_aeroways)
    else:
        s = pd.read_parquet(a.out / "stands.parquet"); centres = {k: (g.lat.median(), g.lon.median()) for k, g in s.groupby("airport")}
        t = from_overpass(centres)
    t.to_parquet(a.out / "osm_stands.parquet", index=False); print(f"{len(t):,} OSM stand refs -> {a.out / 'osm_stands.parquet'}")


if __name__ == "__main__":
    main()
