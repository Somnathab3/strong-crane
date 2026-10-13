# SPDX-License-Identifier: GPL-3.0-only
"""Stand positions of the ten airports from the X-Plane Scenery Gateway (input of scripts/build_adsb_clocks.py).

    python scripts/download_xplane_stands.py [--out data/xplane] [--recommended]

Downloads each airport's scenery pack through the Gateway API (https://gateway.x-plane.com/api), keeps its apt.dat as
<out>/<ICAO>_apt.dat and writes <out>/stands.parquet from the 1300 (startup location: lat, lon, heading, type, name)
and 1301 (ICAO width, operation type, airlines) rows, plus manifest.json. By default the scenery packs our clock files
were built with are fetched (fetched 24 September 2026); --recommended takes each airport's current recommended pack.
apt.dat layout: https://developer.x-plane.com/article/airport-data-apt-dat-12-00-file-format-specification/
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

API = "https://gateway.x-plane.com/apiv1"
SCENERY = {"EDDF": 103368, "EDDM": 111124, "EGLL": 111341, "EHAM": 111943, "LEBL": 102255,
           "LEMD": 99851, "LFPG": 111022, "LIRF": 110482, "LSZH": 106102, "LTFM": 105504}


def fetch(icao: str, sid: int, out: Path) -> dict:
    scenery = json.load(urllib.request.urlopen(f"{API}/scenery/{sid}", timeout=120))["scenery"]
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(scenery["masterZipBlob"])))
    name = [n for n in z.namelist() if n.lower().endswith(".dat")][0]; data = z.read(name)
    (out / f"{icao}_apt.dat").write_bytes(data)
    return {"icao": icao, "sceneryId": sid, "uploaded": scenery.get("dateUploaded"), "artist": scenery.get("userName"), "file": name, "bytes": len(data)}


def stands_of(icao: str, out: Path) -> list[dict]:
    rows, last = [], None
    for line in open(out / f"{icao}_apt.dat", encoding="utf-8", errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "1300" and len(p) >= 7:
            last = {"airport": icao, "lat": float(p[1]), "lon": float(p[2]), "heading": float(p[3]), "type": p[4], "ac_types": p[5], "name": " ".join(p[6:])}
            rows.append(last)
        elif p[0] == "1301" and last is not None:
            last["icao_width"] = p[1] if len(p) > 1 else None; last["op_type"] = p[2] if len(p) > 2 else None
            last["airlines"] = " ".join(p[3:]) if len(p) > 3 else ""
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "xplane")
    ap.add_argument("--recommended", action="store_true", help="each airport's current recommended scenery instead of the pinned packs")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); manifest, stands = [], []
    for icao, sid in SCENERY.items():
        if a.recommended:
            sid = json.load(urllib.request.urlopen(f"{API}/airport/{icao}", timeout=60))["airport"]["recommendedSceneryId"]
        manifest.append(fetch(icao, sid, a.out)); print(manifest[-1], flush=True); time.sleep(1)
        stands += stands_of(icao, a.out)
    pd.DataFrame(stands).to_parquet(a.out / "stands.parquet", index=False)
    (a.out / "manifest.json").write_text(json.dumps({"source": "X-Plane Scenery Gateway", "api": API, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                     "airports": manifest}, indent=2), encoding="utf-8")
    print(f"{len(stands):,} stands -> {a.out / 'stands.parquet'}")


if __name__ == "__main__":
    main()
