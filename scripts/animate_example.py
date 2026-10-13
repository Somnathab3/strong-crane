# SPDX-License-Identifier: GPL-3.0-only
"""Animated example: every aircraft on the airfield from ADS-B, around one departure (the paper's SWR758 by default).

    python scripts/animate_example.py                               # interactive window
    python scripts/animate_example.py --export example.mp4          # 1920 x 1080, 30 fps video (+ a few PNG stills)
    python scripts/animate_example.py --mvt <MVT_ID_mvt>            # another departure of the training months
    python scripts/animate_example.py --supersample 1               # less memory (slightly softer edges)

Keys: space pause - left / right seek 10 s - up / down speed - r restart - s save a still - esc quit.
Needs pygame-ce, imageio and imageio-ffmpeg (pip install pygame-ce imageio imageio-ffmpeg), the organiser files, and
the day's ADS-B clock and raw traces from scripts/build_adsb_clocks.py (adsb_clocks/clock_<day>.parquet,
adsb_clocks/raw/<day>/). The traces are those of every aircraft that departs from one of the ten airports that day, so
the animation shows those aircraft (including their earlier arrival and ground moves), not aircraft that only arrive.
Airport layout: data/osm_layout/<ICAO>.json, OpenStreetMap runways, taxiways, aprons, buildings and stands of the ten
airports from a snapshot taken on 12 August 2026 (the map of the paper's figure); for another airport the Overpass API
(cached under results/animation/), or --osm-snapshot <folder> with <ICAO>/aeroways_raw.parquet. Data: adsb.lol (ODbL 1.0), OpenStreetMap contributors (ODbL 1.0), the challenge files.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import struct
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
from taxiout.config import paths  # noqa: E402

OUT = REPO / "results" / "animation"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
TYPES = {"BCS3": "A220-300", "BCS1": "A220-100", "A20N": "A320neo", "A21N": "A321neo", "A319": "A319", "A320": "A320", "A321": "A321",
         "A333": "A330-300", "A343": "A340-300", "A359": "A350-900", "B77W": "777-300ER", "B738": "737-800", "B38M": "737 MAX 8", "E190": "E190", "E295": "E195-E2"}

# palette (white page, quiet layout, one accent)
BG = (255, 255, 255); APRON = (243, 243, 240); BUILDING = (228, 227, 222); RUNWAY = (203, 203, 197); RW_MARK = (255, 255, 255)
TAXIWAY = (226, 226, 221); STAND = (214, 214, 208); INK = (24, 24, 22); INK2 = (92, 91, 87); INK3 = (150, 149, 144)
GROUND = (78, 77, 73); AIR = (160, 159, 154); TRAIL = (205, 204, 199); ACCENT = (235, 104, 52)
C_BLOCK, C_AOBT, C_MOVE, C_LINEUP, C_TKO = (214, 39, 40), (42, 120, 214), (235, 104, 52), (123, 50, 148), (27, 27, 27)
NAMES = {"EDDF": "Frankfurt Airport", "EDDM": "Munich Airport", "EGLL": "London Heathrow", "EHAM": "Amsterdam Schiphol", "LEBL": "Barcelona Airport",
         "LEMD": "Madrid-Barajas", "LFPG": "Paris-Charles de Gaulle", "LIRF": "Rome Fiumicino", "LSZH": "Zurich Airport", "LTFM": "Istanbul Airport"}
LAYOUT_KINDS = ("runway", "taxiway", "apron", "terminal", "hangar", "parking_position")


# ---------------------------------------------------------------------------------------------------------- data
def wkb(b: bytes):
    """WKB -> list of (kind, [(lon, lat), ...]) parts; kind 'pt' | 'ln' | 'pg' (outer ring)."""
    out = []

    def read(o):
        e = "<" if b[o] == 1 else ">"; t = struct.unpack(e + "I", b[o + 1:o + 5])[0] & 0xFF; o += 5
        if t == 1:
            x, y = struct.unpack(e + "dd", b[o:o + 16]); out.append(("pt", [(x, y)])); return o + 16
        if t == 2:
            n = struct.unpack(e + "I", b[o:o + 4])[0]; xy = np.frombuffer(b[o + 4:o + 4 + 16 * n], dtype=e + "f8").reshape(-1, 2)
            out.append(("ln", [tuple(p) for p in xy])); return o + 4 + 16 * n
        if t == 3:
            nr = struct.unpack(e + "I", b[o:o + 4])[0]; o += 4
            for k in range(nr):
                n = struct.unpack(e + "I", b[o:o + 4])[0]; xy = np.frombuffer(b[o + 4:o + 4 + 16 * n], dtype=e + "f8").reshape(-1, 2)
                if k == 0:
                    out.append(("pg", [tuple(p) for p in xy]))
                o += 4 + 16 * n
            return o
        if t in (4, 5, 6, 7):
            n = struct.unpack(e + "I", b[o:o + 4])[0]; o += 4
            for _ in range(n):
                o = read(o)
            return o
        return len(b)
    read(0)
    return out


def layout(airport: str, centre: tuple[float, float], snapshot: Path | None) -> list:
    """(kind, 'pt' | 'ln' | 'pg', points, ref) for runways, taxiways, aprons, buildings and stands around the airport."""
    if snapshot is not None:
        t = pd.read_parquet(Path(snapshot) / airport / "aeroways_raw.parquet", columns=["aeroway", "geometry", "ref"])
        t = t[t.aeroway.isin(LAYOUT_KINDS)]
        return [(kind, k, np.array(pts, dtype=np.float64), str(ref) if ref is not None else "")
                for kind, g, ref in zip(t.aeroway, t.geometry, t.ref) for k, pts in wkb(bytes(g))]
    shipped = REPO / "data" / "osm_layout" / f"{airport}.json"         # extracted from the 12 August 2026 snapshot (the paper's map)
    cache = shipped if shipped.exists() else OUT / f"osm_layout_{airport}.json"
    if not cache.exists():
        lat, lon = centre[1], centre[0]
        q = (f'[out:json][timeout:180];(way["aeroway"~"^(runway|taxiway|apron|terminal|hangar|parking_position)$"](around:6000,{lat},{lon});'
             f'node["aeroway"="parking_position"](around:6000,{lat},{lon}););out geom;')
        els, err = None, None
        for attempt in range(6):                                   # the public servers are often busy: retry, then the mirrors
            url = OVERPASS[attempt % len(OVERPASS)]
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "strong-crane (research)"})
                els = json.load(urllib.request.urlopen(req, timeout=240)).get("elements", []); break
            except Exception as e:                                 # noqa: BLE001
                err = e; print(f"Overpass {url}: {type(e).__name__} {e}; retrying", flush=True); time.sleep(5 * (attempt + 1))
        if els is None:
            raise SystemExit(f"no answer from the Overpass servers ({err}); try later or pass --osm-snapshot")
        parts = []
        for el in els:
            tags = el.get("tags", {}); kind = tags.get("aeroway")
            if el["type"] == "node":
                parts.append([kind, "pt", [[el["lon"], el["lat"]]], tags.get("ref", "")])
            elif "geometry" in el:
                pts = [[p["lon"], p["lat"]] for p in el["geometry"]]
                area = kind in ("apron", "terminal", "hangar") or tags.get("area") == "yes"
                parts.append([kind, "pg" if area and len(pts) >= 4 else "ln", pts, tags.get("ref", "")])
        OUT.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(parts), encoding="utf-8")
    return [(kind, k, np.array(pts, dtype=np.float64), ref) for kind, k, pts, ref in json.loads(cache.read_text(encoding="utf-8"))]


def read_trace(path: Path) -> np.ndarray:
    """t, lon, lat, track, ground speed, on ground, altitude for one trace_full_<icao24>.json.gz (readsb rows)."""
    d = json.load(gzip.open(path, "rt", encoding="utf-8")); base = float(d.get("timestamp", 0.0)); rows = []
    for e in d.get("trace", []) or []:
        if not isinstance(e, list) or len(e) < 6 or e[1] is None or e[2] is None:
            continue
        num = lambda v: float(v) if isinstance(v, (int, float)) else np.nan
        ground = isinstance(e[3], str) and e[3] == "ground"
        rows.append((base + float(e[0]), float(e[2]), float(e[1]), num(e[5]), num(e[4]), float(ground), -1.0 if ground else num(e[3])))
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


def prepare(mvt: int, snapshot: Path | None) -> dict:
    P = paths(); org = Path(P["organiser"]); clocks = Path(P["adsb_clocks"])
    cols = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt", "ADEP_mvt"]
    r = None
    for f in sorted(org.glob("training_2025-*.parquet")):
        x = pd.read_parquet(f, columns=cols, filters=[("MVT_ID_mvt", "==", float(mvt))])
        if len(x):
            r = x.iloc[0]; break
    if r is None:
        raise SystemExit(f"departure {mvt} not found in the training files of {org}")
    e = lambda v: pd.Timestamp(v).timestamp() if pd.notna(v) else np.nan
    tko, blk, a3 = e(r.MVT_TIME_UTC_mvt), e(r.BLOCK_TIME_UTC_mvt), e(r.AOBT_3_flt); day = pd.Timestamp(r.MVT_TIME_UTC_mvt).strftime("%Y-%m-%d")
    cache = OUT / f"{r.ADEP_mvt}_{day}_{mvt}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); return {k: z[k].item() for k in z.files}
    clock = pd.read_parquet(clocks / f"clock_{day}.parquet", columns=["MVT_ID_mvt", "icao24"]); icao = clock.loc[clock.MVT_ID_mvt == mvt, "icao24"]
    if icao.isna().all():
        raise SystemExit(f"departure {mvt}: no aircraft address in clock_{day}.parquet (run scripts/build_adsb_clocks.py --days {day})")
    icao = str(icao.iloc[0]).lower(); raw = clocks / "raw" / day; t0, t1 = blk - 300, tko + 150
    me = read_trace(raw / f"trace_full_{icao}.json.gz"); g = me[(me[:, 5] == 1) & (me[:, 0] >= t0) & (me[:, 0] <= tko)]
    centre = (float(np.median(g[:, 1])), float(np.median(g[:, 2])))
    parts = layout(str(r.ADEP_mvt), centre, snapshot)
    xp = REPO / "data" / "xplane" / "stands.parquet"                 # X-Plane stands (scripts/download_xplane_stands.py) where OSM maps few
    if sum(1 for p_ in parts if p_[0] == "parking_position") < 50 and xp.exists():
        s_ = pd.read_parquet(xp); s_ = s_[s_.airport == str(r.ADEP_mvt)]
        parts += [("parking_position", "pt", np.array([[lo_, la_]]), str(n_)) for lo_, la_, n_ in zip(s_.lon, s_.lat, s_.name)]
    rw = np.vstack([pts for kind, k, pts, _ in parts if kind == "runway"]) if any(p[0] == "runway" for p in parts) else g[:, 1:3]
    lon0, lat0 = rw.min(axis=0); lon1, lat1 = rw.max(axis=0); padx, pady = (lon1 - lon0) * 0.12, (lat1 - lat0) * 0.10
    tracks = {}
    for f in glob.glob(str(raw / "trace_full_*.json.gz")):
        tr = read_trace(Path(f))
        if not len(tr):
            continue
        keep = (tr[:, 0] >= t0 - 120) & (tr[:, 0] <= t1 + 60) & (tr[:, 1] > lon0 - padx) & (tr[:, 1] < lon1 + padx) & (tr[:, 2] > lat0 - pady) & (tr[:, 2] < lat1 + pady)
        keep &= (tr[:, 5] == 1) | (np.nan_to_num(tr[:, 6], nan=99999) < 3500)
        if keep.sum() >= 3:
            tracks[Path(f).name[len("trace_full_"):-len(".json.gz")]] = tr[keep]
    p = tracks[icao]; pt, pgs = p[:, 0], np.nan_to_num(p[:, 4])
    lineup = pt[(pt < tko) & (pgs < 30)][-1]; first_move = pt[(pgs >= 2) & (pt >= blk - 600)][0]
    meta = {"t0": t0, "t1": t1, "blk": blk, "aobt3": a3, "tko": tko, "lineup": float(lineup), "move": float(first_move), "icao": icao,
            "label": float(r.TAXITIME_SEC_mvt), "type": TYPES.get(str(r.AIRCRAFT_TYPE_mvt), str(r.AIRCRAFT_TYPE_mvt)), "stand": str(r.STAND_mvt),
            "runway": str(r.RUNWAY_mvt), "flight": str(r.FLIGHT_mvt), "airport": str(r.ADEP_mvt), "day": day}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tracks=np.array(tracks, dtype=object), parts=np.array(parts, dtype=object), meta=np.array(meta, dtype=object))
    return {"tracks": tracks, "parts": parts, "meta": meta}


# ---------------------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, data: dict, W: int, H: int, ss: int = 2):
        import pygame
        self.pg = pygame; self.W, self.H, self.ss = W, H, ss
        self.tracks, self.parts, self.m = data["tracks"], data["parts"], data["meta"]
        g = np.vstack([tr[(tr[:, 5] == 1) & (tr[:, 0] >= self.m["t0"]) & (tr[:, 0] <= self.m["t1"])][:, 1:3] for tr in self.tracks.values()])
        lo, hi = np.quantile(g, 0.005, axis=0), np.quantile(g, 0.995, axis=0)
        mx, my = (hi[0] - lo[0]) * 0.10, (hi[1] - lo[1]) * 0.12
        lon0, lon1, lat0, lat1 = lo[0] - mx, hi[0] + mx, lo[1] - my, hi[1] + my; self.k = math.cos(math.radians((lat0 + lat1) / 2))
        top, bottom, side = 205, 150, 60                                        # room for the header and the timeline
        sx = (W - 2 * side) / ((lon1 - lon0) * self.k); sy = (H - top - bottom) / (lat1 - lat0); self.s = min(sx, sy)
        self.cx = side + ((W - 2 * side) - (lon1 - lon0) * self.k * self.s) / 2; self.cy = top + ((H - top - bottom) - (lat1 - lat0) * self.s) / 2
        self.lon0, self.lat1 = lon0, lat1
        self.clip = pygame.Rect(0, (top - 22) * ss, W * ss, (H - top - bottom + 44) * ss)   # the map area between the header and the timeline
        self.m_per_px = 111_320 / self.s
        self.base = self._base()
        f = lambda size, bold=False: self._font(size * ss, bold)
        self.f_title, self.f_sub, self.f_clock, self.f_lab, self.f_small, self.f_card = f(26, True), f(15), f(30, True), f(14, True), f(13), f(15)
        self.trail = []

    def _font(self, size, bold):
        for name in (("segoeuisemibold", "segoeui") if bold else ("segoeui",)):
            path = self.pg.font.match_font(name)
            if path:
                return self.pg.font.Font(path, size)
        return self.pg.font.SysFont(None, size, bold=bold)

    def xy(self, lon, lat):
        return ((self.cx + (lon - self.lon0) * self.k * self.s) * self.ss, (self.cy + (self.lat1 - lat) * self.s) * self.ss)

    def px(self, metres):
        return max(1, int(round(metres / self.m_per_px * self.ss)))

    def _base(self):
        pg = self.pg; surf = pg.Surface((self.W * self.ss, self.H * self.ss)); surf.fill(BG); surf.set_clip(self.clip)
        order = [("apron", APRON), ("terminal", BUILDING), ("hangar", BUILDING), ("taxiway", TAXIWAY), ("runway", RUNWAY), ("parking_position", STAND)]
        for kind, col in order:
            for kd, k, pts, ref in self.parts:
                if kd != kind:
                    continue
                xy = [self.xy(x, y) for x, y in pts]
                if k == "pg" and len(xy) >= 3:
                    pg.draw.polygon(surf, col, xy)
                elif k == "ln" and len(xy) >= 2:
                    w = self.px(45 if kind == "runway" else 14 if kind == "taxiway" else 6)
                    pg.draw.lines(surf, col, False, xy, w)
                    for q in xy:
                        pg.draw.circle(surf, col, q, w // 2)
                    if kind == "runway":                                   # dashed centreline
                        for a, b in zip(xy[:-1], xy[1:]):
                            L = math.dist(a, b); n = int(L // (self.ss * 14))
                            for i in range(0, n, 2):
                                p0 = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n); p1 = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                                pg.draw.line(surf, RW_MARK, p0, p1, max(1, self.ss))
                elif k == "pt" and kind == "parking_position":
                    pg.draw.circle(surf, col, self.xy(*pts[0]), self.ss * 1.5)
        surf.set_clip(None); rule = (232, 232, 228)
        pg.draw.line(surf, rule, (40 * self.ss, self.clip.top), (self.W * self.ss - 40 * self.ss, self.clip.top), self.ss)
        pg.draw.line(surf, rule, (40 * self.ss, self.clip.bottom), (self.W * self.ss - 40 * self.ss, self.clip.bottom), self.ss)
        return surf

    # ------------------------------------------------------------------ aircraft
    @staticmethod
    def state(tr: np.ndarray, T: float):
        t = tr[:, 0]
        if T < t[0] - 5 or T > t[-1] + 20:
            return None
        i = int(np.searchsorted(t, T)); i = min(max(i, 1), len(t) - 1); a, b = tr[i - 1], tr[i]
        if T <= t[0]:
            a = b = tr[0]
        elif T >= t[-1]:
            a = b = tr[-1]
        if b[0] - a[0] > 120 and not (T - a[0] < 20 or b[0] - T < 20):  # long silence: not shown
            return None
        w = 0.0 if b[0] == a[0] else (T - a[0]) / (b[0] - a[0])
        lon, lat = a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w
        hdg = a[3] if a[3] == a[3] else (b[3] if b[3] == b[3] else 0.0)
        if b[3] == b[3] and a[3] == a[3]:
            d = ((b[3] - a[3] + 540) % 360) - 180; hdg = a[3] + d * w
        return lon, lat, hdg, (a[4] if a[4] == a[4] else 0.0), bool(a[5] if w < 0.5 else b[5]), a[6]

    def glyph(self, surf, x, y, hdg, col, size):
        """An aircraft shape (fuselage, wings, tail) pointing along hdg (degrees from north)."""
        L = size; pts = [(0, -0.50), (0.06, -0.40), (0.07, -0.08), (0.50, 0.10), (0.50, 0.17), (0.07, 0.10), (0.05, 0.36), (0.20, 0.47),
                         (0.20, 0.52), (0, 0.47), (-0.20, 0.52), (-0.20, 0.47), (-0.05, 0.36), (-0.07, 0.10), (-0.50, 0.17), (-0.50, 0.10),
                         (-0.07, -0.08), (-0.06, -0.40)]
        c, s = math.cos(math.radians(hdg)), math.sin(math.radians(hdg))
        self.pg.draw.polygon(surf, col, [(x + (px * c - py * s) * L, y + (px * s + py * c) * L) for px, py in pts])

    # ------------------------------------------------------------------ frame
    def draw(self, T: float, speed: float):
        pg, ss, m = self.pg, self.ss, self.m
        surf = self.base.copy(); surf.set_clip(self.clip)
        moving = 0; tkos = 0; ldgs = 0
        size_other, size_me = max(self.px(48), 18 * ss), max(self.px(48) * 1.3, 26 * ss)
        for ic, tr in self.tracks.items():
            if ic == self.m["icao"]:
                continue
            st = self.state(tr, T)
            if st is None:
                continue
            lon, lat, hdg, gs, ground, alt = st; x, y = self.xy(lon, lat)
            hist = tr[(tr[:, 0] <= T) & (tr[:, 0] >= T - 60)]
            if len(hist) >= 2 and ground:
                pg.draw.lines(surf, TRAIL, False, [self.xy(p[1], p[2]) for p in hist] + [(x, y)], max(1, ss))
            if ground and gs >= 3:
                moving += 1
            self.glyph(surf, x, y, hdg, GROUND if ground else AIR, size_other * (1.0 if ground else 1.15))
            g = tr[:, 5]; tt = tr[:, 0]; flips = np.where(np.diff(g) != 0)[0]
            for f in flips:
                tf = tt[f + 1]
                if tf <= T:
                    if g[f] == 1 and g[f + 1] == 0:
                        tkos += 1
                    elif g[f] == 0 and g[f + 1] == 1:
                        ldgs += 1
                if 0 <= T - tf <= 12 and g[f] == 1 and g[f + 1] == 0:          # a take-off: a short fading ring
                    r = (T - tf) / 12; rad = int((10 + 40 * r) * ss); col = tuple(int(v + (255 - v) * r) for v in INK2)
                    pg.draw.circle(surf, col, self.xy(tr[f + 1, 1], tr[f + 1, 2]), rad, max(1, ss))
        # SWR758: trail coloured from light to the accent, then the glyph and a label
        me = self.tracks[self.m["icao"]]; st = self.state(me, T)
        g = me[:, 5]; f = np.where((g[:-1] == 1) & (g[1:] == 0))[0]
        tkos += int(np.sum(me[f + 1, 0] <= T))
        hist = me[(me[:, 0] <= T) & (me[:, 0] >= m["blk"] - 30)]
        if len(hist) >= 2:
            n = len(hist)
            for i in range(n - 1):
                w = i / max(n - 2, 1); col = tuple(int(255 + (a - 255) * (0.35 + 0.65 * w)) for a in ACCENT)
                pg.draw.line(surf, col, self.xy(hist[i, 1], hist[i, 2]), self.xy(hist[i + 1, 1], hist[i + 1, 2]), max(2, int(2.2 * ss)))
        if st is not None:
            lon, lat, hdg, gs, ground, alt = st; x, y = self.xy(lon, lat)
            pg.draw.circle(surf, (253, 226, 214), (x, y), size_me * 0.85)
            self.glyph(surf, x, y, hdg, ACCENT, size_me)
            lab = self.f_lab.render(m["flight"], True, ACCENT); surf.blit(lab, (x + size_me * 0.75, y - size_me * 0.95))
        surf.set_clip(None)
        self._hud(surf, T, speed, moving, tkos, ldgs, st)
        return pg.transform.smoothscale(surf, (self.W, self.H)) if ss != 1 else surf

    def _status(self, T, st):
        m = self.m
        if T < m["move"]:
            return "at stand"
        if T < m["lineup"]:
            return "taxiing" if st is not None and st[3] >= 2 else "holding"
        if st is None or not st[4]:
            return "airborne"
        return "take-off roll" if st[3] >= 30 else "lined up"

    def _hud(self, surf, T, speed, moving, tkos, ldgs, st):
        pg, ss, m, W, H = self.pg, self.ss, self.m, self.W * self.ss, self.H * self.ss
        pad = 40 * ss
        surf.blit(self.f_title.render(f"{NAMES.get(m['airport'], m['airport'])} ({m['airport']}): aircraft on the ground, from ADS-B", True, INK), (pad, 30 * ss))
        surf.blit(self.f_sub.render(f"{pd.Timestamp(m['day']).day} {pd.Timestamp(m['day']):%B %Y}  ·  highlighted: " + m["flight"] + f" ({m['type']}), stand {m['stand']} to runway {m['runway']}",
                                    True, INK2), (pad, 68 * ss))
        clock = self.f_clock.render(pd.Timestamp(T, unit="s", tz="UTC").strftime("%H:%M:%S") + " UTC", True, INK)
        surf.blit(clock, (W - pad - clock.get_width(), 26 * ss))
        sp = self.f_small.render(f"×{speed:g}  ·  on the ground moving {moving}  ·  take-offs {tkos}  ·  landings {ldgs}", True, INK3)
        surf.blit(sp, (W - pad - sp.get_width(), 72 * ss))
        # SWR758 card
        el = T - m["blk"]; txt = f"{'+' if el >= 0 else '−'}{int(abs(el)) // 60:d}:{int(abs(el)) % 60:02d} since the recorded off-block"
        lines = [(m["flight"] + "  " + self._status(T, st), ACCENT, self.f_lab), (txt, INK2, self.f_card),
                 (f"recorded taxi-out {int(m['label']) // 60}:{int(m['label']) % 60:02d}", INK3, self.f_card)]
        y = 104 * ss
        for s_, col, fnt in lines:
            r = fnt.render(s_, True, col); surf.blit(r, (W - pad - r.get_width(), y)); y += r.get_height() + 2 * ss
        # timeline
        y0 = H - 92 * ss; x0, x1 = pad, W - pad; t0, t1 = m["t0"], m["t1"]; X = lambda t: x0 + (t - t0) / (t1 - t0) * (x1 - x0)
        pg.draw.line(surf, (226, 226, 221), (x0, y0), (x1, y0), 3 * ss)
        pg.draw.line(surf, ACCENT, (X(m["blk"]), y0), (X(min(max(T, m["blk"]), m["tko"])), y0), 3 * ss) if T > m["blk"] else None
        ev = [("recorded off-block", m["blk"], C_BLOCK), ("NM AOBT_3", m["aobt3"], C_AOBT), ("first movement", m["move"], C_MOVE),
              ("line-up", m["lineup"], C_LINEUP), ("take-off", m["tko"], C_TKO)]
        last_x = -1e9; row = 0
        for name, t, col in ev:
            if t != t:
                continue
            x = X(t); pg.draw.line(surf, col, (x, y0 - 9 * ss), (x, y0 + 9 * ss), 2 * ss)
            row = row + 1 if x - last_x < 150 * ss else 0; last_x = x
            lab = self.f_small.render(f"{name} {pd.Timestamp(t, unit='s', tz='UTC'):%H:%M:%S}", True, col)
            surf.blit(lab, (min(max(x - lab.get_width() / 2, x0), x1 - lab.get_width()), y0 + 14 * ss + row * 18 * ss))
        pg.draw.circle(surf, INK, (X(min(max(T, t0), t1)), y0), 6 * ss)
        cred = self.f_small.render("ADS-B: adsb.lol (ODbL)  ·  map: © OpenStreetMap contributors (ODbL)  ·  times: PRC Data Challenge 2026", True, INK3)
        surf.blit(cred, (pad, H - 26 * ss))


# ---------------------------------------------------------------------------------------------------------- run
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--export", type=Path); ap.add_argument("--speed", type=float, default=20)
    ap.add_argument("--mvt", type=int, default=186843268, help="MVT_ID_mvt of the departure (default: SWR758, Zurich, 2 April 2025)")
    ap.add_argument("--osm-snapshot", type=Path, default=None, help="folder with <ICAO>/aeroways_raw.parquet instead of the Overpass query")
    ap.add_argument("--supersample", type=int, default=2, help="render at this multiple and scale down (2: smoother edges; 1: half the memory)")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--size", default="1920x1080"); ap.add_argument("--stills", type=int, default=5)
    a = ap.parse_args(); W, H = map(int, a.size.split("x"))
    if a.export:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init(); data = prepare(a.mvt, a.osm_snapshot)
    screen = pygame.display.set_mode((W, H) if not a.export else (1, 1)); pygame.display.set_caption("ADS-B ground movements")
    sc = Scene(data, W, H, ss=a.supersample); m = data["meta"]; T = m["t0"]
    if a.export:
        import imageio.v2 as iio
        frames = int((m["t1"] - m["t0"]) / a.speed * a.fps); w = iio.get_writer(str(a.export), fps=a.fps, codec="libx264", quality=9, macro_block_size=8)
        still_at = set(np.linspace(0, frames - 1, a.stills).astype(int)) | {int((t - m["t0"]) / a.speed * a.fps) for t in (m["move"] + 60, m["lineup"], m["tko"] + 20)}
        for i in range(frames):
            img = sc.draw(m["t0"] + i * a.speed / a.fps, a.speed)
            w.append_data(np.transpose(pygame.surfarray.array3d(img), (1, 0, 2)))
            if i in still_at:
                pygame.image.save(img, str(a.export.with_name(f"{a.export.stem}_{i:05d}.png")))
            if i % 300 == 0:
                print(f"frame {i}/{frames}", flush=True)
        w.close(); print("wrote", a.export); return
    clock = pygame.time.Clock(); speed = a.speed; paused = False
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE: paused = not paused
                elif ev.key == pygame.K_RIGHT: T += 10
                elif ev.key == pygame.K_LEFT: T -= 10
                elif ev.key == pygame.K_UP: speed = min(speed * 2, 160)
                elif ev.key == pygame.K_DOWN: speed = max(speed / 2, 1)
                elif ev.key == pygame.K_r: T = m["t0"]
                elif ev.key == pygame.K_s: OUT.mkdir(parents=True, exist_ok=True); pygame.image.save(screen, str(OUT / f"still_{int(T)}.png"))
        dt = clock.tick(30) / 1000
        if not paused:
            T += dt * speed
        if T > m["t1"]:
            T = m["t0"]
        screen.blit(sc.draw(max(T, m["t0"]), speed), (0, 0)); pygame.display.flip()


if __name__ == "__main__":
    main()
