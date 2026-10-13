# SPDX-License-Identifier: GPL-3.0-only
"""One sequential pass over an adsb.lol day archive, keeping only the wanted aircraft.

Each adsb.lol day (GitHub releases of adsblol/globe_history_<year>, ODbL 1.0) is an uncompressed tar split into ~2 GB
parts. Tar has no central directory and the members come in directory order, so the only correct retrieval is a
sequential pass: one pass per UTC day for all of the day's aircraft, stopped as soon as the last wanted member has been
seen; only the per-aircraft traces (tens of KB each) are written, never the day archive; cached traces are not fetched
again.
"""
from __future__ import annotations

import gzip
import json
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .traces import release_tag, wanted_members

GITHUB_API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
USER_AGENT = "strong-crane/adsb-clocks (research; ODbL data)"


class _AllFound(Exception):
    """Stops the tar stream once every wanted member has been seen."""


@dataclass
class PartStream:
    """Forward-only reader over the ordered parts of one split tar (tarfile's stream mode never seeks back)."""
    urls: list[str]
    timeout: float = 120.0
    bytes_read: int = 0

    def __post_init__(self) -> None:
        self._index = 0; self._handle = None

    def _open_next(self):
        if self._index >= len(self.urls):
            return None
        request = urllib.request.Request(self.urls[self._index], headers={"User-Agent": USER_AGENT}); self._index += 1
        return urllib.request.urlopen(request, timeout=self.timeout)

    def read(self, size: int = -1) -> bytes:
        if self._handle is None:
            self._handle = self._open_next()
            if self._handle is None:
                return b""
        chunks: list[bytes] = []; remaining = size
        while remaining != 0:
            piece = self._handle.read(remaining if remaining > 0 else 1 << 20)
            if piece:
                chunks.append(piece); self.bytes_read += len(piece)
                if remaining > 0:
                    remaining -= len(piece)
                continue
            self._handle.close(); self._handle = self._open_next()
            if self._handle is None:
                break
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


def resolve_assets(tag: str, timeout: float = 60.0) -> list[dict]:
    """Ordered asset list (.tar.aa before .tar.ab) of one day's release."""
    request = urllib.request.Request(GITHUB_API.format(year=tag[1:5], tag=tag), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return sorted(({"name": a["name"], "size": a["size"], "url": a["browser_download_url"]} for a in payload.get("assets") or []),
                  key=lambda a: a["name"])


def fetch_day(date: str, icao24s: list[str], raw_root: Path, *, pod: str = "planes-readsb-prod-0",
              timeout: float = 180.0, force: bool = False) -> dict:
    """Write <raw_root>/<date>/trace_full_<icao24>.json.gz for every wanted aircraft found in the day's archive."""
    tag = release_tag(date, pod=pod); day_dir = Path(raw_root) / date; day_dir.mkdir(parents=True, exist_ok=True)
    outstanding = {m: c for m, c in wanted_members(icao24s).items()
                   if force or not ((day_dir / f"trace_full_{c}.json.gz").exists() and (day_dir / f"trace_full_{c}.json.gz").stat().st_size > 0)}
    if not outstanding:
        return {"date": date, "tag": tag, "wanted": len(icao24s), "found": 0, "cached": len(icao24s), "gb_streamed": 0.0}
    assets = resolve_assets(tag, timeout=timeout)
    if not assets:
        return {"date": date, "tag": tag, "wanted": len(icao24s), "found": 0, "cached": len(icao24s) - len(outstanding), "gb_streamed": 0.0, "note": "release missing"}
    print(f"  {date}: streaming {tag} ({sum(a['size'] for a in assets) / 1e9:.2f} GB in {len(assets)} parts) for {len(outstanding)} aircraft", flush=True)
    stream = PartStream([a["url"] for a in assets], timeout=timeout); started = time.time(); found = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                code = outstanding.get(member.name)
                if code is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                payload = handle.read(); destination = day_dir / f"trace_full_{code}.json.gz"
                destination.write_bytes(payload if payload[:2] == b"\x1f\x8b" else gzip.compress(payload))   # members are usually gzip already
                found += 1
                if found == len(outstanding):
                    raise _AllFound
    except _AllFound:
        pass
    except (tarfile.TarError, urllib.error.URLError, OSError) as error:
        print(f"    stream ended early: {type(error).__name__}: {error}", flush=True)
    finally:
        stream.close()
    print(f"  {date}: {found}/{len(outstanding)} found, {stream.bytes_read / 1e9:.2f} GB streamed in {time.time() - started:.0f} s", flush=True)
    return {"date": date, "tag": tag, "wanted": len(icao24s), "found": found, "cached": len(icao24s) - len(outstanding), "gb_streamed": stream.bytes_read / 1e9}
