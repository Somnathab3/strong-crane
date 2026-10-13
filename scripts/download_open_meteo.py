# SPDX-License-Identifier: GPL-3.0-only
"""Archived operational weather forecasts for the ten challenge airports.

METAR is what was observed at the field. This is what the forecast said would
happen, which is a different and arguably more operational quantity: de-icing
calls, runway configuration changes and flow restrictions are decided on
forecasts, not on a reconstruction made afterwards. So the Historical Forecast
archive is requested here rather than ERA5 reanalysis -- reanalysis assimilates
observations that only existed later, and would describe an atmosphere nobody
running the airport could see.

What this adds that METAR cannot:

    snow depth, snowfall           METAR reports present weather, not amount
    precipitation amount           likewise, in millimetres rather than a code
    freezing level height          absent from METAR entirely
    boundary layer height          absent; governs how fog and haze disperse
    CAPE                           absent; convective potential
    wet bulb temperature           usually derived, here direct
    cloud cover by layer           METAR gives aviation octas at reported bases
    soil and surface temperature   relevant to whether a surface actually ices

Coverage runs from the first training month through the end of the ranking
period, so one request per airport covers both populations.

Free tier, no key, non-commercial use. Open-Meteo counts long ranges and more
than ten variables as multiple calls, so requests are split into 14-day
chunks. Each chunk is written atomically and cached, making the download
resumable without repeating completed work.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"

#: Aerodrome reference points. Precision to four decimals is roughly 10 m,
#: far finer than the model grid, so the exact apron chosen does not matter.
AIRPORTS = {
    "EDDF": (50.0333, 8.5706),
    "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785),
    "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479),
    "LIRF": (41.8003, 12.2389),
    "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}

#: Chosen for what METAR does not carry, plus a few overlaps kept deliberately
#: so the two sources can be compared where they should agree.
HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "wet_bulb_temperature_2m",
    "precipitation", "rain", "showers", "snowfall", "snow_depth",
    "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "evapotranspiration", "vapour_pressure_deficit",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
    "wind_direction_100m", "wind_gusts_10m",
    "temperature_80m", "soil_temperature_0cm", "soil_moisture_0_to_1cm",
    "boundary_layer_height", "freezing_level_height", "cape",
    "weather_code",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fetch(icao: str, latitude: float, longitude: float,
          start: str, end: str, attempts: int = 8) -> pd.DataFrame:
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY),
        "timezone": "UTC",
        "wind_speed_unit": "kn",
    }
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(ENDPOINT, params=params, timeout=180)
            if response.status_code == 429:
                if attempt == attempts:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                wait = max(delay, float(retry_after)) if retry_after else delay
                log(f"  {icao} attempt {attempt} rate limited; retrying in {wait:.0f}s")
                time.sleep(wait)
                delay = min(delay * 2, 120.0)
                continue
            response.raise_for_status()
            payload = response.json()
            if "hourly" not in payload:
                raise RuntimeError(f"no hourly block: {list(payload)[:5]}")
            frame = pd.DataFrame(payload["hourly"])
            frame["valid_utc"] = pd.to_datetime(frame.pop("time"), utc=True)
            frame.insert(0, "icao", icao)
            return frame
        except Exception as error:  # noqa: BLE001 - retried and reported
            if attempt == attempts:
                raise
            log(f"  {icao} attempt {attempt} failed ({error}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def date_chunks(start: str, end: str, days: int) -> Iterator[tuple[str, str]]:
    """Yield inclusive fixed-length chunks covering [start, end]."""
    cursor = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    if cursor > stop:
        raise ValueError(f"start date {start} is after end date {end}")
    if days < 1:
        raise ValueError("chunk days must be positive")
    while cursor <= stop:
        chunk_end = min(stop, cursor + timedelta(days=days - 1))
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def valid_cached_chunk(path: Path, icao: str, start: str, end: str) -> bool:
    """Accept only a readable chunk with the expected airport and UTC range."""
    if not path.is_file() or path.stat().st_size <= 1000:
        return False
    try:
        frame = pd.read_parquet(path, columns=["icao", "valid_utc"])
    except Exception:  # noqa: BLE001 - a broken cache should be replaced
        return False
    expected_start = pd.Timestamp(start, tz="UTC")
    expected_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    return bool(
        len(frame)
        and frame["icao"].eq(icao).all()
        and frame["valid_utc"].min() == expected_start
        and frame["valid_utc"].max() == expected_end
        and not frame["valid_utc"].duplicated().any()
    )


def download_airport(root: Path, chunks: list[tuple[str, str]], icao: str,
                     latitude: float, longitude: float) -> Path:
    """Download missing chunks for one airport and assemble its full file."""
    chunk_root = root / "chunks"
    destination = root / f"{icao}_open_meteo.parquet"
    airport_chunks = []
    log(f"{icao}: ({latitude}, {longitude})")
    for chunk_start, chunk_end in chunks:
        chunk_path = chunk_root / f"{icao}_{chunk_start}_{chunk_end}.parquet"
        airport_chunks.append(chunk_path)
        if valid_cached_chunk(chunk_path, icao, chunk_start, chunk_end):
            log(f"  {icao} {chunk_start}..{chunk_end}: cached")
            continue
        log(f"  {icao} {chunk_start}..{chunk_end}: requesting")
        frame = fetch(icao, latitude, longitude, chunk_start, chunk_end)
        temporary = chunk_path.with_suffix(".parquet.part")
        frame.to_parquet(temporary, index=False)
        temporary.replace(chunk_path)
        log(f"    {icao}: {len(frame):,} hourly rows -> {chunk_path.name}")
        time.sleep(1.0)

    frame = pd.concat([pd.read_parquet(p) for p in airport_chunks], ignore_index=True)
    frame = frame.sort_values("valid_utc").drop_duplicates(["icao", "valid_utc"])
    temporary = destination.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    log(f"  {icao}: {len(frame):,} hourly rows -> {destination.name}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=REPOSITORY_ROOT / "data" / "raw" / "open_meteo")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    chunk_root = args.root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)

    chunks = list(date_chunks(args.start, args.end, args.chunk_days))

    log(f"Historical Forecast archive, {args.start} to {args.end}, "
        f"{len(HOURLY)} hourly variables, {len(chunks)} chunks of up to "
        f"{args.chunk_days} days per airport")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_airport, args.root, chunks, icao, latitude, longitude)
            for icao, (latitude, longitude) in AIRPORTS.items()
        ]
        written = [future.result() for future in futures]

    combined = pd.concat([pd.read_parquet(p) for p in written], ignore_index=True)
    target = args.root / "open_meteo_hourly.parquet"
    combined.to_parquet(target, index=False)
    coverage = combined.notna().mean().sort_values()
    log(f"\ncombined: {len(combined):,} rows, {combined['icao'].nunique()} airports, "
        f"{combined['valid_utc'].min()} to {combined['valid_utc'].max()}")
    log(f"wrote {target}")
    thin = coverage[coverage < 0.95]
    if len(thin):
        log("\nvariables under 95% coverage:")
        for name, value in thin.items():
            log(f"  {name:<34}{100 * value:5.1f}%")
    else:
        log("\nevery requested variable is above 95% coverage")
    (args.root / "manifest.json").write_text(json.dumps({
        "endpoint": ENDPOINT, "start": args.start, "end": args.end,
        "airports": list(AIRPORTS), "hourly": HOURLY,
        "chunk_days": args.chunk_days, "workers": args.workers,
        "http_requests": len(AIRPORTS) * len(chunks),
        "rows": int(len(combined)),
        "note": ("Archived operational forecasts, not ERA5 reanalysis: this is "
                 "what was forecast at the time rather than a later "
                 "reconstruction using observations that did not yet exist."),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
