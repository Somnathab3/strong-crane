"""Shared fixtures for the pre-training checks (tests/test_pretrain.py): real data, loaded once per session.

The slice: every EHAM departure on ten days - six 2025 development months, one January and one July 2025 day (the
holdout months) and one January and one July 2026 day (ranking) - about 13,000 rows. LIRF-specific stages are checked
on LIRF rows separately. Evidence from every test is collected into results/pretrain_report.md.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO / "src"))        # src layout: `import taxiout`
SLICE_AIRPORT = "EHAM"
SLICE_DAYS = ["2025-02-12", "2025-04-09", "2025-06-11", "2025-08-13", "2025-10-15", "2025-12-10",   # development months
              "2025-01-15", "2025-07-16",                                                       # holdout months
              "2026-01-14", "2026-07-15"]                                                       # ranking
_REPORT: list[str] = []


def _day(t: np.ndarray) -> np.ndarray:
    return pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d").to_numpy()


@pytest.fixture(scope="session")
def report():
    def add(title: str, text: str) -> None:
        _REPORT.append(f"## {title}\n\n{text}\n")
        print(f"\n=== {title}\n{text}")
    return add


def pytest_sessionfinish(session, exitstatus):
    if _REPORT:
        p = REPO / "results" / "pretrain_report.md"; p.parent.mkdir(exist_ok=True)
        head = f"# Pre-training checks ({dt.datetime.now():%Y-%m-%d %H:%M}), slice {SLICE_AIRPORT} x {len(SLICE_DAYS)} days\n\n"
        p.write_text(head + "\n".join(_REPORT), encoding="utf-8")


@pytest.fixture(scope="session")
def raw_movements():
    """The organiser table exactly as data.load returns it (before field filling)."""
    from taxiout import data
    from taxiout.config import paths
    m, _ = data.load(paths()["organiser"])
    return m


@pytest.fixture(scope="session")
def movements(raw_movements):
    from taxiout.build_extra import extra_dir
    from taxiout.fields import fill_fields
    m, rep = fill_fields(raw_movements, extra_dir())
    return m


@pytest.fixture(scope="session")
def clocks():
    from taxiout import adsb
    from taxiout.config import paths
    c, _ = adsb.load(paths()["adsb_clocks"])
    return c


@pytest.fixture(scope="session")
def features():
    """The feature table training will use (built or read from the cache by the same function run.py calls)."""
    from taxiout.run import load_features
    f, _ = load_features()
    return f


@pytest.fixture(scope="session")
def fs(features):
    """The EHAM ten-day slice of the feature table."""
    d = _day(features.t.to_numpy())
    s = features[(features.airport == SLICE_AIRPORT) & np.isin(d, SLICE_DAYS)].copy()
    s["day"] = _day(s.t.to_numpy())
    return s


@pytest.fixture(scope="session")
def ms(movements):
    """All movements (departures and arrivals) at the slice airport on the slice days +- 1 day, for rebuilding features."""
    d = pd.to_datetime(movements.t, unit="s", utc=True).dt.normalize()
    days = pd.to_datetime(SLICE_DAYS).tz_localize("UTC")
    near = np.zeros(len(movements), bool)
    for x in days:
        near |= ((d >= x - pd.Timedelta(days=1)) & (d <= x + pd.Timedelta(days=1))).to_numpy()
    return movements[(movements.airport == SLICE_AIRPORT) & near].copy()
