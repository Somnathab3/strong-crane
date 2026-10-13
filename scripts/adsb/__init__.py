# SPDX-License-Identifier: GPL-3.0-only
"""Building the ADS-B inputs from the public adsb.lol archive (scripts/build_adsb_clocks.py, scripts/download_xplane_stands.py).

    traces.py       readsb trace parsing, geometry, archive addressing, callsign variants
    fetch.py        one sequential pass over an adsb.lol day archive, keeping only the wanted aircraft
    link.py         departure -> aircraft address (icao24) through the OPDI flight list
    stand_clock.py  the stand clock, the pushback signature and the first ground point of one departure
"""
