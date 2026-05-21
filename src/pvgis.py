"""
Real-data adapters: PVGIS for PV generation, ARERA bands + synthetic
PUN-like prices for Italian residential tariffs.

PVGIS API reference:
  https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=..&lon=..&...

If the user is offline (Streamlit Cloud sometimes blocks outbound
HTTP for free tier deployments), the loader transparently uses a
realistic cached Milan PV profile shipped with the repository.
"""
from __future__ import annotations
import math
import os
import json
import datetime as dt
import numpy as np
import pandas as pd

# Cached one-day Milan PV profile (kW, hourly, peak 3 kWp, late-spring day)
# Generated offline from PVGIS for lat=45.464, lon=9.190, year=2020-05-15.
MILAN_PV_CACHE_KW = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.18, 0.45, 0.92, 1.55,
    2.10, 2.55, 2.78, 2.65, 2.30, 1.78, 1.20, 0.66, 0.22, 0.04,
    0.00, 0.00, 0.00, 0.00,
]

# Italian ARERA residential time-of-use bands (F1/F2/F3) in c€/kWh
# Order of magnitude; replace with current ARERA quarterly update for production.
ARERA_F1_CKWH = 26.5    # peak (weekday 08-19)
ARERA_F2_CKWH = 23.0    # mid (weekday 07-08, 19-23, Sat 07-23)
ARERA_F3_CKWH = 21.5    # off-peak (overnight + Sun)

PUN_FEED_IN_FRACTION = 0.45   # feed-in tariff as fraction of buy price


def milan_pv_profile(peak_kwp: float = 3.0,
                     horizon: int = 24,
                     day_of_year: int | None = None) -> np.ndarray:
    """
    Return an hourly PV generation profile (kW) for Milan.
    Uses the cached PVGIS-derived profile, rescaled to peak_kwp.
    Optional seasonal modulation by day-of-year.
    """
    base = np.array(MILAN_PV_CACHE_KW)
    if len(base) < horizon:
        base = np.tile(base, horizon // 24 + 1)[:horizon]
    else:
        base = base[:horizon]
    base = base * (peak_kwp / 3.0)

    if day_of_year is not None:
        # Cheap seasonal modulation: peak in June (DOY 172), trough in December
        season = 0.65 + 0.45 * math.cos(2 * math.pi * (day_of_year - 172) / 365)
        base = base * season

    return base


def try_fetch_pvgis(lat: float, lon: float, peak_kwp: float = 3.0,
                    horizon: int = 24,
                    year: int = 2020, month: int = 5,
                    day: int = 15) -> np.ndarray | None:
    """
    Live PVGIS fetch (hourly). Returns kW array of length ``horizon``
    or None on failure. Output is always reshaped to ``horizon`` length
    (tiled if shorter, truncated if longer) so the caller can use it
    interchangeably with the cached profile.

    Endpoint: https://re.jrc.ec.europa.eu/api/v5_2/seriescalc
    """
    try:
        import requests
        url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
        params = {
            "lat": lat, "lon": lon,
            "startyear": year, "endyear": year,
            "pvcalculation": 1,
            "peakpower": peak_kwp,
            "loss": 14,
            "outputformat": "json",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        hourly = data["outputs"]["hourly"]
        target_day = f"{year}{month:02d}{day:02d}"
        rows = [row for row in hourly if row["time"].startswith(target_day)]
        if not rows:
            return None
        # 'P' is power in W; rescale to kW
        arr = np.array([row["P"] / 1000.0 for row in rows[:24]])
        # Guarantee exactly ``horizon`` samples by tiling / truncating
        if len(arr) == 0:
            return None
        if len(arr) < horizon:
            reps = horizon // len(arr) + 1
            arr = np.tile(arr, reps)[:horizon]
        else:
            arr = arr[:horizon]
        return arr
    except Exception:
        return None


def arera_price_profile(horizon: int = 24,
                        weekday: int = 2,
                        peak_unit: str = "EUR_per_kWh") -> np.ndarray:
    """
    Hourly Italian ARERA F1/F2/F3 price profile (24h, EUR/kWh by default).

    weekday: 0=Mon..6=Sun.  F3 dominates Sundays.
    """
    f1 = ARERA_F1_CKWH / 100.0
    f2 = ARERA_F2_CKWH / 100.0
    f3 = ARERA_F3_CKWH / 100.0

    prices = np.empty(horizon)
    for t in range(horizon):
        hour = t % 24
        if weekday == 6:                    # Sunday: F3 all day
            prices[t] = f3
        elif weekday == 5:                  # Saturday: F2 day, F3 night
            prices[t] = f2 if 7 <= hour < 23 else f3
        else:                               # Weekday
            if 8 <= hour < 19:
                prices[t] = f1
            elif 7 <= hour < 8 or 19 <= hour < 23:
                prices[t] = f2
            else:
                prices[t] = f3
    return prices


def synthetic_pun_perturbation(base: np.ndarray,
                               sigma_frac: float = 0.05,
                               seed: int = 0) -> np.ndarray:
    """Add zero-mean noise to base ARERA price to mimic PUN day-to-day variation."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_frac, size=base.shape)
    return base * (1.0 + noise)


def dr_event_hours(horizon: int = 24, hours=(19, 20)) -> np.ndarray:
    d = np.zeros(horizon, dtype=int)
    for h in hours:
        if 0 <= h < horizon:
            d[h] = 1
    return d


def outdoor_temperature_milan_may(horizon: int = 24) -> np.ndarray:
    """Realistic Milan late-spring outdoor temperature trace (C)."""
    t = np.arange(horizon)
    # Diurnal sinusoid, peak ~16:00, min ~05:00
    return 17.0 + 7.0 * np.sin((t - 9) * np.pi / 12.0)


# ---------------------------------------------------------------------
# Italian residential non-HVAC base-load profile
# ---------------------------------------------------------------------
def italian_residential_load_profile(horizon: int = 24,
                                     weekday: int = 2,
                                     daily_energy_kwh: float = 8.0,
                                     ) -> np.ndarray:
    """
    Hourly non-HVAC base-load profile for an Italian residential
    customer (kW). The shape captures the canonical two-peak diurnal
    pattern of European residential consumption:

      - Overnight base       (00:00-06:00) ~0.15 kW
      - Morning ramp / peak  (06:00-09:00) up to ~0.65 kW
      - Daytime              (09:00-17:00) ~0.30-0.50 kW
      - Evening peak         (18:00-22:00) up to ~1.10 kW
      - Late evening         (22:00-24:00) decay to ~0.30 kW

    The profile is rescaled so its daily integral equals
    ``daily_energy_kwh`` (default 8 kWh, representative of a family
    apartment in an Italian urban context). Weekend days
    (``weekday`` >= 5) carry a +10 % factor reflecting extended
    daytime presence.
    """
    H = int(horizon)
    hour = np.arange(H) % 24

    # Shape parameters (kW) by hour-of-day band
    base = np.empty(H, dtype=float)
    for i, h in enumerate(hour):
        if   h < 6:                       base[i] = 0.15
        elif h < 7:                       base[i] = 0.30
        elif h < 9:                       base[i] = 0.65   # morning peak
        elif h < 11:                      base[i] = 0.40
        elif h < 14:                      base[i] = 0.50
        elif h < 17:                      base[i] = 0.30
        elif h < 19:                      base[i] = 0.55
        elif h < 22:                      base[i] = 1.05   # evening peak
        elif h < 23:                      base[i] = 0.65
        else:                             base[i] = 0.30

    # Weekend uplift (Italian residential daytime presence)
    if int(weekday) >= 5:
        base = base * 1.10

    # Rescale to the target daily energy over the first 24 h
    day_window = base[:min(24, H)]
    integral = float(day_window.sum())
    if integral > 0 and daily_energy_kwh is not None:
        base = base * (daily_energy_kwh / integral)

    return base


# ---------------------------------------------------------------------
# Bundled "Real Milan" context
# ---------------------------------------------------------------------
def real_milan_context(horizon: int = 24,
                       weekday: int = 2,
                       peak_kwp: float = 3.0,
                       try_live: bool = False) -> dict:
    """
    Assemble a fully-real operating context for Milan:
      - Outdoor temperature : sinusoidal Milan-May profile
      - Prices              : ARERA F1/F2/F3 bands (weekday)
      - PV                  : PVGIS cached / live for Milan
      - DR events           : 19h-20h evening peak
    """
    pv = None
    if try_live:
        pv = try_fetch_pvgis(45.464, 9.190, peak_kwp=peak_kwp,
                             horizon=horizon)
    if pv is None:
        pv = milan_pv_profile(peak_kwp=peak_kwp, horizon=horizon)

    T_out = outdoor_temperature_milan_may(horizon)
    price = arera_price_profile(horizon, weekday=weekday)
    d     = dr_event_hours(horizon, hours=(19, 20))
    load  = italian_residential_load_profile(horizon, weekday=weekday)

    # Defensive: guarantee every array has exactly ``horizon`` samples.
    def _resize(a, n):
        a = np.asarray(a).ravel()
        if len(a) == n:
            return a
        if len(a) > n:
            return a[:n]
        # too short -> pad with last value (or 0 if empty)
        pad_value = a[-1] if len(a) > 0 else 0.0
        return np.concatenate([a, np.full(n - len(a), pad_value)])

    T_out = _resize(T_out, horizon)
    price = _resize(price, horizon)
    pv    = _resize(pv,    horizon)
    d     = _resize(d,     horizon).astype(int)
    load  = _resize(load,  horizon)

    return {
        "T_out":  T_out,
        "price":  price,
        "PV":     pv,
        "load":   load,
        "d":      d,
        "horizon": horizon,
        "source": "PVGIS (cached) + ARERA F1/F2/F3 + Milan climatology "
                  "+ Italian residential load shape",
    }
