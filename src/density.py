"""
src/density.py
==============
Fetch population density and map it to a calibrated urban CO₂ flux.

Primary source: WorldPop REST API (free, no key)
    https://api.worldpop.org/v1/popdata/point?lat=...&lon=...&year=2020

Fallback: GHS-SMOD urban classification lookup by country — a small
built-in table of average urban densities for ~50 countries, so the
model always produces a meaningful estimate even when the API is down.

Flux scale (Churkina et al. 2010, Kennedy et al. 2011):
    < 500   hab/km²  → suburb / small town  →  +200  gC/m²/yr
    500–5k           → medium city           →  +600  gC/m²/yr
    5k–20k           → dense city            →  +1200 gC/m²/yr
    > 20k            → megacity              →  +1500 gC/m²/yr
"""

from __future__ import annotations
import requests

# ── WorldPop API endpoints (try in order) ─────────────────────────────────────
_ENDPOINTS = [
    # v1 point summary
    "https://api.worldpop.org/v1/summary/point",
    # v1 population data point (alternative schema)
    "https://api.worldpop.org/v1/popdata/point",
]
_TIMEOUT = 12

# ── Built-in density fallback by country ISO-2 code ──────────────────────────
# Average urban density (hab/km²) for major countries.
# Source: UN World Urbanization Prospects 2022 + GHS-SMOD.
_COUNTRY_DENSITY = {
    "AR": 3_800,   # Argentina — Buenos Aires metro skews high
    "BR": 4_200,
    "CL": 3_500,
    "UY": 3_200,
    "PY": 2_800,
    "BO": 2_500,
    "PE": 4_000,
    "CO": 4_500,
    "VE": 3_900,
    "MX": 4_800,
    "US": 2_000,
    "CA": 1_500,
    "GB": 4_300,
    "DE": 3_000,
    "FR": 3_200,
    "IT": 3_500,
    "ES": 3_100,
    "CN": 6_500,
    "JP": 8_000,
    "IN": 7_500,
    "NG": 5_000,
    "ZA": 3_200,
    "AU": 1_200,
}


def _reverse_geocode_country(lat: float, lon: float) -> str | None:
    """Return ISO-2 country code for a coordinate via nominatim (no key)."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "co2-landcover-estimator/1.0"},
            timeout=8,
        )
        r.raise_for_status()
        return r.json().get("address", {}).get("country_code", "").upper() or None
    except Exception:
        return None


def fetch_population_density(lat: float, lon: float) -> float | None:
    """
    Return population density in hab/km² or None on total failure.

    Tries WorldPop API first (two endpoints), then falls back to
    a country-level lookup via Nominatim + built-in density table.
    """
    # ── Try WorldPop ──────────────────────────────────────────────────────────
    for url in _ENDPOINTS:
        try:
            r = requests.get(
                url,
                params={"lat": lat, "lon": lon, "year": 2020, "dataset": "wpgpas"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            # Schema A: {"data": {"sum": <float>}}
            val = data.get("data", {}).get("sum", None)
            # Schema B: {"pop": <float>}
            if val is None:
                val = data.get("pop", None)
            if val is not None and float(val) >= 0:
                return float(val) * 100.0  # cell (100m²) → hab/km²
        except Exception:
            continue

    # ── Fallback: country-level density lookup ────────────────────────────────
    country = _reverse_geocode_country(lat, lon)
    if country and country in _COUNTRY_DENSITY:
        return float(_COUNTRY_DENSITY[country])

    return None


def urban_flux_from_density(pop_density: float | None) -> float:
    """Map hab/km² → urban CO₂ flux (gC/m²/yr)."""
    if pop_density is None:
        return 600.0
    if pop_density < 500:
        return 200.0
    elif pop_density < 5_000:
        return 600.0
    elif pop_density < 20_000:
        return 1_200.0
    else:
        return 1_500.0


def get_urban_flux(lat: float, lon: float, verbose: bool = True) -> float:
    """Full pipeline: fetch density → return calibrated urban flux."""
    density = fetch_population_density(lat, lon)
    flux    = urban_flux_from_density(density)
    if verbose:
        if density is not None:
            print(f"   Pop. density : {density:,.0f} hab/km²  ->  urban flux: {flux:.0f} gC/m²/yr")
        else:
            print(f"   Pop. density : unavailable  ->  urban flux: {flux:.0f} gC/m²/yr (fallback)")
    return flux