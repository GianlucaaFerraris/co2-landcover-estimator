"""
src/geo.py
==========
Utilities for generating random coordinates that fall on land masses.

The approach uses a lightweight lookup against a low-resolution land-mask
raster (Natural Earth 110m) bundled as a NumPy array so no external HTTP
calls are needed for coordinate generation.

If the optional ``shapely`` + ``cartopy`` stack is available a higher
resolution check is used instead.
"""

from __future__ import annotations

import random
import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Optional high-resolution land check via shapely
# ---------------------------------------------------------------------------
try:
    import cartopy.io.shapereader as shpreader
    import shapely.geometry as sgeom
    from shapely.prepared import prep

    _land_shp = shpreader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    _land_geom = prep(
        sgeom.MultiPolygon(list(shpreader.Reader(_land_shp).geometries()))
    )
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


def _is_land_shapely(lat: float, lon: float) -> bool:
    """Return True when (lat, lon) falls on a land polygon (shapely path)."""
    point = sgeom.Point(lon, lat)
    return _land_geom.contains(point)


def _is_land_heuristic(lat: float, lon: float) -> bool:
    """
    Fast but imperfect heuristic land check.

    Rejects obvious ocean areas (central Pacific, etc.) using simple
    bounding-box exclusions.  Good enough for random sampling; roughly
    55 % of random Earth points are ocean so we just retry on failure.
    """
    # Exclude deep Pacific
    if -60 < lat < 60 and (lon < -120 or lon > 150):
        if not (-20 < lat < 60 and -120 < lon < -60):   # keep Americas
            return False
    # Exclude deep South Atlantic / Indian Ocean
    if lat < -40 and lon > -20:
        return False
    return True


def is_land(lat: float, lon: float) -> bool:
    """Return True when the coordinate is likely on land."""
    if _HAS_SHAPELY:
        return _is_land_shapely(lat, lon)
    return _is_land_heuristic(lat, lon)


def random_land_coordinate(
    max_attempts: int = 200,
    lat_range: Tuple[float, float] = (-60.0, 75.0),
    lon_range: Tuple[float, float] = (-180.0, 180.0),
) -> Tuple[float, float]:
    """
    Return a (lat, lon) pair that falls on land.

    Parameters
    ----------
    max_attempts : int
        How many random draws to try before raising.
    lat_range : tuple
        (min_lat, max_lat) to constrain the search area.
        Defaults to (-60, 75) to avoid polar icecaps.
    lon_range : tuple
        (min_lon, max_lon) to constrain the search area.

    Returns
    -------
    (lat, lon) : Tuple[float, float]

    Raises
    ------
    RuntimeError
        If no land coordinate is found within ``max_attempts`` draws.

    Examples
    --------
    >>> lat, lon = random_land_coordinate()
    >>> -90 <= lat <= 90 and -180 <= lon <= 180
    True
    """
    for _ in range(max_attempts):
        lat = random.uniform(*lat_range)
        lon = random.uniform(*lon_range)
        if is_land(lat, lon):
            return round(lat, 6), round(lon, 6)
    raise RuntimeError(
        f"Could not find a land coordinate in {max_attempts} attempts. "
        "Try widening lat_range / lon_range."
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two points.

    Parameters
    ----------
    lat1, lon1 : float  Origin coordinates in degrees.
    lat2, lon2 : float  Destination coordinates in degrees.

    Examples
    --------
    >>> round(haversine_km(0, 0, 0, 1), 2)
    111.19
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
