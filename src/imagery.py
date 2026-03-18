"""
src/imagery.py
==============
Fetch satellite imagery via ESRI World Imagery tile server.

Completely free, no API key, no account, no credit card required.
This is the same tile server used by QGIS, ArcGIS and OpenStreetMap editors.

How it works
------------
Web map tiles follow the XYZ scheme: the world is divided into a grid at
each zoom level, and each tile is a 256×256 px image identified by (z, x, y).
We convert (lat, lon, zoom) → (x, y) tile coordinates, then download a
grid of tiles and stitch them into a single image of the requested size.

ESRI tile endpoint:
    https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/
    MapServer/tile/{z}/{y}/{x}
"""

from __future__ import annotations

import io
import math
from typing import Tuple

import numpy as np
import requests
from PIL import Image

from config import DEFAULT_ZOOM, DEFAULT_SIZE

_ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_TILE_SIZE = 256  # pixels per tile (ESRI standard)
_HEADERS = {"User-Agent": "co2-landcover-estimator/1.0 (educational project)"}


# ── Coordinate math ───────────────────────────────────────────────────────────

def _deg_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert (lat, lon, zoom) to XYZ tile indices (x, y)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_pixel_offset(lat: float, lon: float, zoom: int, x: int, y: int) -> Tuple[int, int]:
    """Return the pixel offset of (lat, lon) within tile (x, y) at zoom."""
    n = 2 ** zoom
    # Pixel column within the world at this zoom
    world_px_x = (lon + 180.0) / 360.0 * n * _TILE_SIZE
    lat_rad = math.radians(lat)
    world_px_y = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0 * n * _TILE_SIZE
    )
    # Offset within the tile
    off_x = int(world_px_x - x * _TILE_SIZE)
    off_y = int(world_px_y - y * _TILE_SIZE)
    return off_x, off_y


def _fetch_tile(z: int, x: int, y: int, session: requests.Session) -> Image.Image:
    """Download a single 256×256 tile and return it as a PIL Image."""
    url = _ESRI_TILE_URL.format(z=z, y=y, x=x)
    response = session.get(url, headers=_HEADERS, timeout=15)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


# ── Public interface ──────────────────────────────────────────────────────────

def fetch_satellite_image(
    lat: float,
    lon: float,
    zoom: int = DEFAULT_ZOOM,
    size: int = DEFAULT_SIZE,
    **_kwargs,  # absorb unused api_key / token args for interface compatibility
) -> np.ndarray:
    """
    Download a satellite image centred on ``(lat, lon)`` via ESRI World Imagery.

    No API key or account needed — completely free and open.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    zoom : int
        Zoom level. Sensible range: 10–16.
        - 10 → ≈ 40 km field of view
        - 12 → ≈ 10 km field of view  (default)
        - 14 → ≈ 2.5 km field of view
    size : int
        Desired output image size in pixels (square). The function fetches
        enough tiles to cover this area and crops to exactly ``size × size``.

    Returns
    -------
    np.ndarray
        Image as (H, W, 3) uint8 array in **BGR** order (OpenCV convention).

    Raises
    ------
    requests.HTTPError
        If a tile request fails.

    Examples
    --------
    >>> img = fetch_satellite_image(-31.4, -64.2, zoom=12, size=512)
    >>> img.shape
    (512, 512, 3)
    """
    # Centre tile and pixel offset of (lat, lon) within it
    cx, cy = _deg_to_tile(lat, lon, zoom)
    off_x, off_y = _tile_to_pixel_offset(lat, lon, zoom, cx, cy)

    # How many tiles do we need on each side to cover `size` pixels?
    half = size // 2
    tiles_needed = math.ceil((half + _TILE_SIZE) / _TILE_SIZE)

    # Build a mosaic canvas
    mosaic_tiles = 2 * tiles_needed + 1
    canvas_size = mosaic_tiles * _TILE_SIZE
    canvas = Image.new("RGB", (canvas_size, canvas_size))

    session = requests.Session()
    max_tile = 2 ** zoom - 1

    for dy in range(-tiles_needed, tiles_needed + 1):
        for dx in range(-tiles_needed, tiles_needed + 1):
            tx = max(0, min(max_tile, cx + dx))
            ty = max(0, min(max_tile, cy + dy))
            try:
                tile_img = _fetch_tile(zoom, tx, ty, session)
            except Exception:
                # On failure paste a black tile and continue
                tile_img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE))

            paste_x = (dx + tiles_needed) * _TILE_SIZE
            paste_y = (dy + tiles_needed) * _TILE_SIZE
            canvas.paste(tile_img, (paste_x, paste_y))

    # The centre of the canvas corresponds to the top-left of the centre tile.
    # Adjust so (lat, lon) is at the exact centre of the crop.
    centre_canvas_x = tiles_needed * _TILE_SIZE + off_x
    centre_canvas_y = tiles_needed * _TILE_SIZE + off_y

    left   = centre_canvas_x - half
    top    = centre_canvas_y - half
    right  = left + size
    bottom = top  + size

    cropped = canvas.crop((left, top, right, bottom))

    # Convert RGB → BGR for OpenCV compatibility
    bgr = np.array(cropped)[:, :, ::-1].copy()
    return bgr


def image_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W, 3) BGR array to RGB order.

    Convenience wrapper used by the visualiser.
    """
    return image[:, :, ::-1].copy()
