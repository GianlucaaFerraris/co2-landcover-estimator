"""
src/imagery.py
==============
Fetch satellite imagery via the Google Maps Static API.

The returned image is a NumPy array (H × W × 3, uint8, BGR) compatible
with OpenCV downstream functions.

Google Maps Static API docs:
    https://developers.google.com/maps/documentation/maps-static/overview

Setup
-----
    export GOOGLE_MAPS_API_KEY="your_key_here"
or add it to a .env file in the project root.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import requests
from PIL import Image

from config import GOOGLE_MAPS_API_KEY, DEFAULT_ZOOM, DEFAULT_SIZE


_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"


def fetch_satellite_image(
    lat: float,
    lon: float,
    zoom: int = DEFAULT_ZOOM,
    size: int = DEFAULT_SIZE,
    api_key: Optional[str] = None,
) -> np.ndarray:
    """
    Download a satellite image centred on ``(lat, lon)``.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    zoom : int
        Google Maps zoom level.  Sensible range: 10–16.
        - 10 → ≈ 40 km field of view
        - 12 → ≈ 10 km field of view  (default)
        - 14 → ≈ 2.5 km field of view
    size : int
        Pixel size of the (square) image.  Maximum allowed by Google: 640.
    api_key : str, optional
        Override the key from config / environment.

    Returns
    -------
    np.ndarray
        Image as (H, W, 3) uint8 array in **BGR** order (OpenCV convention).

    Raises
    ------
    EnvironmentError
        If no API key is configured.
    requests.HTTPError
        If the Google API returns a non-200 status.
    ValueError
        If the response is not a valid image.

    Examples
    --------
    >>> img = fetch_satellite_image(-31.4, -64.2, zoom=12, size=256)
    >>> img.shape
    (256, 256, 3)
    """
    key = api_key or GOOGLE_MAPS_API_KEY
    if not key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size}x{size}",
        "maptype": "satellite",
        "key": key,
    }

    response = requests.get(_STATIC_MAPS_URL, params=params, timeout=15)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise ValueError(
            f"Unexpected Content-Type '{content_type}'. "
            "The API may have returned an error page — check your key and quota."
        )

    pil_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    # Convert RGB → BGR for OpenCV compatibility
    bgr = np.array(pil_img)[:, :, ::-1].copy()
    return bgr


def image_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W, 3) BGR array to RGB order.

    Convenience wrapper used by the visualiser.
    """
    return image[:, :, ::-1].copy()
