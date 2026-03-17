"""
co2-landcover-estimator · main.py
==================================
Entry point. Picks a random land coordinate, fetches a satellite image,
segments it by land-cover type, solves the CO₂ flux ODE and renders
a result summary.

Usage
-----
    python main.py                        # fully random location
    python main.py --lat -31.4 --lon -64.2  # fixed coordinates
    python main.py --mode ml              # use SegFormer instead of HSV
    python main.py --zoom 13 --size 512   # custom image settings
"""

import argparse
import sys

from src.geo import random_land_coordinate
from src.imagery import fetch_satellite_image
from src.segmentation import segment_image
from src.co2_model import estimate_co2_flux
from src.visualizer import render_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate CO₂ flux for a random (or given) location on Earth."
    )
    parser.add_argument("--lat", type=float, default=None, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude (-180 to 180)")
    parser.add_argument(
        "--mode",
        choices=["hsv", "ml"],
        default="hsv",
        help="Segmentation mode: 'hsv' (fast) or 'ml' (SegFormer, more accurate)",
    )
    parser.add_argument("--zoom", type=int, default=12, help="Google Maps zoom level (10-16)")
    parser.add_argument("--size", type=int, default=512, help="Image size in pixels (max 640)")
    parser.add_argument("--output", type=str, default=None, help="Save result plot to this path")
    parser.add_argument(
        "--years", type=float, default=1.0, help="Time horizon for the ODE simulation (years)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Coordinates ────────────────────────────────────────────────────────
    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        print(f"📍 Using provided coordinates: ({lat:.4f}, {lon:.4f})")
    else:
        print("🌍 Picking a random land coordinate …")
        lat, lon = random_land_coordinate()
        print(f"📍 Selected: ({lat:.4f}, {lon:.4f})")

    # ── 2. Satellite image ────────────────────────────────────────────────────
    print("🛰  Fetching satellite image …")
    image = fetch_satellite_image(lat, lon, zoom=args.zoom, size=args.size)

    # ── 3. Land-cover segmentation ────────────────────────────────────────────
    print(f"🔍 Segmenting image (mode={args.mode}) …")
    cover = segment_image(image, mode=args.mode)
    print(
        f"   vegetation={cover['vegetation']:.1%}  water={cover['water']:.1%}  "
        f"arid={cover['arid']:.1%}  urban={cover['urban']:.1%}"
    )

    # ── 4. CO₂ flux model ─────────────────────────────────────────────────────
    print("🧮 Solving CO₂ flux ODE …")
    result = estimate_co2_flux(cover, years=args.years)
    flux_label = "SINK ✅" if result["net_flux_gC_m2_yr"] < 0 else "SOURCE ⚠️"
    print(f"   Net flux : {result['net_flux_gC_m2_yr']:.1f} gC/m²/yr  →  {flux_label}")
    print(f"   ΔC (1 yr): {result['delta_C_ppm']:.4f} ppm over zone")

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    print("📊 Rendering results …")
    render_results(
        image=image,
        cover=cover,
        result=result,
        lat=lat,
        lon=lon,
        output_path=args.output,
    )
    print("Done." if args.output else "Done. Close the plot window to exit.")


if __name__ == "__main__":
    main()
