"""
co2-landcover-estimator · main.py
==================================
Entry point. Picks a random land coordinate, fetches a satellite image,
segments it by land-cover type, solves the CO₂ flux ODE and renders
a result summary.

Usage
-----
    python main.py                              # random location, barrio scale
    python main.py --lat -31.4 --lon -64.2      # fixed coordinates
    python main.py --scale ciudad               # city-wide view
    python main.py --scale barrio --mode ml     # neighbourhood + SegFormer
    python main.py --output result.png          # save plot to file
"""

import argparse

from src.geo import random_land_coordinate
from src.imagery import fetch_satellite_image
from src.segmentation import segment_image
from src.density import get_urban_flux
from src.co2_model import estimate_co2_flux
from src.visualizer import render_results
from config import SCALE_PRESETS, DEFAULT_SCALE, DEFAULT_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate CO₂ flux and tree offset for any location on Earth."
    )
    parser.add_argument("--lat",  type=float, default=None, help="Latitude  (-90 to 90)")
    parser.add_argument("--lon",  type=float, default=None, help="Longitude (-180 to 180)")
    parser.add_argument(
        "--scale",
        choices=list(SCALE_PRESETS.keys()),
        default=DEFAULT_SCALE,
        help=(
            "Spatial scale of analysis: "
            "'barrio' (~500 m, default), "
            "'ciudad' (~10 km), "
            "'region' (~40 km)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["hsv", "ml", "hybrid"],
        default="hybrid",
        help="Segmentation mode: 'hybrid' (default), 'hsv' (fast), 'ml' (full SegFormer)",
    )
    parser.add_argument("--size",   type=int,   default=DEFAULT_SIZE, help="Image size in pixels")
    parser.add_argument("--output", type=str,   default=None,         help="Save result plot to this path")
    parser.add_argument("--years",  type=float, default=1.0,          help="ODE time horizon (years)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Scale preset ──────────────────────────────────────────────────────────
    preset = SCALE_PRESETS[args.scale]
    zoom   = preset["zoom"]
    print(f"🔭 Scale: {args.scale}  ({preset['label']}, zoom={zoom})")

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
    image = fetch_satellite_image(lat, lon, zoom=zoom, size=args.size)

    # ── 3. Land-cover segmentation ────────────────────────────────────────────
    print(f"🔍 Segmenting image (mode={args.mode}) …")
    cover = segment_image(image, mode=args.mode)
    print(
        f"   vegetation={cover['vegetation']:.1%}  water={cover['water']:.1%}  "
        f"arid={cover['arid']:.1%}  urban={cover['urban']:.1%}"
    )

    # ── 4. Dynamic urban flux from population density ─────────────────────────
    print("🏙  Fetching population density …")
    urban_flux = get_urban_flux(lat, lon, verbose=True)

    # ── 5. CO₂ flux model ─────────────────────────────────────────────────────
    print("🧮 Solving CO₂ flux ODE …")
    result = estimate_co2_flux(
        cover,
        lat=lat,
        zoom=zoom,
        size_px=args.size,
        urban_flux=urban_flux,
        years=args.years,
    )

    flux_label = "SINK ✅" if result["is_sink"] else "SOURCE ⚠️"
    print(f"   Net flux : {result['net_flux_gC_m2_yr']:.1f} gC/m²/yr  →  {flux_label}")
    print(f"   Area     : {result['area_m2'] / 10_000:.1f} ha")

    # ── 6. Tree estimator output ──────────────────────────────────────────────
    trees = result["trees"]
    if not trees["is_sink"]:
        print(f"   Trees needed to offset : {trees['trees']:,}")
        print(f"   Forest equivalent      : {trees['forest_ha']:.0f} ha")
        print(f"   Total emission         : {trees['tonnes_CO2_yr']:,.0f} tonnes CO2/yr")
    else:
        print(f"   Zone absorbs           : {trees['tonnes_CO2_yr']:,.0f} tonnes CO2/yr")
        print(f"   Tree equivalent        : {trees['trees']:,} trees")

    # ── 7. Visualise ──────────────────────────────────────────────────────────
    print("📊 Rendering results …")
    render_results(
        image=image,
        cover=cover,
        result=result,
        lat=lat,
        lon=lon,
        scale_label=preset["label"],
        output_path=args.output,
    )
    print("Done." if args.output else "Done. Close the plot window to exit.")


if __name__ == "__main__":
    main()