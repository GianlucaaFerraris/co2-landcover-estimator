"""
app.py  —  CO₂ Land-Cover Estimator · Streamlit Web App
=========================================================
Run with:
    streamlit run app.py

This is a thin UI wrapper around the same pipeline used by main.py.
No changes to the core model are needed.
"""

import io
import random
import tempfile
from pathlib import Path

import streamlit as st
import matplotlib
matplotlib.use("Agg")   # headless — must be set before importing pyplot

from src.geo import random_land_coordinate, is_land
from src.imagery import fetch_satellite_image
from src.segmentation import segment_image
from src.density import get_urban_flux
from src.co2_model import estimate_co2_flux
from src.visualizer import render_results
from config import SCALE_PRESETS, DEFAULT_SCALE, DEFAULT_SIZE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Land-Cover Estimator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9f8f5; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 2.2rem !important; font-weight: 800 !important; }
    .stButton > button {
        width: 100%;
        background-color: #27ae60;
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #1e8449; }
    .stSpinner > div { border-top-color: #27ae60 !important; }
    .result-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-top: 1rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .metric-card {
        flex: 1;
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 CO₂ Land-Cover Estimator")
    st.markdown(
        "Analyse any spot on Earth and find out whether it absorbs or emits CO₂ — "
        "and how many trees would be needed to offset its emissions."
    )
    st.divider()

    location_mode = st.radio(
        "Location input",
        ["🎲 Random land coordinate", "📍 Enter coordinates", "🏙️ Search by city name"],
        index=0,
    )

    lat, lon = None, None

    if location_mode == "📍 Enter coordinates":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                  value=-31.393, format="%.4f")
        with col2:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                  value=-64.195, format="%.4f")

    elif location_mode == "🏙️ Search by city name":
        city = st.text_input("City name", placeholder="e.g. Tokyo, Buenos Aires, Lagos…")
        if city:
            with st.spinner("Looking up coordinates…"):
                try:
                    import requests
                    r = requests.get(
                        "https://nominatim.openstreetmap.org/search",
                        params={"q": city, "format": "json", "limit": 6},
                        headers={"User-Agent": "co2-landcover-estimator/1.0"},
                        timeout=8,
                    )
                    results = r.json()
                except Exception as e:
                    st.error(f"Geocoding failed: {e}")
                    results = []

            if results:
                # Build display labels — show country and type to disambiguate
                options = {
                    f"{res['display_name'][:70]}": (float(res["lat"]), float(res["lon"]))
                    for res in results
                }
                if len(options) == 1:
                    # Only one result — pick it automatically
                    chosen_label = list(options.keys())[0]
                    st.success(f"Found: {chosen_label[:70]}")
                else:
                    chosen_label = st.selectbox(
                        f"Found {len(options)} results — pick one:",
                        options=list(options.keys()),
                    )
                lat, lon = options[chosen_label]
                st.caption(f"Coordinates: ({lat:.4f}, {lon:.4f})")
            elif city:
                st.error("No results found. Try a different name or spelling.")

    st.divider()
    st.markdown("#### Settings")

    scale = st.select_slider(
        "Analysis scale",
        options=list(SCALE_PRESETS.keys()),
        value="barrio",
        format_func=lambda k: SCALE_PRESETS[k]["label"],
    )

    mode = st.selectbox(
        "Segmentation model",
        ["hybrid", "hsv", "ml"],
        index=0,
        help=(
            "hybrid: HSV for water/veg + SegFormer for urban/arid (best)\n"
            "hsv: colour masks only (fast, no GPU)\n"
            "ml: full SegFormer (most accurate, slow on CPU)"
        ),
    )

    st.divider()
    run_btn = st.button("🔍  Analyse this location")

    st.markdown("---")
    st.caption(
        "Data: ESRI World Imagery · WorldPop 2020 · "
        "Model: NEE + SegFormer-B2\n\n"
        "[View on GitHub](https://github.com/yourname/co2-landcover-estimator)"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🌍 CO₂ Land-Cover Estimator")
st.markdown(
    "Point to any location on Earth and instantly find out if that zone is "
    "**absorbing or emitting CO₂**, how many tonnes per year, and how much "
    "forest would be needed to compensate."
)

if not run_btn:
    # Landing state
    st.info(
        "👈  Choose a location in the sidebar and click **Analyse this location** to start.",
        icon="ℹ️",
    )
    st.markdown("#### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🛰 1. Satellite image**\nFetched from ESRI World Imagery — free, no API key.")
    with col2:
        st.markdown("**🔍 2. Segmentation**\nHybrid HSV + SegFormer-B2 classifies vegetation, water, arid and urban pixels.")
    with col3:
        st.markdown("**🧮 3. CO₂ model**\nNEE flux weighted by cover fractions and calibrated urban density.")
    with col4:
        st.markdown("**🌳 4. Tree offset**\nHow many trees would neutralise this zone's emissions?")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
preset      = SCALE_PRESETS[scale]
zoom        = preset["zoom"]
scale_label = preset["label"]

# Resolve coordinates
if location_mode == "🎲 Random land coordinate":
    with st.spinner("Finding a random land coordinate…"):
        lat, lon = random_land_coordinate()

if lat is None or lon is None:
    st.error("No coordinates available. Please enter a location.")
    st.stop()

progress = st.progress(0, text="Starting…")

try:
    progress.progress(10, text="🛰  Fetching satellite image…")
    image = fetch_satellite_image(lat, lon, zoom=zoom, size=DEFAULT_SIZE)

    progress.progress(35, text="🔍  Segmenting land cover…")
    cover = segment_image(image, mode=mode)

    progress.progress(60, text="🏙  Fetching population density…")
    urban_flux = get_urban_flux(lat, lon, verbose=False)

    progress.progress(75, text="🧮  Computing CO₂ flux…")
    result = estimate_co2_flux(
        cover, lat=lat, zoom=zoom, size_px=DEFAULT_SIZE,
        urban_flux=urban_flux,
    )

    progress.progress(90, text="📊  Rendering figure…")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        render_results(
            image=image, cover=cover, result=result,
            lat=lat, lon=lon, scale_label=scale_label,
            output_path=tmp.name,
        )
        img_path = tmp.name

    progress.progress(100, text="Done!")
    progress.empty()

except Exception as e:
    progress.empty()
    st.error(f"Something went wrong: {e}")
    st.exception(e)
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
trees   = result["trees"]
is_sink = result["is_sink"]

# Verdict banner
if is_sink:
    st.success(
        f"✅  **Carbon SINK** — This zone absorbs CO₂.  "
        f"Equivalent to **{trees['trees']:,} trees** absorbing carbon.",
        icon="🌱",
    )
else:
    st.error(
        f"⚠️  **Carbon SOURCE** — This zone emits CO₂.  "
        f"Would need **{trees['trees']:,} trees** to reach neutrality.",
        icon="🏭",
    )

# Quick metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Net flux",       f"{result['net_flux_gC_m2_yr']:+.0f} gC/m²/yr")
m2.metric("CO₂ per year",   f"{trees['tonnes_CO2_yr']:,.0f} t CO₂/yr",
          delta="absorbed" if is_sink else "emitted",
          delta_color="normal" if is_sink else "inverse")
m3.metric("Trees needed",   f"{trees['trees']:,}")
m4.metric("Forest equiv.",  f"{trees['forest_ha']:.0f} ha")

st.divider()

# Full figure
st.image(img_path, use_container_width=True)

# Download button
with open(img_path, "rb") as f:
    img_bytes = f.read()
st.download_button(
    label="⬇️  Download result image",
    data=img_bytes,
    file_name=f"co2_analysis_{lat:.3f}_{lon:.3f}.png",
    mime="image/png",
    use_container_width=True,
)

st.caption(
    f"Analysis: ({lat:.4f}°, {lon:.4f}°) · {scale_label} · "
    f"Mode: {mode} · Urban flux: {result['urban_flux_used']:.0f} gC/m²/yr · "
    f"Source: ESRI World Imagery, WorldPop 2020, SegFormer-B2"
)