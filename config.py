"""
config.py
=========
Central configuration for the co2-landcover-estimator project.

All physical constants, API settings and colour thresholds live here so
that experiments never require touching source files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
# No API key required for ESRI World Imagery tiles

# ── Satellite imagery defaults ────────────────────────────────────────────────
DEFAULT_ZOOM: int = 12      # 10 = ~40 km across · 14 = ~2.5 km across
DEFAULT_SIZE: int = 512     # pixels per side (Google cap = 640)

# ── CO₂ flux constants (gC · m⁻² · yr⁻¹) ────────────────────────────────────
# Positive  → source (emits CO₂ to atmosphere)
# Negative  → sink   (absorbs CO₂ from atmosphere)
#
# Sources:
#   Chapin et al. (2011) "Principles of Terrestrial Ecosystem Ecology"
#   Sitch et al. (2015) "Recent trends and drivers of regional sources and
#       sinks of carbon dioxide." Biogeosciences 12, 653–679.
#   Churkina et al. (2010) urban CO₂ flux estimates.
CO2_FLUX_BY_COVER = {
    "vegetation": -400.0,   # temperate/tropical forest & dense canopy
    "water":       -20.0,   # lakes, rivers (weak net sink)
    "arid":        +20.0,   # bare soil respiration ≈ minimal GPP
    "urban":      +1500.0,  # combustion + infrastructure (Churkina 2010)
}

# Mixed-pixel fallback for unclassified area
CO2_FLUX_UNCLASSIFIED: float = 0.0

# ── Boundary-layer height ─────────────────────────────────────────────────────
# Effective height (m) of the atmospheric column used to convert gC/m²
# into a ΔC_ppm signal.  Typical planetary boundary layer ≈ 1000 m.
BOUNDARY_LAYER_HEIGHT_M: float = 1000.0

# Molar mass of CO₂ in g/mol
MOLAR_MASS_CO2: float = 44.01
# Molar mass of C in g/mol
MOLAR_MASS_C: float = 12.011
# Dry-air molar mass g/mol
MOLAR_MASS_AIR: float = 28.97
# Dry-air density at STP kg/m³
AIR_DENSITY_KG_M3: float = 1.225

# ── HSV segmentation thresholds ──────────────────────────────────────────────
# Each entry: (lower_HSV, upper_HSV)  — OpenCV uses H:0-179, S:0-255, V:0-255
HSV_RANGES = {
    "vegetation": ([35,  40,  30], [85,  255, 255]),
    "water":      ([90,  50,  30], [130, 255, 255]),
    "arid":       ([15,  20,  80], [34,  200, 230]),
    # Urban is whatever remains after the above masks (low saturation, mid value)
    "urban":      ([0,   0,  100], [180,  35, 215]),
}

# ── SegFormer ML model ────────────────────────────────────────────────────────
SEGFORMER_MODEL_NAME: str = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20k label indices mapped to our four categories
# Full label list: https://huggingface.co/datasets/huggingface/label-files
SEGFORMER_LABEL_MAP = {
    "vegetation": [4, 9, 17, 21, 22, 66, 68, 72, 96, 123, 124, 125, 126, 132, 137],
    "water":      [21, 26, 60, 109, 113, 128],
    "urban":      [0, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 43, 45, 52, 53,
                   54, 55, 56, 57, 58, 59, 63, 64, 65, 67, 76, 80, 84, 87, 91, 92,
                   93, 94, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111,
                   114, 116, 118, 119, 120, 121, 122, 127, 129, 130, 131, 133, 134,
                   135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
    "arid":       [29, 46, 94],   # earth / sand / dirt
}