"""
src/co2_model.py  —  NEE-based CO₂ flux model
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import numpy as np

from config import (
    CO2_FLUX_BY_COVER, CO2_FLUX_UNCLASSIFIED,
    MOLAR_MASS_CO2, MOLAR_MASS_C,
    TREE_CO2_KG_PER_YEAR, TREE_CANOPY_M2,
)

CoverFractions = Dict[str, float]
_ATM_CARBON_MASS_GC = 8.60e17   # IPCC AR6 — total atmospheric carbon [gC]


def image_area_m2(lat: float, zoom: int, size_px: int = 512) -> float:
    res = 156_543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    return (res * size_px) ** 2


def _build_flux_table(cover: CoverFractions, urban_flux: float) -> Dict[str, float]:
    t = dict(CO2_FLUX_BY_COVER)
    t["urban"] = urban_flux
    return t


def _net_flux(cover: CoverFractions, flux_table: Dict[str, float]) -> float:
    return sum(f * flux_table.get(l, CO2_FLUX_UNCLASSIFIED) for l, f in cover.items())


def global_ppm_contribution(net_flux_gC_m2_yr: float, area_m2: float) -> float:
    """Zone's honest contribution to global atmospheric CO₂ (ppm/yr)."""
    return (net_flux_gC_m2_yr * area_m2 / _ATM_CARBON_MASS_GC) * 1e6


def estimate_trees(net_flux_gC_m2_yr: float, area_m2: float) -> dict:
    """
    Tree equivalence.
    - Source: trees needed to offset total emission (anywhere, not per ha).
    - Sink:   equivalent trees worth of absorption already happening.
    """
    total_gC_yr    = net_flux_gC_m2_yr * area_m2
    total_kgCO2_yr = abs(total_gC_yr * (MOLAR_MASS_CO2 / MOLAR_MASS_C) / 1000.0)
    area_ha        = area_m2 / 10_000.0
    is_sink        = net_flux_gC_m2_yr < 0
    trees          = math.ceil(total_kgCO2_yr / TREE_CO2_KG_PER_YEAR)
    # Equivalent forest area (at 400 trees/ha dense urban forest)
    forest_ha      = trees / 400.0

    return {
        "is_sink":        is_sink,
        "kgCO2_yr":       total_kgCO2_yr,
        "tonnes_CO2_yr":  total_kgCO2_yr / 1000.0,
        "trees":          trees,
        "forest_ha":      forest_ha,   # ha of forest needed, NOT trees/ha of the zone
        "area_ha":        area_ha,
    }


def estimate_co2_flux(
    cover: CoverFractions,
    lat: float = 0.0,
    zoom: int = 15,
    size_px: int = 512,
    urban_flux: Optional[float] = None,
    years: float = 10.0,
    C0_ppm: float = 420.0,
) -> dict:
    u_flux     = urban_flux if urban_flux is not None else CO2_FLUX_BY_COVER["urban"]
    flux_table = _build_flux_table(cover, u_flux)
    net_flux   = _net_flux(cover, flux_table)
    area       = image_area_m2(lat, zoom, size_px)
    tree_data  = estimate_trees(net_flux, area)
    gppm       = global_ppm_contribution(net_flux, area)

    return {
        "net_flux_gC_m2_yr": net_flux,
        "is_sink":           net_flux < 0,
        "area_m2":           area,
        "trees":             tree_data,
        "global_ppm_yr":     gppm,
        "urban_flux_used":   u_flux,
        "flux_table":        flux_table,
        "cover":             cover,
        "years":             years,
        "C0_ppm":            C0_ppm,
        "delta_C_ppm":       0.0,   # legacy
    }


def flux_sensitivity_table(
    cover: CoverFractions,
    flux_table: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    table = flux_table or CO2_FLUX_BY_COVER
    return {l: f * table.get(l, CO2_FLUX_UNCLASSIFIED) for l, f in cover.items()}