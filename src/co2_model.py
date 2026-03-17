"""
src/co2_model.py
================
CO₂ flux estimation using a Net Ecosystem Exchange (NEE) approach.

Physical model
--------------
We model the rate of change of CO₂ concentration inside the planetary
boundary layer (PBL) above the analysed area as:

    dC/dt = F_total / (ρ_air · h · MW_ratio)              … (1)

where

    F_total [gC · m⁻² · yr⁻¹]  = Σᵢ  fᵢ · Fᵢ
    fᵢ                           = fractional cover of class i
    Fᵢ                           = literature flux for class i (config.py)
    ρ_air  [g · m⁻³]            = 1225 g/m³
    h      [m]                   = boundary-layer height (default 1000 m)
    MW_ratio                     = (M_CO₂ / M_C) / M_air_per_mol

The ODE is solved with ``scipy.integrate.solve_ivp`` (RK45).

The output ΔC_ppm is the change in column-average CO₂ concentration (ppm)
over the chosen time horizon.  This is a **local, zonal** estimate; it does
*not* account for horizontal mixing with the background atmosphere.

References
----------
Chapin F.S. III et al. (2011) "Principles of Terrestrial Ecosystem Ecology".
Sitch S. et al. (2015) Biogeosciences 12, 653–679.
Churkina G. et al. (2010) Global Change Biology 16, 2296–2309.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp

from config import (
    CO2_FLUX_BY_COVER,
    CO2_FLUX_UNCLASSIFIED,
    BOUNDARY_LAYER_HEIGHT_M,
    MOLAR_MASS_CO2,
    MOLAR_MASS_C,
    MOLAR_MASS_AIR,
    AIR_DENSITY_KG_M3,
)

CoverFractions = Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _net_flux_gC_m2_yr(cover: CoverFractions) -> float:
    """
    Weighted sum of per-cover-type CO₂ fluxes.

    Parameters
    ----------
    cover : CoverFractions
        Output of ``segmentation.segment_image``.

    Returns
    -------
    float
        Net flux in gC · m⁻² · yr⁻¹.
        Negative  → zone is a carbon **sink**.
        Positive  → zone is a carbon **source**.
    """
    flux = 0.0
    for label, fraction in cover.items():
        f_unit = CO2_FLUX_BY_COVER.get(label, CO2_FLUX_UNCLASSIFIED)
        flux += fraction * f_unit
    return flux


def _build_ode(net_flux_gC_m2_yr: float) -> callable:
    """
    Return the right-hand side function for the CO₂ ODE.

    The ODE is:
        dC/dt = net_flux_gC_m2_yr
                / (rho_air_g_m3 · h · (M_C / M_CO2) · (1 / M_air) · 1e6)

    Simplified to a unit conversion constant × flux.

    The factor converts gC/m²/yr → ppm_CO₂/yr in the PBL column.
    """
    # Air mass per m² of PBL column [g/m²]
    rho_g_m3 = AIR_DENSITY_KG_M3 * 1000.0          # kg/m³ → g/m³
    air_column_g_m2 = rho_g_m3 * BOUNDARY_LAYER_HEIGHT_M

    # gC → gCO₂
    flux_gCO2 = net_flux_gC_m2_yr * (MOLAR_MASS_CO2 / MOLAR_MASS_C)

    # gCO₂/m² → mol_CO₂/m²
    flux_mol_CO2 = flux_gCO2 / MOLAR_MASS_CO2

    # mol_air per m² of column
    mol_air_col = air_column_g_m2 / MOLAR_MASS_AIR

    # ppm change per year (mol fraction × 1e6)
    dppm_per_year = (flux_mol_CO2 / mol_air_col) * 1e6

    def rhs(t: float, y: list[float]) -> list[float]:  # noqa: ANN001
        return [dppm_per_year]

    return rhs, dppm_per_year


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def estimate_co2_flux(
    cover: CoverFractions,
    years: float = 1.0,
    C0_ppm: float = 420.0,
) -> dict:
    """
    Solve the CO₂ flux ODE and return a summary dictionary.

    Parameters
    ----------
    cover : CoverFractions
        Land-cover fractions from ``segmentation.segment_image``.
    years : float
        Time horizon for the ODE integration (default: 1 year).
    C0_ppm : float
        Initial atmospheric CO₂ concentration in ppm (default: 420 ppm,
        approximate global average for 2024).

    Returns
    -------
    dict with keys:
        ``net_flux_gC_m2_yr``  – Net flux (gC/m²/yr, neg = sink).
        ``delta_C_ppm``        – Change in PBL CO₂ over the time horizon.
        ``C_final_ppm``        – Final CO₂ concentration in PBL (ppm).
        ``t``                  – Time array (years).
        ``C``                  – CO₂ concentration array (ppm) over time.
        ``is_sink``            – Boolean, True if zone absorbs CO₂.
        ``cover``              – Echo of the input cover fractions.

    Examples
    --------
    >>> cover = {"vegetation": 0.7, "water": 0.1, "arid": 0.1, "urban": 0.1}
    >>> result = estimate_co2_flux(cover)
    >>> result["is_sink"]
    True
    >>> result["net_flux_gC_m2_yr"] < 0
    True
    """
    net_flux = _net_flux_gC_m2_yr(cover)
    rhs, dppm_per_year = _build_ode(net_flux)

    t_span = (0.0, years)
    t_eval = np.linspace(0.0, years, max(200, int(years * 200)))

    sol = solve_ivp(
        rhs,
        t_span,
        [C0_ppm],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
    )

    C_array = sol.y[0]
    delta_C = float(C_array[-1] - C_array[0])

    return {
        "net_flux_gC_m2_yr": net_flux,
        "delta_C_ppm": delta_C,
        "dppm_per_year": dppm_per_year,
        "C_final_ppm": float(C_array[-1]),
        "C0_ppm": C0_ppm,
        "t": sol.t,
        "C": C_array,
        "is_sink": net_flux < 0,
        "cover": cover,
        "years": years,
    }


def flux_sensitivity_table(cover: CoverFractions) -> Dict[str, float]:
    """
    Show the contribution (gC/m²/yr) of each land-cover class independently.

    Useful for understanding which class dominates the signal.

    Parameters
    ----------
    cover : CoverFractions

    Returns
    -------
    Dict mapping label → individual flux contribution.

    Examples
    --------
    >>> cover = {"vegetation": 0.5, "urban": 0.5}
    >>> table = flux_sensitivity_table(cover)
    >>> table["vegetation"]
    -200.0
    >>> table["urban"]
    750.0
    """
    return {
        label: fraction * CO2_FLUX_BY_COVER.get(label, CO2_FLUX_UNCLASSIFIED)
        for label, fraction in cover.items()
    }
