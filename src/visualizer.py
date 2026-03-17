"""
src/visualizer.py
=================
Render a 4-panel summary figure for a single location analysis.

Panel layout
------------
┌──────────────────┬──────────────────┐
│  Satellite image │  Segmentation    │
│  with lat/lon    │  overlay         │
├──────────────────┼──────────────────┤
│  Land-cover      │  CO₂(t) ODE      │
│  donut chart     │  trajectory      │
└──────────────────┴──────────────────┘
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.imagery import image_bgr_to_rgb
from src.segmentation import build_segmentation_overlay
from src.co2_model import flux_sensitivity_table


# ── Colour palette ────────────────────────────────────────────────────────────
COVER_COLOURS = {
    "vegetation":    "#2d8a4e",
    "water":         "#3a86c8",
    "arid":          "#c8a84b",
    "urban":         "#c0392b",
    "unclassified":  "#aaaaaa",
}


def render_results(
    image: np.ndarray,
    cover: dict,
    result: dict,
    lat: float,
    lon: float,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> None:
    """
    Render and optionally save the 4-panel analysis figure.

    Parameters
    ----------
    image : np.ndarray
        Satellite image (H, W, 3) in BGR order.
    cover : dict
        Land-cover fractions from ``segmentation.segment_image``.
    result : dict
        Output of ``co2_model.estimate_co2_flux``.
    lat, lon : float
        Coordinates displayed in the title.
    output_path : str, optional
        If given, save the figure to this path instead of showing it.
    figsize : tuple
        Matplotlib figure size.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")

    ax_sat, ax_seg, ax_donut, ax_ode = axes.flatten()

    _panel_satellite(ax_sat, image, lat, lon)
    _panel_segmentation(ax_seg, image, cover)
    _panel_donut(ax_donut, cover)
    _panel_ode(ax_ode, result)

    # Overall title
    flux = result["net_flux_gC_m2_yr"]
    verdict = "🌱 Carbon SINK" if result["is_sink"] else "🏭 Carbon SOURCE"
    fig.suptitle(
        f"{verdict}   |   Net flux: {flux:+.1f} gC·m⁻²·yr⁻¹   |   "
        f"({lat:.4f}, {lon:.4f})",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   Figure saved → {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Individual panels
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#16213e")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")


def _panel_satellite(ax: plt.Axes, image_bgr: np.ndarray, lat: float, lon: float) -> None:
    rgb = image_bgr_to_rgb(image_bgr)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_ax(ax, f"Satellite image  ({lat:.4f}°, {lon:.4f}°)")


def _panel_segmentation(ax: plt.Axes, image_bgr: np.ndarray, cover: dict) -> None:
    overlay = build_segmentation_overlay(image_bgr, cover, alpha=0.45)
    ax.imshow(overlay)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_ax(ax, "Land-cover segmentation overlay")

    # Legend
    patches = [
        mpatches.Patch(color=COVER_COLOURS[k], label=k.capitalize())
        for k in COVER_COLOURS
        if k in cover and cover[k] > 0.01
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=7,
        framealpha=0.6,
        facecolor="#1a1a2e",
        labelcolor="white",
    )


def _panel_donut(ax: plt.Axes, cover: dict) -> None:
    labels = [k for k, v in cover.items() if v > 0.005]
    sizes  = [cover[k] for k in labels]
    colours = [COVER_COLOURS.get(k, "#888888") for k in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colours,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": "#1a1a2e", "linewidth": 1.5},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)

    ax.legend(
        wedges,
        [l.capitalize() for l in labels],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
        facecolor="#1a1a2e",
        labelcolor="white",
        framealpha=0.5,
    )
    _style_ax(ax, "Land-cover composition")
    ax.set_facecolor("#16213e")


def _panel_ode(ax: plt.Axes, result: dict) -> None:
    t = result["t"]
    C = result["C"]
    net_flux = result["net_flux_gC_m2_yr"]

    colour = "#2ecc71" if result["is_sink"] else "#e74c3c"
    ax.plot(t, C, color=colour, linewidth=2.5)
    ax.axhline(result["C0_ppm"], color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Time (years)", color="white", fontsize=9)
    ax.set_ylabel("PBL CO₂ concentration (ppm)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.set_facecolor("#16213e")

    # Annotate contributors
    sensitivity = flux_sensitivity_table(result["cover"])
    top = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    annot = "\n".join(
        f"{lbl}: {val:+.0f} gC/m²/yr" for lbl, val in top if abs(val) > 0.5
    )
    ax.annotate(
        annot,
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.5,
        color="#cccccc",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#444466", alpha=0.8),
    )

    _style_ax(
        ax,
        f"CO₂ trajectory over {result['years']:.0f} yr  "
        f"(Δ{result['delta_C_ppm']:+.4f} ppm)",
    )
