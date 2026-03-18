"""
src/visualizer.py  —  CO2 Land-Cover Estimator result figure
Clean, light-mode, infographic-style. No emoji dependencies.
"""
from __future__ import annotations
from typing import Optional
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import numpy as np

from src.imagery import image_bgr_to_rgb
from src.segmentation import build_segmentation_overlay

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#ffffff"
PANEL_BG  = "#fafaf8"
BORDER    = "#e8e6e0"
DARK      = "#1c1c1c"
MUTED     = "#888880"
FAINT     = "#eeede8"

C_GREEN   = "#22a55b"
C_RED     = "#d63c2f"
C_BLUE    = "#2471a3"
C_YELLOW  = "#d4a017"
C_ORANGE  = "#c0620a"
C_TEAL    = "#0e7f6e"
C_PURPLE  = "#6c3483"

COVER_COLOURS = {
    "vegetation":   C_GREEN,
    "water":        C_BLUE,
    "arid":         C_YELLOW,
    "urban":        C_RED,
    "unclassified": MUTED,
}

_FLUX_MIN = -400.0
_FLUX_MAX = 1500.0


def _flux_score(flux: float) -> float:
    return max(0.0, min(1.0, (flux - _FLUX_MIN) / (_FLUX_MAX - _FLUX_MIN)))


def render_results(
    image: np.ndarray,
    cover: dict,
    result: dict,
    lat: float,
    lon: float,
    scale_label: str = "",
    output_path: Optional[str] = None,
    figsize: tuple = (16, 14),
) -> None:
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = GridSpec(
        2, 3, figure=fig,
        height_ratios=[5, 6.5],
        hspace=0.30, wspace=0.20,
        left=0.04, right=0.96, top=0.91, bottom=0.04,
    )

    ax_sat   = fig.add_subplot(gs[0, 0])
    ax_seg   = fig.add_subplot(gs[0, 1])
    ax_pie   = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_therm = fig.add_subplot(gs[1, 1])
    ax_area  = fig.add_subplot(gs[1, 2])

    _panel_satellite(ax_sat, image, lat, lon)
    _panel_segmentation(ax_seg, image, cover)
    _panel_pie(ax_pie, cover)
    _panel_stats(ax_stats, result, cover)
    _panel_thermometer(ax_therm, result)
    _panel_area_comparison(ax_area, result)
    _draw_header(fig, result, lat, lon, scale_label)

    if output_path:
        plt.savefig(output_path, dpi=160, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"   Figure saved -> {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

def _draw_header(fig, result, lat, lon, scale_label):
    is_sink  = result["is_sink"]
    colour   = C_GREEN if is_sink else C_RED
    symbol   = "+" if is_sink else "-"
    verdict  = (
        "This zone absorbs CO2 — it is a carbon sink"
        if is_sink else
        "This zone emits CO2 — it is a carbon source"
    )
    fig.text(0.5, 0.962, verdict,
             ha="center", fontsize=17, fontweight="bold",
             color=colour, transform=fig.transFigure)
    fig.text(0.5, 0.942,
             f"{lat:.4f} deg,  {lon:.4f} deg   |   {scale_label}   |   "
             f"Net flux  {result['net_flux_gC_m2_yr']:+.0f} gC/m2/yr",
             ha="center", fontsize=9.5, color=MUTED,
             transform=fig.transFigure)

    # Accent line under header, colored by verdict
    fig.add_artist(plt.Line2D(
        [0.20, 0.80], [0.933, 0.933],
        transform=fig.transFigure,
        color=colour, linewidth=2.5, alpha=0.35,
    ))
    fig.add_artist(plt.Line2D(
        [0.04, 0.96], [0.933, 0.933],
        transform=fig.transFigure,
        color=BORDER, linewidth=0.8,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Panel style helper
# ─────────────────────────────────────────────────────────────────────────────

def _panel_title(ax, title: str, accent_colour: str = DARK):
    """Draw a panel title with a small left accent bar."""
    ax.set_title("")
    # Accent bar via annotation
    ax.annotate(
        "",
        xy=(0, 1.045), xytext=(0, 1.09),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color=accent_colour, lw=3),
    )
    ax.text(0.018, 1.072, title,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=DARK)


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — Satellite image
# ─────────────────────────────────────────────────────────────────────────────

def _panel_satellite(ax, image_bgr, lat, lon):
    ax.imshow(image_bgr_to_rgb(image_bgr))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(1)
    # Coords chip bottom-left
    ax.text(0.02, 0.03, f"{lat:.4f}  {lon:.4f}",
            transform=ax.transAxes, fontsize=7.5, color="white",
            va="bottom",
            bbox=dict(fc="#000000bb", ec="none", pad=3, boxstyle="round,pad=0.3"))
    _panel_title(ax, "Satellite image", C_BLUE)


# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — Segmented image
# ─────────────────────────────────────────────────────────────────────────────

def _panel_segmentation(ax, image_bgr, cover):
    overlay = build_segmentation_overlay(image_bgr, cover, alpha=0.72)
    ax.imshow(overlay)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(1)

    present = [(k, v) for k, v in cover.items() if v > 0.01]
    patches = [
        mpatches.Patch(
            facecolor=COVER_COLOURS.get(k, MUTED),
            edgecolor="white", linewidth=0.6,
            label=f"{k.capitalize()}  {v:.0%}",
        )
        for k, v in sorted(present, key=lambda x: -x[1])
    ]
    ax.legend(
        handles=patches, loc="lower right", fontsize=7.5,
        framealpha=0.93, facecolor="white",
        labelcolor=DARK, edgecolor=BORDER,
        handlelength=1.0, handletextpad=0.5,
    )
    _panel_title(ax, "Land-cover segmentation", C_ORANGE)


# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — Pie chart
# ─────────────────────────────────────────────────────────────────────────────

def _panel_pie(ax, cover):
    present = [(k, v) for k, v in cover.items() if v > 0.005]
    labels  = [k.capitalize() for k, _ in present]
    sizes   = [v for _, v in present]
    colours = [COVER_COLOURS.get(k, MUTED) for k, _ in present]

    wedges, _, autotexts = ax.pie(
        sizes, colors=colours,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90,
        pctdistance=0.70,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("white")

    # Inline legend inside plot, no extra bbox
    ax.legend(
        wedges, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2, fontsize=8.5,
        facecolor=BG, edgecolor=BORDER,
        framealpha=1.0,
        handlelength=1.0,
    )
    ax.set_facecolor(BG)
    _panel_title(ax, "Cover composition", C_RED)


# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — Stat cards
# ─────────────────────────────────────────────────────────────────────────────

def _panel_stats(ax, result, cover):
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    _panel_title(ax, "Zone summary", DARK)

    trees   = result["trees"]
    is_sink = result["is_sink"]

    cards = [
        {
            "label":  "CO2 per year",
            "value":  f"{trees['tonnes_CO2_yr']:,.0f} t",
            "sub":    f"{'absorbed' if is_sink else 'emitted'} — {trees['area_ha']:.0f} ha zone",
            "colour": C_GREEN if is_sink else C_RED,
        },
        {
            "label":  "Tree equivalence",
            "value":  f"{trees['trees']:,}",
            "sub":    "trees' worth of absorption" if is_sink
                      else "mature trees needed to offset",
            "colour": C_ORANGE,
        },
        {
            "label":  "Forest equivalent",
            "value":  f"{trees['forest_ha']:.0f} ha",
            "sub":    "of dense forest to compensate\n(~400 trees / ha)",
            "colour": C_TEAL,
        },
    ]

    card_h = 0.255
    gaps   = [0.715, 0.385, 0.055]

    for card, y0 in zip(cards, gaps):
        # Shadow effect
        ax.add_patch(FancyBboxPatch(
            (0.055, y0 - 0.008), 0.90, card_h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.015",
            fc="#00000012", ec="none", zorder=1,
        ))
        # Card
        ax.add_patch(FancyBboxPatch(
            (0.04, y0), 0.92, card_h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.015",
            fc="white", ec=card["colour"],
            linewidth=1.8, zorder=2,
        ))
        # Left accent bar
        ax.add_patch(FancyBboxPatch(
            (0.04, y0), 0.018, card_h,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            fc=card["colour"], ec="none", zorder=3,
        ))
        ax.text(0.13, y0 + card_h * 0.80,
                card["label"].upper(),
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=7, color=MUTED,
                fontweight="bold",
                zorder=4)
        ax.text(0.13, y0 + card_h * 0.47,
                card["value"],
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=20, fontweight="black",
                color=card["colour"], zorder=4)
        ax.text(0.13, y0 + card_h * 0.16,
                card["sub"],
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=7.5, color=MUTED,
                linespacing=1.4, zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — Vertical thermometer
# ─────────────────────────────────────────────────────────────────────────────

def _panel_thermometer(ax, result):
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    _panel_title(ax, "Impact on greenhouse effect", DARK)

    n        = 400
    gradient = np.linspace(0, 1, n).reshape(n, 1)
    cmap     = LinearSegmentedColormap.from_list(
        "therm", ["#22a55b", "#f1c40f", "#e67e22", "#d63c2f"])

    bar_x, bar_w = 0.42, 0.22
    bar_y, bar_h = 0.06, 0.84

    # Gradient bar
    ax.imshow(
        gradient[::-1],
        aspect="auto",
        extent=[bar_x, bar_x + bar_w, bar_y, bar_y + bar_h],
        transform=ax.transAxes,
        cmap=cmap, vmin=0, vmax=1, zorder=2,
    )
    # Bar border
    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_y), bar_w, bar_h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.008",
        fc="none", ec=BORDER, linewidth=1.0, zorder=3,
    ))

    # Physics-based ticks
    # score = (flux + 400) / 1900
    ticks = [
        (-400,  "Dense forest",   C_GREEN),
        (   0,  "Carbon neutral", "#8bc34a"),
        ( 600,  "Medium city",    C_ORANGE),
        (1200,  "Dense city",     "#c0390a"),
        (1500,  "Megacity",       C_RED),
    ]
    for flux_val, label, col in ticks:
        score = (flux_val - _FLUX_MIN) / (_FLUX_MAX - _FLUX_MIN)
        y_pos = bar_y + score * bar_h
        # Tick line
        ax.plot(
            [bar_x + bar_w, bar_x + bar_w + 0.055],
            [y_pos, y_pos],
            transform=ax.transAxes,
            color=col, linewidth=1.3, zorder=4,
        )
        ax.text(bar_x + bar_w + 0.07, y_pos,
                f"{label}  ({flux_val:+})",
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=8, color=DARK)

    # ── Needle for this zone ─────────────────────────────────────────────────
    score        = _flux_score(result["net_flux_gC_m2_yr"])
    needle_y     = bar_y + score * bar_h
    needle_col   = cmap(score)

    # Arrow pointing right into bar
    ax.annotate(
        "",
        xy=(bar_x + bar_w * 0.55, needle_y),
        xytext=(bar_x - 0.22, needle_y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=needle_col,
            lw=3.0,
            mutation_scale=16,
        ),
        zorder=6,
    )

    # Zone label pill
    zone_text = (
        "Carbon SINK"   if score < 0.211 else
        "Low emission"  if score < 0.316 else
        "Carbon SOURCE"
    )
    label_col = (
        C_GREEN  if score < 0.211 else
        C_YELLOW if score < 0.316 else
        C_RED
    )
    ax.add_patch(FancyBboxPatch(
        (0.01, needle_y + 0.03), 0.36, 0.085,
        transform=ax.transAxes,
        boxstyle="round,pad=0.01",
        fc=label_col + "22", ec=label_col,
        linewidth=1.5, zorder=5,
    ))
    ax.text(0.19, needle_y + 0.073,
            zone_text,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color=label_col, zorder=6)
    ax.text(0.19, needle_y - 0.035,
            f"{result['net_flux_gC_m2_yr']:+.0f} gC/m2/yr",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8, color=MUTED, zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — Area comparison
# ─────────────────────────────────────────────────────────────────────────────

def _panel_area_comparison(ax, result):
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    _panel_title(ax, "Area comparison", DARK)

    trees     = result["trees"]
    zone_ha   = trees["area_ha"]
    forest_ha = trees["forest_ha"] if not result["is_sink"] else zone_ha * 0.05
    is_sink   = result["is_sink"]

    max_ha    = max(zone_ha, forest_ha)
    max_side  = 0.36   # max square side in axes fraction

    def _side(ha):
        return max_side * math.sqrt(ha / max_ha)

    # ── Zone square (left) ───────────────────────────────────────────────────
    sz    = _side(zone_ha)
    zx    = 0.05
    zy    = 0.50 - sz / 2
    zcol  = C_RED if not is_sink else C_GREEN

    ax.add_patch(FancyBboxPatch(
        (zx, zy), sz, sz,
        transform=ax.transAxes,
        boxstyle="square,pad=0",
        fc=zcol + "28", ec=zcol,
        linewidth=2.2, zorder=2,
    ))
    ax.text(zx + sz/2, zy + sz * 0.62,
            f"{zone_ha:.0f} ha",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=13, fontweight="black", color=zcol, zorder=3)
    ax.text(zx + sz/2, zy + sz * 0.35,
            "analysed\nzone",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8.5, color=zcol, linespacing=1.4, zorder=3)

    # ── VS badge ─────────────────────────────────────────────────────────────
    ax.text(0.50, 0.50, "VS",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=22, fontweight="black",
            color=DARK, alpha=0.18, zorder=3)

    # ── Forest square (right) ────────────────────────────────────────────────
    fs    = _side(forest_ha)
    fx    = 0.95 - fs
    fy    = 0.50 - fs / 2
    fcol  = C_TEAL

    if is_sink:
        f_label_top = f"~{zone_ha * 0.05:.0f} ha"
        f_label_bot = "already\ngreen!"
    else:
        f_label_top = f"{forest_ha:.0f} ha"
        f_label_bot = "forest\nneeded"

    ax.add_patch(FancyBboxPatch(
        (fx, fy), fs, fs,
        transform=ax.transAxes,
        boxstyle="square,pad=0",
        fc=fcol + "28", ec=fcol,
        linewidth=2.2, zorder=2,
    ))
    ax.text(fx + fs/2, fy + fs * 0.62,
            f_label_top,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=13, fontweight="black", color=fcol, zorder=3)
    ax.text(fx + fs/2, fy + fs * 0.35,
            f_label_bot,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8.5, color=fcol, linespacing=1.4, zorder=3)

    # ── Caption ──────────────────────────────────────────────────────────────
    if not is_sink:
        ratio   = forest_ha / zone_ha
        caption = f"To compensate, you'd need {ratio:.1f}x the zone's area\ncovered in dense forest  (~400 trees/ha)"
    else:
        caption = "This zone is already a net CO2 absorber."

    ax.text(0.50, 0.06, caption,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8.5, color=MUTED,
            linespacing=1.5, style="italic")