"""
src/segmentation.py
===================
Land-cover segmentation from a satellite image.

Three back-ends are available:

``hsv``  (default)
    Fast rule-based classification in the HSV colour space.
    Very accurate for water and vegetation. Less accurate for
    urban vs arid distinction.

``ml``
    Uses SegFormer-B2 (ADE20k). Accurate for urban/arid distinction
    but tends to over-classify urban in satellite imagery (trained on
    ground-level photos). Requires ``transformers`` + ``torch``.

``hybrid``  (recommended)
    Best of both worlds:
    - HSV detects water and vegetation (colour signatures are strong)
    - SegFormer classifies the remaining pixels as urban or arid
    This avoids SegFormer's tendency to label rivers and parks as urban.
"""

from __future__ import annotations
from typing import Dict, Literal

import cv2
import numpy as np

from config import HSV_RANGES, SEGFORMER_MODEL_NAME, SEGFORMER_LABEL_MAP

CoverFractions = Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# HSV back-end
# ─────────────────────────────────────────────────────────────────────────────

def _hsv_masks(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Return a boolean mask per HSV class (no priority — raw masks)."""
    hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks  = {}
    for label, (lower, upper) in HSV_RANGES.items():
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        masks[label] = cv2.inRange(hsv, lo, hi).astype(bool)
    return masks


def _segment_hsv(image_bgr: np.ndarray) -> CoverFractions:
    """
    Classify each pixel with HSV colour-range masks (priority order).

    vegetation → water → arid → urban → unclassified
    """
    total   = image_bgr.shape[0] * image_bgr.shape[1]
    raw     = _hsv_masks(image_bgr)
    claimed = np.zeros(image_bgr.shape[:2], dtype=bool)
    fracs   = {}

    for label in ("vegetation", "water", "arid", "urban"):
        mask        = raw[label] & ~claimed
        fracs[label] = float(mask.sum()) / total
        claimed     |= mask

    return fracs


# ─────────────────────────────────────────────────────────────────────────────
# SegFormer ML back-end
# ─────────────────────────────────────────────────────────────────────────────

def _segformer_label_map(pred_labels: np.ndarray) -> CoverFractions:
    """Convert a SegFormer prediction map to CoverFractions."""
    total  = pred_labels.size
    fracs  = {}
    claimed = np.zeros_like(pred_labels, dtype=bool)

    for category, label_ids in SEGFORMER_LABEL_MAP.items():
        mask          = np.isin(pred_labels, label_ids) & ~claimed
        fracs[category] = float(mask.sum()) / total
        claimed       |= mask

    fracs["unclassified"] = float((~claimed).sum()) / total
    return fracs


def _run_segformer(image_bgr: np.ndarray) -> np.ndarray:
    """Run SegFormer and return a (H, W) prediction label map."""
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch
    except ImportError as exc:
        raise ImportError(
            "The 'ml' / 'hybrid' mode requires transformers and torch.\n"
            "Install with: pip install transformers torch\n"
            "Or use --mode hsv"
        ) from exc

    from PIL import Image as PILImage
    pil_img   = PILImage.fromarray(image_bgr[:, :, ::-1])
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    model     = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME)
    model.eval()

    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    upsampled = torch.nn.functional.interpolate(
        logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1).squeeze(0).numpy()


def _segment_ml(image_bgr: np.ndarray) -> CoverFractions:
    """Full SegFormer segmentation (all 4 classes from ML)."""
    pred = _run_segformer(image_bgr)
    return _segformer_label_map(pred)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid back-end  (HSV water+vegetation  +  SegFormer urban/arid)
# ─────────────────────────────────────────────────────────────────────────────

def _segment_hybrid(image_bgr: np.ndarray) -> CoverFractions:
    """
    Hybrid segmentation combining HSV and SegFormer strengths.

    Step 1 — HSV detects water and vegetation with high confidence.
             These pixels are locked and excluded from SegFormer.
    Step 2 — SegFormer classifies remaining pixels as urban or arid.
             Vegetation/water labels from SegFormer are ignored for
             pixels not already claimed by HSV.

    This prevents SegFormer from mis-labelling rivers and parks as
    urban (a common artefact when using satellite imagery with a
    model trained on ground-level photos).
    """
    total    = image_bgr.shape[0] * image_bgr.shape[1]
    hsv_raw  = _hsv_masks(image_bgr)

    # ── Step 1: HSV claims vegetation and water ───────────────────────────────
    claimed = np.zeros(image_bgr.shape[:2], dtype=bool)
    veg_mask   = hsv_raw["vegetation"] & ~claimed;  claimed |= veg_mask
    water_mask = hsv_raw["water"]      & ~claimed;  claimed |= water_mask

    veg_frac   = float(veg_mask.sum())   / total
    water_frac = float(water_mask.sum()) / total

    # ── Step 2: SegFormer classifies unclaimed pixels ─────────────────────────
    pred    = _run_segformer(image_bgr)
    ml_fracs = _segformer_label_map(pred)

    # Urban and arid fractions from SegFormer apply only to unclaimed pixels
    unclaimed_total = (~claimed).sum()
    if unclaimed_total == 0:
        return {"vegetation": veg_frac, "water": water_frac,
                "urban": 0.0, "arid": 0.0}

    # Re-compute urban / arid restricted to unclaimed mask
    urban_ids = SEGFORMER_LABEL_MAP.get("urban", [])
    arid_ids  = SEGFORMER_LABEL_MAP.get("arid",  [])

    urban_mask_ml = np.isin(pred, urban_ids) & ~claimed
    arid_mask_ml  = np.isin(pred, arid_ids)  & ~claimed & ~urban_mask_ml

    urban_frac = float(urban_mask_ml.sum()) / total
    arid_frac  = float(arid_mask_ml.sum())  / total
    uncls_frac = float((~claimed & ~urban_mask_ml & ~arid_mask_ml).sum()) / total

    return {
        "vegetation":   veg_frac,
        "water":        water_frac,
        "urban":        urban_frac,
        "arid":         arid_frac,
        "unclassified": uncls_frac,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def segment_image(
    image_bgr: np.ndarray,
    mode: Literal["hsv", "ml", "hybrid"] = "hybrid",
) -> CoverFractions:
    """
    Return land-cover fractions for a satellite image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Satellite image as (H, W, 3) BGR uint8 array.
    mode : {"hsv", "ml", "hybrid"}
        - ``"hsv"``    – fast colour-range classification.
        - ``"ml"``     – full SegFormer segmentation.
        - ``"hybrid"`` – HSV water/veg + SegFormer urban/arid (default).

    Returns
    -------
    CoverFractions
        Dict with keys ``"vegetation"``, ``"water"``, ``"arid"``,
        ``"urban"`` and optionally ``"unclassified"``.
    """
    if mode == "hsv":
        cover = _segment_hsv(image_bgr)
    elif mode == "ml":
        cover = _segment_ml(image_bgr)
    elif mode == "hybrid":
        cover = _segment_hybrid(image_bgr)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'hsv', 'ml' or 'hybrid'.")

    for key in ("vegetation", "water", "arid", "urban"):
        cover.setdefault(key, 0.0)

    return cover


def build_segmentation_overlay(
    image_bgr: np.ndarray,
    cover: CoverFractions,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay colour-coded HSV land-cover masks on the original image.

    Returns RGB (H, W, 3) suitable for matplotlib imshow.
    """
    COLOURS_BGR = {
        "vegetation": (34,  139, 34),
        "water":      (200, 120, 30),
        "arid":       (30,  160, 210),
        "urban":      (30,   30, 180),
    }

    hsv     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    overlay = image_bgr.copy()
    claimed = np.zeros(image_bgr.shape[:2], dtype=bool)

    for label, (lower, upper) in HSV_RANGES.items():
        lo   = np.array(lower, dtype=np.uint8)
        hi   = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi).astype(bool) & ~claimed
        claimed |= mask
        overlay[mask] = COLOURS_BGR.get(label, (128, 128, 128))

    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    return blended[:, :, ::-1]   # return RGB