"""
src/segmentation.py
===================
Land-cover segmentation from a satellite image.

Two back-ends are available:

``hsv``  (default)
    Fast rule-based classification in the HSV colour space.
    No extra model downloads.  Works offline.

``ml``
    Uses NVIDIA's SegFormer-B2 fine-tuned on ADE20k via HuggingFace
    Transformers.  More accurate, especially in ambiguous areas.
    Requires ``transformers`` and ``torch`` (or ``onnxruntime``) to be
    installed (see requirements.txt).

Both back-ends return a ``CoverFractions`` dict with keys
``vegetation``, ``water``, ``arid``, ``urban`` whose values sum to ≤ 1.
The remainder is labelled ``unclassified``.
"""

from __future__ import annotations

from typing import Dict, Literal

import cv2
import numpy as np

from config import HSV_RANGES, SEGFORMER_MODEL_NAME, SEGFORMER_LABEL_MAP

# Type alias for the fractions dict
CoverFractions = Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# HSV back-end
# ─────────────────────────────────────────────────────────────────────────────

def _segment_hsv(image_bgr: np.ndarray) -> CoverFractions:
    """
    Classify each pixel with HSV colour-range masks.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image (H, W, 3) in BGR colour space.

    Returns
    -------
    CoverFractions
        Fraction of image area belonging to each land-cover class.

    Notes
    -----
    Masks are applied in priority order: vegetation → water → arid → urban.
    Residual unclassified pixels get fraction ``1 - sum(fractions)``.

    The thresholds in ``config.HSV_RANGES`` are conservative by design.
    Twilight/shadow pixels are left unclassified rather than mis-labelled.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    fractions: CoverFractions = {}
    claimed = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=bool)

    for label, (lower, upper) in HSV_RANGES.items():
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi).astype(bool)

        # Exclude pixels already assigned to a higher-priority class
        new_mask = mask & ~claimed
        fractions[label] = float(new_mask.sum()) / total_pixels
        claimed |= new_mask

    return fractions


# ─────────────────────────────────────────────────────────────────────────────
# SegFormer ML back-end
# ─────────────────────────────────────────────────────────────────────────────

def _segment_ml(image_bgr: np.ndarray) -> CoverFractions:
    """
    Classify pixels using SegFormer-B2 (ADE20k fine-tune).

    ADE20k's 150 classes are aggregated into our four land-cover types
    using the mapping defined in ``config.SEGFORMER_LABEL_MAP``.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image (H, W, 3) in BGR colour space.

    Returns
    -------
    CoverFractions
        Fraction of image area per land-cover class.

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` are not installed.
    """
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch
    except ImportError as exc:
        raise ImportError(
            "The 'ml' segmentation mode requires the 'transformers' and 'torch' "
            "packages.  Install them with:\n"
            "    pip install transformers torch\n"
            "Or switch to the default HSV mode: --mode hsv"
        ) from exc

    # Convert BGR → RGB for HuggingFace processor
    image_rgb = image_bgr[:, :, ::-1]
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image_rgb)

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME)
    model.eval()

    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_labels, H/4, W/4)

    # Upsample to original resolution
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=pil_img.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    pred_labels = upsampled.argmax(dim=1).squeeze(0).numpy()  # (H, W)

    total_pixels = pred_labels.size
    fractions: CoverFractions = {}

    for category, label_ids in SEGFORMER_LABEL_MAP.items():
        mask = np.isin(pred_labels, label_ids)
        fractions[category] = float(mask.sum()) / total_pixels

    # Any label not covered by the map → distribute as unclassified
    all_mapped = set(
        lid for ids in SEGFORMER_LABEL_MAP.values() for lid in ids
    )
    unclassified_mask = ~np.isin(pred_labels, list(all_mapped))
    fractions["unclassified"] = float(unclassified_mask.sum()) / total_pixels

    return fractions


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def segment_image(
    image_bgr: np.ndarray,
    mode: Literal["hsv", "ml"] = "hsv",
) -> CoverFractions:
    """
    Return land-cover fractions for a satellite image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Satellite image as (H, W, 3) BGR uint8 array.
    mode : {"hsv", "ml"}
        Segmentation back-end to use.
        - ``"hsv"``  – fast colour-range classification (default).
        - ``"ml"``   – SegFormer semantic segmentation (requires torch).

    Returns
    -------
    CoverFractions
        Dictionary with keys ``"vegetation"``, ``"water"``, ``"arid"``,
        ``"urban"`` and optionally ``"unclassified"``.
        Values are floats in [0, 1] representing pixel fractions.

    Examples
    --------
    >>> import numpy as np
    >>> green_img = np.zeros((64, 64, 3), dtype=np.uint8)
    >>> green_img[:, :] = (40, 180, 80)   # BGR greenish
    >>> cover = segment_image(green_img, mode="hsv")
    >>> cover["vegetation"] > 0.5
    True
    """
    if mode == "hsv":
        cover = _segment_hsv(image_bgr)
    elif mode == "ml":
        cover = _segment_ml(image_bgr)
    else:
        raise ValueError(f"Unknown segmentation mode: {mode!r}. Use 'hsv' or 'ml'.")

    # Ensure all expected keys are present (fill missing with 0.0)
    for key in ("vegetation", "water", "arid", "urban"):
        cover.setdefault(key, 0.0)

    return cover


def build_segmentation_overlay(
    image_bgr: np.ndarray,
    cover: CoverFractions,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay colour-coded land-cover masks on the original image (HSV only).

    Parameters
    ----------
    image_bgr : np.ndarray
        Original satellite image.
    cover : CoverFractions
        Output of ``segment_image``.
    alpha : float
        Transparency of the overlay (0 = fully transparent, 1 = opaque).

    Returns
    -------
    np.ndarray
        Blended RGB image (H, W, 3) suitable for matplotlib imshow.
    """
    OVERLAY_COLOURS_BGR = {
        "vegetation": (34, 139, 34),    # forest green
        "water":      (255, 165, 0),    # orange (avoid blue clash with water)
        "arid":       (0, 165, 255),    # amber
        "urban":      (0, 0, 200),      # red
    }

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    overlay = image_bgr.copy()
    claimed = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)

    for label, (lower, upper) in HSV_RANGES.items():
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi).astype(bool) & ~claimed
        claimed |= mask
        colour = OVERLAY_COLOURS_BGR.get(label, (128, 128, 128))
        overlay[mask] = colour

    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    return blended[:, :, ::-1]  # return RGB
