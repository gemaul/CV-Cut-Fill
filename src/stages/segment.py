from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rasterize import RasterPage


@dataclass
class SheetSegments:
    """Pixel-space regions for a typical civil sheet layout."""

    drawing_bbox: tuple[int, int, int, int]  # x0,y0,x1,y1
    legend_bbox: tuple[int, int, int, int] | None
    title_block_bbox: tuple[int, int, int, int] | None
    notes: str


def segment_sheet(page: RasterPage) -> SheetSegments:
    """Split drawing viewport vs right-side legend/title block using layout + text."""
    h, w = page.image_bgr.shape[:2]

    # Default: right ~28% is sidebar (notes/legend/title) on C-201-style sheets
    split_x = int(w * 0.72)

    legend_bbox = None
    title_block_bbox = (split_x, int(h * 0.55), w, h)

    legend_hits = [
        s
        for s in page.text_spans
        if "legend" in s["text"].lower() or s["text"].strip().upper() == "MAP LEGEND"
    ]
    if legend_hits:
        # Expand around legend header into a panel on the right
        xs0, ys0, xs1, ys1 = [], [], [], []
        for s in legend_hits:
            x0, y0, x1, y1 = s["bbox_px"]
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        lx0 = int(max(0, min(xs0) - 20))
        ly0 = int(max(0, min(ys0) - 10))
        # Legend entries usually sit below the header within ~35% page height
        lx1 = w
        ly1 = int(min(h, max(ys1) + h * 0.35))
        legend_bbox = (lx0, ly0, lx1, ly1)
        split_x = min(split_x, lx0)

    # Prefer text cues for sheet title block ("GRADING PLAN", sheet no.)
    title_hits = [
        s
        for s in page.text_spans
        if any(
            k in s["text"].upper()
            for k in ("GRADING PLAN", "TITLE", "SHEET", "PROJECT NO")
        )
        and s["bbox_px"][0] > w * 0.55
    ]
    if title_hits:
        ys = [s["bbox_px"][1] for s in title_hits]
        title_block_bbox = (split_x, int(max(0, min(ys) - 40)), w, h)

    drawing_bbox = (0, 0, split_x, h)
    # Clamp / order all boxes into valid pixel rectangles
    drawing_bbox = _clamp_bbox(drawing_bbox, w, h)
    if legend_bbox is not None:
        legend_bbox = _clamp_bbox(legend_bbox, w, h)
    if title_block_bbox is not None:
        title_block_bbox = _clamp_bbox(title_block_bbox, w, h)

    note = (
        f"Drawing viewport x<{drawing_bbox[2]}px; "
        f"legend={'yes' if legend_bbox else 'heuristic'}; "
        f"title block right rail."
    )
    return SheetSegments(
        drawing_bbox=drawing_bbox,
        legend_bbox=legend_bbox,
        title_block_bbox=title_block_bbox,
        notes=note,
    )


def _clamp_bbox(
    bbox: tuple[int, int, int, int], w: int, h: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = int(max(0, min(w, x0)))
    x1 = int(max(0, min(w, x1)))
    y0 = int(max(0, min(h, y0)))
    y1 = int(max(0, min(h, y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    # Avoid zero-area boxes
    if x1 == x0:
        x1 = min(w, x0 + 1)
    if y1 == y0:
        y1 = min(h, y0 + 1)
    return (x0, y0, x1, y1)


def crop_bgr(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image_bgr[y0:y1, x0:x1].copy()


def point_in_bbox(x: float, y: float, bbox: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1
