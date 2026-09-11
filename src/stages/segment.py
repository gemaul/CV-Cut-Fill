from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rasterize import RasterPage


@dataclass
class Region:
    name: str
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1
    color: str  # hex for UI
    role: str  # drawing | chrome | metadata


@dataclass
class SheetSegments:
    """Systematic layout regions for a civil plan sheet."""

    width: int
    height: int
    drawing_bbox: tuple[int, int, int, int]
    legend_bbox: tuple[int, int, int, int] | None = None
    notes_bbox: tuple[int, int, int, int] | None = None
    title_block_bbox: tuple[int, int, int, int] | None = None
    scale_bbox: tuple[int, int, int, int] | None = None
    regions: list[Region] = field(default_factory=list)
    notes: str = ""

    @property
    def chrome_bboxes(self) -> list[tuple[int, int, int, int]]:
        out = []
        for b in (
            self.legend_bbox,
            self.notes_bbox,
            self.title_block_bbox,
            self.scale_bbox,
        ):
            if b is not None:
                out.append(b)
        return out


def segment_sheet(page: RasterPage) -> SheetSegments:
    """Segment sheet into drawing vs chrome using text anchors + right-rail geometry.

    Civil sheets in this corpus are landscape with a dense right information rail
    (notes, legend, title). The plan viewport is the remaining left/center area.
    Contour/elevation perception should only run inside ``drawing_bbox``.
    """
    h, w = page.image_bgr.shape[:2]
    spans = page.text_spans

    # --- 1) Find right information rail from text density ---
    # Use 60th percentile of text x-centers in the right half as a soft split,
    # then snap to the leftmost "chrome" header (MAP LEGEND / NOTES / SHEET TITLE).
    right_spans = [
        s
        for s in spans
        if (s["bbox_px"][0] + s["bbox_px"][2]) / 2 > w * 0.55
    ]
    split_x = int(w * 0.70)
    if right_spans:
        xs = sorted((s["bbox_px"][0] for s in right_spans))
        # Left edge of the densest right-rail content
        split_x = int(max(w * 0.58, min(xs[max(0, len(xs) // 8)], w * 0.82)))

    chrome_headers = {
        "legend": ("MAP LEGEND", "LEGEND"),
        "notes": ("GENERAL NOTES", "DEMOLITION KEYNOTES", "GRADING KEYNOTES", "NOTES"),
        "title": ("SHEET TITLE", "GRADING PLAN", "EXISTING CONDITIONS", "SITE PLAN"),
        "scale": ("SCALE", '1"', "GRAPHIC SCALE"),
    }

    def find_header(keys: tuple[str, ...]) -> dict | None:
        hits = []
        for s in spans:
            t = s["text"].strip().upper()
            if any(k in t for k in keys):
                hits.append(s)
        if not hits:
            return None
        # Prefer hits on the right rail
        hits.sort(
            key=lambda s: (
                0 if s["bbox_px"][0] > w * 0.5 else 1,
                s["bbox_px"][1],
            )
        )
        return hits[0]

    legend_header = find_header(chrome_headers["legend"])
    notes_header = find_header(chrome_headers["notes"])
    title_header = find_header(chrome_headers["title"])
    scale_header = find_header(chrome_headers["scale"])

    # Snap split to leftmost chrome header when present
    header_xs = []
    for hdr in (legend_header, notes_header, title_header):
        if hdr is not None and hdr["bbox_px"][0] > w * 0.45:
            header_xs.append(hdr["bbox_px"][0])
    if header_xs:
        split_x = int(min(split_x, min(header_xs) - 12))

    drawing_bbox = _clamp_bbox((0, 0, split_x, h), w, h)

    # --- 2) Build chrome regions as vertical stacks in the right rail ---
    legend_bbox = None
    notes_bbox = None
    title_block_bbox = None
    scale_bbox = None

    if legend_header is not None:
        y0 = int(legend_header["bbox_px"][1] - 8)
        # Legend typically occupies a mid band; stop before title block if known
        y1 = int(min(h, legend_header["bbox_px"][3] + h * 0.32))
        if title_header is not None and title_header["bbox_px"][1] > y0:
            y1 = int(min(y1, title_header["bbox_px"][1] - 8))
        legend_bbox = _clamp_bbox((split_x, y0, w, y1), w, h)

    if notes_header is not None:
        y0 = int(notes_header["bbox_px"][1] - 8)
        y1 = int(min(h, notes_header["bbox_px"][3] + h * 0.28))
        if legend_bbox is not None and legend_bbox[1] > y0:
            y1 = min(y1, legend_bbox[1] - 4)
        notes_bbox = _clamp_bbox((split_x, y0, w, max(y0 + 40, y1)), w, h)

    if title_header is not None:
        y0 = int(max(h * 0.45, title_header["bbox_px"][1] - 30))
        title_block_bbox = _clamp_bbox((split_x, y0, w, h), w, h)
    else:
        title_block_bbox = _clamp_bbox((split_x, int(h * 0.62), w, h), w, h)

    if scale_header is not None:
        x0, y0, x1, y1 = scale_header["bbox_px"]
        scale_bbox = _clamp_bbox(
            (int(x0 - 40), int(y0 - 20), int(min(w, x1 + 220)), int(y1 + 40)),
            w,
            h,
        )

    # If notes/legend overlap drawing due to bad headers, force them into right rail
    if legend_bbox is not None and legend_bbox[0] < split_x:
        legend_bbox = _clamp_bbox((split_x, legend_bbox[1], w, legend_bbox[3]), w, h)
    if notes_bbox is not None and notes_bbox[0] < split_x:
        notes_bbox = _clamp_bbox((split_x, notes_bbox[1], w, notes_bbox[3]), w, h)

    regions: list[Region] = [
        Region("drawing", drawing_bbox, "#3B82F6", "drawing"),
    ]
    if notes_bbox is not None:
        regions.append(Region("notes", notes_bbox, "#F59E0B", "chrome"))
    if legend_bbox is not None:
        regions.append(Region("legend", legend_bbox, "#A855F7", "chrome"))
    if title_block_bbox is not None:
        regions.append(Region("title_block", title_block_bbox, "#EAB308", "chrome"))
    if scale_bbox is not None:
        regions.append(Region("scale", scale_bbox, "#10B981", "metadata"))

    note = (
        f"Drawing {drawing_bbox[2] - drawing_bbox[0]}×{drawing_bbox[3] - drawing_bbox[1]}px; "
        f"right rail @ x={split_x}; regions={[r.name for r in regions]}"
    )
    return SheetSegments(
        width=w,
        height=h,
        drawing_bbox=drawing_bbox,
        legend_bbox=legend_bbox,
        notes_bbox=notes_bbox,
        title_block_bbox=title_block_bbox,
        scale_bbox=scale_bbox,
        regions=regions,
        notes=note,
    )


def drawing_mask(segments: SheetSegments) -> np.ndarray:
    """Boolean mask (H,W) True where perception is allowed."""
    mask = np.zeros((segments.height, segments.width), dtype=bool)
    x0, y0, x1, y1 = segments.drawing_bbox
    mask[y0:y1, x0:x1] = True
    # Punch out any chrome that leaked into drawing
    for bbox in segments.chrome_bboxes:
        cx0, cy0, cx1, cy1 = bbox
        # only punch if overlapping drawing
        ox0, oy0 = max(x0, cx0), max(y0, cy0)
        ox1, oy1 = min(x1, cx1), min(y1, cy1)
        if ox1 > ox0 and oy1 > oy0:
            mask[oy0:oy1, ox0:ox1] = False
    return mask


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
