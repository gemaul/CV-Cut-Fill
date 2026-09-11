from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .rasterize import RasterPage

# Labels that sit in the MAP LEGEND with a graphic swatch to their left.
_LEGEND_ENTRY_LABELS = (
    "BUILDING LINE",
    "PROPERTY LINE",
    "EXISTING CONTOUR",
    "PROPOSED CONTOUR",
    "TOP OF CURB",
    "MATCH EXISTING",
    "DRAINAGE FLOW",
    "SLOPE GRADE",
    "CONCRETE SIDEWALK",
    "HEAVY-DUTY CONCRETE",
    "HEAVY DUTY CONCRETE",
    "PROPOSED BUILDING",
    "CURB CUT",
    "RIP RAP",
    "SPOT ELEVATION",
)


@dataclass
class Region:
    name: str
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1
    color: str  # hex for UI
    role: str  # drawing | chrome | metadata | site | topo_zone


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
    """Segment sheet into drawing vs chrome, with legend graphics included.

    Order of operations (legend-first):
      1. Locate MAP LEGEND and its entry labels + symbol column
      2. Locate notes / title / scale chrome
      3. Set the drawing/right-rail split from the *left* edge of chrome
         (so legend swatches are not cut into the drawing viewport)
    """
    h, w = page.image_bgr.shape[:2]
    spans = page.text_spans
    gray = cv2.cvtColor(page.image_bgr, cv2.COLOR_BGR2GRAY)

    legend_header = _find_header(
        spans, w, ("MAP LEGEND",), prefer_right=True
    )
    # Fallback if the sheet only says LEGEND (avoid matching random words)
    if legend_header is None:
        legend_header = _find_header(spans, w, ("MAP LEGEND", "LEGEND"), prefer_right=True)

    notes_header = _find_header(
        spans,
        w,
        ("GENERAL NOTES", "DEMOLITION KEYNOTES", "GRADING KEYNOTES"),
        prefer_right=True,
    )
    title_header = _find_header(
        spans,
        w,
        ("SHEET TITLE", "GRADING PLAN", "EXISTING CONDITIONS", "SITE PLAN"),
        prefer_right=True,
    )
    scale_header = _find_header(
        spans, w, ("SCALE", '1"', "GRAPHIC SCALE"), prefer_right=True
    )

    # --- 1) Legend panel (symbols + labels) ---
    legend_bbox = _detect_legend_panel(
        page, gray, legend_header, spans, title_header=title_header
    )

    # --- 2) Right-rail split from chrome left edges (legend symbols included) ---
    split_candidates: list[int] = []
    if legend_bbox is not None:
        split_candidates.append(legend_bbox[0])
    for hdr in (notes_header, title_header, legend_header):
        if hdr is not None and hdr["bbox_px"][0] > w * 0.55:
            # Leave room for legend swatches left of the header text
            split_candidates.append(int(hdr["bbox_px"][0] - 200))
    if not split_candidates:
        split_x = _estimate_right_rail_split(spans, w, headers=[])
    else:
        # Prefer the rightmost-valid chrome edge so V-101 doesn't collapse to 50%
        # when a single false legend hit pulls left. Cap like C-201 (~0.76w).
        split_x = int(max(w * 0.62, min(min(split_candidates), w * 0.82)))

    drawing_bbox = _clamp_bbox((0, 0, split_x, h), w, h)

    # --- 3) Stack remaining chrome panels to the right of split ---
    panel_headers: list[tuple[str, dict]] = []
    if notes_header is not None:
        panel_headers.append(("notes", notes_header))
    if legend_header is not None:
        panel_headers.append(("legend", legend_header))
    if title_header is not None:
        panel_headers.append(("title", title_header))
    panel_headers.sort(key=lambda item: item[1]["bbox_px"][1])

    notes_bbox = None
    title_block_bbox = None

    for i, (name, hdr) in enumerate(panel_headers):
        if name == "legend":
            continue  # already refined
        y0 = int(max(0, hdr["bbox_px"][1] - 10))
        if i + 1 < len(panel_headers):
            y1 = int(panel_headers[i + 1][1]["bbox_px"][1] - 6)
        else:
            y1 = h
        # Don't let notes bleed into the refined legend
        if name == "notes" and legend_bbox is not None and legend_bbox[1] > y0:
            y1 = min(y1, legend_bbox[1] - 4)
        y1 = max(y1, y0 + 48)
        bbox = _clamp_bbox((split_x, y0, w, y1), w, h)
        if name == "notes":
            notes_bbox = bbox
        elif name == "title":
            title_block_bbox = bbox

    if title_block_bbox is None:
        title_block_bbox = _clamp_bbox((split_x, int(h * 0.62), w, h), w, h)

    # Snap legend x0 to the rail split (symbols included) and clip bottom before title
    if legend_bbox is not None:
        lx0, ly0, lx1, ly1 = legend_bbox
        if title_block_bbox is not None:
            ly1 = min(ly1, title_block_bbox[1] - 4)
        legend_bbox = _clamp_bbox((min(lx0, split_x), ly0, w, max(ly1, ly0 + 48)), w, h)

    scale_bbox = None
    if scale_header is not None:
        sx0, sy0, sx1, sy1 = scale_header["bbox_px"]
        scale_bbox = _clamp_bbox(
            (
                int(max(split_x, sx0 - 40)),
                int(sy0 - 20),
                int(min(w, sx1 + 220)),
                int(sy1 + 40),
            ),
            w,
            h,
        )

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
        f"right rail @ x={split_x}; "
        f"legend={'full' if legend_bbox else 'miss'}; "
        f"regions={[r.name for r in regions]}"
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


def _detect_legend_panel(
    page: RasterPage,
    gray: np.ndarray,
    legend_header: dict | None,
    spans: list[dict],
    *,
    title_header: dict | None,
) -> tuple[int, int, int, int] | None:
    """Return bbox covering MAP LEGEND title + symbol column + label text."""
    h, w = gray.shape
    if legend_header is None:
        return None

    hx0, hy0, hx1, hy1 = legend_header["bbox_px"]
    # Entry labels must sit in the same right-rail column as MAP LEGEND.
    # Do NOT accept drawing callouts (e.g. "5' CONCRETE SIDEWALK" on V-101)
    # that share legend keywords but live far left in the plan viewport.
    rail_min_x = max(w * 0.58, hx0 - 120)
    entries = []
    for s in spans:
        t = re.sub(r"\s+", " ", s["text"].strip().upper())
        if s["bbox_px"][1] < hy1 - 2:
            continue
        if s["bbox_px"][0] < rail_min_x:
            continue
        if any(label in t for label in _LEGEND_ENTRY_LABELS):
            entries.append(s)

    # Survey-style legends (V-101) use different labels — gather any text in the
    # header column so the panel still sizes like C-201's grading legend.
    column_spans = []
    for s in spans:
        if s["bbox_px"][1] < hy1 - 2:
            continue
        if s["bbox_px"][0] < rail_min_x:
            continue
        if s["bbox_px"][0] > hx1 + 280:
            continue
        if s["bbox_px"][1] > hy1 + int(h * 0.45):
            continue
        column_spans.append(s)

    if entries:
        label_left = min(s["bbox_px"][0] for s in entries)
        label_right = max(s["bbox_px"][2] for s in entries)
        y_top = min(hy0 - 8, min(s["bbox_px"][1] for s in entries) - 8)
        y_bot = max(s["bbox_px"][3] for s in entries) + 40
    elif column_spans:
        label_left = min(s["bbox_px"][0] for s in column_spans)
        label_right = max(s["bbox_px"][2] for s in column_spans)
        y_top = hy0 - 8
        y_bot = max(s["bbox_px"][3] for s in column_spans) + 28
    else:
        label_left = hx0
        label_right = max(hx1, hx0 + 200)
        y_top = hy0 - 8
        y_bot = hy1 + int(h * 0.28)

    # Prefer the denser of keyed entries vs column text for vertical extent
    if column_spans and entries:
        y_bot = max(y_bot, max(s["bbox_px"][3] for s in column_spans) + 28)
        label_left = min(label_left, min(s["bbox_px"][0] for s in column_spans))

    # Stop before sheet title / curb detail when those headers sit below the legend
    stoppers = []
    for s in spans:
        t = s["text"].strip().upper()
        if s["bbox_px"][1] <= hy1:
            continue
        if s["bbox_px"][0] < w * 0.5:
            continue
        if any(
            k in t
            for k in (
                "CURB & GUTTER",
                "CURB AND GUTTER",
                "SHEET TITLE",
                "GRADING PLAN",
                "EXISTING CONDITIONS",
                "REVISION",
                "DEMOLITION KEYNOTES",
                "GENERAL NOTES",
                "GRADING KEYNOTES",
            )
        ):
            # Only stop if clearly below legend body
            if s["bbox_px"][1] > hy1 + 80:
                stoppers.append(int(s["bbox_px"][1] - 10))
    if title_header is not None and title_header["bbox_px"][1] > hy1 + 80:
        stoppers.append(int(title_header["bbox_px"][1] - 10))
    if stoppers:
        y_bot = min(y_bot, min(stoppers))

    # Symbol column: dark ink immediately left of the label column
    # Cap search so plan ink cannot drag the rail into the drawing (V-101 bug).
    max_symbol_reach = int(min(280, max(120, hx0 - w * 0.68)))
    symbol_left = _legend_symbol_left_edge(
        gray,
        label_left=label_left,
        y0=int(max(0, y_top)),
        y1=int(min(h, y_bot)),
        search_px=int(max(120, min(max_symbol_reach, w * 0.07))),
    )
    # Never let symbols pull left of a C-201-like rail (~header − swatch width)
    symbol_left = int(max(symbol_left, hx0 - 220, w * 0.62))

    # Include the printed legend panel border just left of the swatches
    frame_left = _legend_frame_left(
        gray,
        symbol_left=symbol_left,
        y0=int(max(0, y_top)),
        y1=int(min(h, y_bot)),
        search_px=50,
    )
    frame_left = int(max(frame_left, w * 0.60, hx0 - 260))

    x0 = int(min(frame_left, symbol_left) - 10)
    x0 = int(max(x0, w * 0.58))
    x1 = int(min(w, max(label_right, hx1) + 40))
    y0 = int(max(0, min(y_top, hy0 - 24)))
    y1 = int(min(h, max(y_bot, hy1 + 80)))
    return _clamp_bbox((x0, y0, x1, y1), w, h)


def _legend_frame_left(
    gray: np.ndarray,
    *,
    symbol_left: int,
    y0: int,
    y1: int,
    search_px: int,
) -> int:
    """Nudge further left if a vertical panel border sits beside the swatches."""
    h, w = gray.shape
    x_right = int(max(1, min(w - 1, symbol_left)))
    x_left = int(max(0, x_right - search_px))
    if y1 <= y0 + 4 or x_right <= x_left + 2:
        return symbol_left - 20
    band = gray[y0:y1, x_left:x_right]
    # Strong vertical ink = border rule
    col = (band < 140).mean(axis=0)
    hits = np.where(col > 0.35)[0]
    if len(hits) == 0:
        return symbol_left - 20
    return int(x_left + hits.min())


def _legend_symbol_left_edge(
    gray: np.ndarray,
    *,
    label_left: float,
    y0: int,
    y1: int,
    search_px: int,
) -> int:
    """Find the left extent of legend swatches by scanning ink left of labels."""
    h, w = gray.shape
    x_right = int(max(1, min(w - 1, label_left - 2)))
    x_left = int(max(0, x_right - search_px))
    fallback = int(label_left - search_px * 0.85)
    if y1 <= y0 + 2 or x_right <= x_left + 2:
        return fallback

    band = gray[y0:y1, x_left:x_right]
    ink = band < 185
    col_frac = ink.mean(axis=0)
    active = col_frac > 0.012
    if not np.any(active):
        return fallback

    # Confirm this band is attached to the label column (ink near the right edge)
    near_label = active[-max(12, len(active) // 8) :]
    if not np.any(near_label):
        return fallback

    idxs = np.where(active)[0]
    # Allow dashed / broken symbol ink: keep the leftmost active column in the
    # search band as long as there is ink near the labels.
    return int(x_left + int(idxs.min()) - 8)


def _find_header(
    spans: list[dict],
    width: int,
    keys: tuple[str, ...],
    *,
    prefer_right: bool,
) -> dict | None:
    hits = []
    for s in spans:
        t = s["text"].strip().upper()
        if any(k in t for k in keys):
            hits.append(s)
    if not hits:
        return None
    hits.sort(
        key=lambda s: (
            0 if (not prefer_right or s["bbox_px"][0] > width * 0.5) else 1,
            s["bbox_px"][1],
        )
    )
    return hits[0]


def _estimate_right_rail_split(
    spans: list[dict],
    width: int,
    *,
    headers: list[dict],
) -> int:
    """Fallback split when legend detection fails."""
    header_xs = [
        int(h["bbox_px"][0])
        for h in headers
        if h["bbox_px"][0] > width * 0.45
    ]
    if header_xs:
        split = min(header_xs) - 16
        return int(max(width * 0.52, min(split, width * 0.82)))

    right_lefts = sorted(
        s["bbox_px"][0]
        for s in spans
        if (s["bbox_px"][0] + s["bbox_px"][2]) / 2 > width * 0.55
    )
    if not right_lefts:
        return int(width * 0.70)
    idx = max(0, len(right_lefts) // 10)
    split = int(right_lefts[idx] - 12)
    return int(max(width * 0.55, min(split, width * 0.80)))


def drawing_mask(segments: SheetSegments) -> np.ndarray:
    """Boolean mask (H,W) True where perception is allowed."""
    mask = np.zeros((segments.height, segments.width), dtype=bool)
    x0, y0, x1, y1 = segments.drawing_bbox
    mask[y0:y1, x0:x1] = True
    for bbox in segments.chrome_bboxes:
        cx0, cy0, cx1, cy1 = bbox
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
