from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .rasterize import RasterPage
from .segment import SheetSegments, point_in_bbox

_KNOWN_LABELS = [
    "BUILDING LINE",
    "PROPERTY LINE",
    "EXISTING CONTOUR",
    "PROPOSED CONTOUR",
    "TOP OF CURB",
    "MATCH EXISTING",
    "CURB CUT",
    "PROPOSED BUILDING",
    "CONCRETE SIDEWALK",
    "HEAVY-DUTY CONCRETE",
    "HEAVY DUTY CONCRETE",
    "SILT FENCE",
    "LIMIT OF DISTURBANCE",
]


@dataclass
class LegendEntry:
    name: str
    symbol_hint: str
    source: str
    bbox: tuple[float, float, float, float] | None = None
    icon_rgb: np.ndarray | None = field(default=None, repr=False)


def extract_legend(page: RasterPage, segments: SheetSegments) -> list[LegendEntry]:
    """Extract map legend symbol names from PDF text (prefer legend panel)."""
    entries: list[LegendEntry] = []
    seen: set[str] = set()

    spans = page.text_spans
    if segments.legend_bbox is not None:
        legend_spans = [
            s
            for s in spans
            if point_in_bbox(
                (s["bbox_px"][0] + s["bbox_px"][2]) / 2,
                (s["bbox_px"][1] + s["bbox_px"][3]) / 2,
                segments.legend_bbox,
            )
        ]
    else:
        w = page.width_px
        legend_spans = [s for s in spans if s["bbox_px"][0] > w * 0.68]

    start_y = 0.0
    for s in legend_spans:
        if re.search(r"map\s*legend|legend", s["text"], re.I):
            start_y = s["bbox_px"][3]
            break

    for s in sorted(legend_spans, key=lambda x: (x["bbox_px"][1], x["bbox_px"][0])):
        text = " ".join(s["text"].split())
        if not text or len(text) < 3:
            continue
        if s["bbox_px"][1] < start_y - 2:
            continue
        if re.search(r"map\s*legend|^legend$|general notes|keynotes", text, re.I):
            continue
        if re.fullmatch(r"[\d.\-]+", text):
            continue

        name = text.strip(" :-")
        key = name.upper()
        if key in seen or len(name) > 80:
            continue

        known = any(k in key for k in _KNOWN_LABELS)
        looks_like = bool(re.match(r"^[A-Z0-9][A-Z0-9\s\-/&().%]{2,60}$", name)) or known
        noise = (
            "DATE",
            "PROJECT",
            "DESIGNED",
            "DRAWN",
            "SCALE",
            "AS NOTED",
            "NTS",
            "CONSTRUCTION DOCUMENTS",
            "REVISION",
            "SHEET",
            "PDC",
        )
        if any(key == n or key.startswith(n + " ") for n in noise):
            continue
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", name):
            continue
        if not looks_like:
            continue

        if entries and len(name) <= 12 and not known:
            prev = entries[-1].name.upper()
            if len(prev) <= 16 and not any(k in prev for k in _KNOWN_LABELS):
                entries[-1] = LegendEntry(
                    name=f"{entries[-1].name} {name}".strip(),
                    symbol_hint=_symbol_hint(f"{prev} {key}"),
                    source="pdf_legend_panel",
                    bbox=entries[-1].bbox,
                )
                continue

        seen.add(key)
        entries.append(
            LegendEntry(
                name=name,
                symbol_hint=_symbol_hint(key),
                source="pdf_legend_panel",
                bbox=tuple(s["bbox_px"]),  # type: ignore[arg-type]
            )
        )

    blob = " ".join(s["text"].upper() for s in spans)
    for label in _KNOWN_LABELS:
        if label not in blob:
            continue
        if any(label in e.name.upper() or e.name.upper() in label for e in entries):
            continue
        # Try to locate a matching span for cropping
        bbox = None
        for s in spans:
            if label in s["text"].upper():
                bbox = tuple(s["bbox_px"])  # type: ignore[arg-type]
                break
        entries.append(
            LegendEntry(
                name=label.title() if label.isupper() else label,
                symbol_hint=_symbol_hint(label),
                source="pdf_keyword",
                bbox=bbox,
            )
        )

    deduped: list[LegendEntry] = []
    seen_final: set[str] = set()
    for e in entries:
        key = re.sub(r"\s+", " ", e.name.upper()).strip()
        if key in seen_final:
            continue
        if any(key in s or s in key for s in seen_final if abs(len(s) - len(key)) < 12):
            if any(key != s and (key in s) for s in seen_final):
                continue
        seen_final.add(key)
        deduped.append(e)

    return attach_legend_icons(page, deduped, segments)


def attach_legend_icons(
    page: RasterPage,
    entries: list[LegendEntry],
    segments: SheetSegments,
) -> list[LegendEntry]:
    """Crop the graphic to the left of each legend label as a symbol thumbnail."""
    img = page.image_bgr
    h, w = img.shape[:2]
    out: list[LegendEntry] = []

    for e in entries:
        icon = None
        if e.bbox is not None:
            x0, y0, x1, y1 = e.bbox
            row_h = max(18.0, y1 - y0)
            # Symbol swatch is typically left of the label text
            sw = max(48.0, min(140.0, (x1 - x0) * 0.9))
            ix0 = int(max(0, x0 - sw - 8))
            ix1 = int(max(ix0 + 8, x0 - 4))
            iy0 = int(max(0, y0 - row_h * 0.35))
            iy1 = int(min(h, y1 + row_h * 0.35))
            # Keep crop inside legend panel when known
            if segments.legend_bbox is not None:
                lx0, ly0, lx1, ly1 = segments.legend_bbox
                ix0 = max(ix0, lx0)
                ix1 = min(ix1, lx1)
                iy0 = max(iy0, ly0)
                iy1 = min(iy1, ly1)
            if ix1 > ix0 + 4 and iy1 > iy0 + 4:
                crop = img[iy0:iy1, ix0:ix1]
                if crop.size:
                    # Pad to a consistent thumbnail
                    thumb = _pad_square(crop, 96)
                    icon = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

        out.append(
            LegendEntry(
                name=e.name,
                symbol_hint=e.symbol_hint,
                source=e.source,
                bbox=e.bbox,
                icon_rgb=icon,
            )
        )
    return out


def _pad_square(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(
        crop_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    rh, rw = resized.shape[:2]
    y0 = (size - rh) // 2
    x0 = (size - rw) // 2
    canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
    return canvas


def _symbol_hint(name_upper: str) -> str:
    if "EXISTING CONTOUR" in name_upper:
        return "dashed contour line"
    if "PROPOSED CONTOUR" in name_upper:
        return "solid contour line"
    if "PROPERTY" in name_upper:
        return "long-dash / dash-dot line"
    if "BUILDING LINE" in name_upper:
        return "building outline line"
    if "TOP OF CURB" in name_upper or "(TC)" in name_upper:
        return "spot elevation callout (TC)"
    if "MATCH EXISTING" in name_upper or "(ME)" in name_upper:
        return "match-existing marker (ME)"
    if "CURB CUT" in name_upper:
        return "triangle / taper symbol"
    if "SIDEWALK" in name_upper:
        return "stipple / hatch pattern"
    if "HEAVY" in name_upper and "CONCRETE" in name_upper:
        return "dense hatch pattern"
    if "BUILDING" in name_upper:
        return "diagonal hatch / footprint"
    return "see legend graphic"
