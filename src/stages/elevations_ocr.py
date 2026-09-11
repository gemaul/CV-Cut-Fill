from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .rasterize import RasterPage

# Typical site elevations on this job are ~1000–1030 ft; also allow relative values.
_ELEV_RE = re.compile(
    r"^(?:"
    r"(?:EL(?:EV)?\.?\s*)?([+-]?\d{3,4}(?:\.\d{1,3})?)"
    r"|"
    r"([+-]?\d{2,4}\.\d{1,3})"
    r")$"
)


@dataclass
class ElevationCallout:
    text: str
    value_ft: float
    x: float
    y: float
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float = 1.0


def extract_elevations(
    page: RasterPage,
    *,
    use_ocr: bool = True,
    min_elev: float = 800.0,
    max_elev: float = 1200.0,
) -> list[ElevationCallout]:
    callouts: list[ElevationCallout] = []

    for span in page.text_spans:
        parsed = _parse_elevation(span["text"], min_elev=min_elev, max_elev=max_elev)
        if parsed is None:
            continue
        x0, y0, x1, y1 = span["bbox_px"]
        callouts.append(
            ElevationCallout(
                text=span["text"],
                value_ft=parsed,
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                bbox=(x0, y0, x1, y1),
                source="pdf_text",
                confidence=0.95,
            )
        )

    if use_ocr and len(callouts) < 8:
        callouts.extend(
            _ocr_elevations(page.image_bgr, min_elev=min_elev, max_elev=max_elev)
        )

    # Deduplicate near-identical detections
    return _dedupe(callouts)


def _parse_elevation(
    text: str, *, min_elev: float, max_elev: float
) -> float | None:
    cleaned = text.strip().replace(",", "")
    m = _ELEV_RE.match(cleaned)
    if not m:
        # Also accept bare integers that look like elevations when longer
        if re.fullmatch(r"\d{3,4}(?:\.\d{1,3})?", cleaned):
            val = float(cleaned)
        else:
            return None
    else:
        raw = m.group(1) or m.group(2)
        val = float(raw)
    if min_elev <= val <= max_elev:
        return val
    return None


def _ocr_elevations(
    image_bgr: np.ndarray, *, min_elev: float, max_elev: float
) -> list[ElevationCallout]:
    try:
        import easyocr
    except ImportError:
        return []

    # Downscale for speed on large sheets
    h, w = image_bgr.shape[:2]
    scale = 1.0
    work = image_bgr
    if max(h, w) > 2200:
        scale = 2200 / max(h, w)
        work = cv2_resize(image_bgr, scale)

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(work[:, :, ::-1])  # RGB
    out: list[ElevationCallout] = []
    for bbox, text, conf in results:
        parsed = _parse_elevation(text, min_elev=min_elev, max_elev=max_elev)
        if parsed is None:
            continue
        xs = [p[0] / scale for p in bbox]
        ys = [p[1] / scale for p in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        out.append(
            ElevationCallout(
                text=text,
                value_ft=parsed,
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                bbox=(x0, y0, x1, y1),
                source="easyocr",
                confidence=float(conf),
            )
        )
    return out


def cv2_resize(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    return cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def _dedupe(callouts: list[ElevationCallout], dist_px: float = 12.0) -> list[ElevationCallout]:
    kept: list[ElevationCallout] = []
    for c in sorted(callouts, key=lambda x: -x.confidence):
        if any(
            abs(c.x - k.x) < dist_px
            and abs(c.y - k.y) < dist_px
            and abs(c.value_ft - k.value_ft) < 0.05
            for k in kept
        ):
            continue
        kept.append(c)
    return kept
