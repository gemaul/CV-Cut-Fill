from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .rasterize import RasterPage

# Spot grades on this job are typically ####.## ; reject bare integers by default.
_DECIMAL_ELEV = re.compile(r"^(?:EL(?:EV)?\.?\s*)?([+-]?\d{3,4}\.\d{1,3})$", re.I)
_INT_ELEV = re.compile(r"^(?:EL(?:EV)?\.?\s*)?([+-]?\d{3,4})$", re.I)
_CONTEXT_OK = re.compile(
    r"\b(TC|ME|EL|ELEV|ELEVATION|TOP\s*OF\s*CURB|SPOT|GRADE|FF|FFE)\b",
    re.I,
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
    kind: str = "spot"  # spot | ffe | existing_match | weak


def extract_elevations(
    page: RasterPage,
    *,
    use_ocr: bool = True,
    min_elev: float = 800.0,
    max_elev: float = 1200.0,
    drawing_bbox: tuple[int, int, int, int] | None = None,
) -> list[ElevationCallout]:
    callouts: list[ElevationCallout] = []

    for span in page.text_spans:
        x0, y0, x1, y1 = span["bbox_px"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if drawing_bbox is not None:
            dx0, dy0, dx1, dy1 = drawing_bbox
            if not (dx0 <= cx <= dx1 and dy0 <= cy <= dy1):
                continue

        ctx = _nearby_context(page, cx, cy, radius=70)
        parsed = _parse_elevation(
            span["text"],
            context=ctx,
            min_elev=min_elev,
            max_elev=max_elev,
            allow_bare_int=False,
        )
        if parsed is None:
            continue
        value, kind, conf = parsed
        callouts.append(
            ElevationCallout(
                text=span["text"],
                value_ft=value,
                x=cx,
                y=cy,
                bbox=(x0, y0, x1, y1),
                source="pdf_text",
                confidence=conf,
                kind=kind,
            )
        )

    if use_ocr and len(callouts) < 8:
        callouts.extend(
            _ocr_elevations(
                page.image_bgr,
                min_elev=min_elev,
                max_elev=max_elev,
                drawing_bbox=drawing_bbox,
            )
        )

    callouts = _dedupe(callouts)
    callouts = _dedupe_ffe(callouts)
    return callouts


def _nearby_context(page: RasterPage, x: float, y: float, radius: float) -> str:
    parts = []
    for s in page.text_spans:
        sx = (s["bbox_px"][0] + s["bbox_px"][2]) / 2
        sy = (s["bbox_px"][1] + s["bbox_px"][3]) / 2
        if abs(sx - x) <= radius and abs(sy - y) <= radius:
            parts.append(s["text"])
    return " ".join(parts)


def _parse_elevation(
    text: str,
    *,
    context: str,
    min_elev: float,
    max_elev: float,
    allow_bare_int: bool,
) -> tuple[float, str, float] | None:
    cleaned = text.strip().replace(",", "")
    ctx_ok = bool(_CONTEXT_OK.search(context) or _CONTEXT_OK.search(cleaned))

    m = _DECIMAL_ELEV.match(cleaned)
    if m:
        val = float(m.group(1))
        if not (min_elev <= val <= max_elev):
            return None
        kind = "existing_match" if re.search(r"\bME\b", context, re.I) else "spot"
        conf = 0.95 if ctx_ok else 0.8
        return val, kind, conf

    m = _INT_ELEV.match(cleaned)
    if m and (allow_bare_int or ctx_ok):
        val = float(m.group(1))
        if not (min_elev <= val <= max_elev):
            return None
        # Bare integers without elevation context are usually IDs / stations
        if not ctx_ok:
            return None
        return val, "weak", 0.55

    return None


def _dedupe_ffe(callouts: list[ElevationCallout], min_count: int = 5) -> list[ElevationCallout]:
    """Collapse repeated identical values (typical FFE along a building)."""
    counts = Counter(round(c.value_ft, 2) for c in callouts)
    ffe_values = {v for v, n in counts.items() if n >= min_count}
    if not ffe_values:
        return callouts

    kept: list[ElevationCallout] = []
    buckets: dict[float, list[ElevationCallout]] = defaultdict(list)
    for c in callouts:
        key = round(c.value_ft, 2)
        if key in ffe_values:
            buckets[key].append(c)
        else:
            kept.append(c)

    for key, group in buckets.items():
        # Keep a single representative near the cluster centroid
        xs = np.array([g.x for g in group])
        ys = np.array([g.y for g in group])
        cx, cy = float(xs.mean()), float(ys.mean())
        best = min(group, key=lambda g: (g.x - cx) ** 2 + (g.y - cy) ** 2)
        kept.append(
            ElevationCallout(
                text=best.text,
                value_ft=best.value_ft,
                x=best.x,
                y=best.y,
                bbox=best.bbox,
                source=best.source,
                confidence=min(best.confidence, 0.7),
                kind="ffe",
            )
        )
    return kept


def _ocr_elevations(
    image_bgr: np.ndarray,
    *,
    min_elev: float,
    max_elev: float,
    drawing_bbox: tuple[int, int, int, int] | None,
) -> list[ElevationCallout]:
    try:
        import easyocr
    except ImportError:
        return []

    import cv2

    work = image_bgr
    ox = oy = 0
    if drawing_bbox is not None:
        x0, y0, x1, y1 = drawing_bbox
        work = image_bgr[y0:y1, x0:x1]
        ox, oy = x0, y0

    h, w = work.shape[:2]
    scale = 1.0
    if max(h, w) > 2200:
        scale = 2200 / max(h, w)
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(work[:, :, ::-1])
    out: list[ElevationCallout] = []
    for bbox, text, conf in results:
        parsed = _parse_elevation(
            text,
            context=text,
            min_elev=min_elev,
            max_elev=max_elev,
            allow_bare_int=False,
        )
        if parsed is None:
            continue
        value, kind, base_conf = parsed
        xs = [p[0] / scale for p in bbox]
        ys = [p[1] / scale for p in bbox]
        x0, x1 = min(xs) + ox, max(xs) + ox
        y0, y1 = min(ys) + oy, max(ys) + oy
        out.append(
            ElevationCallout(
                text=text,
                value_ft=value,
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                bbox=(x0, y0, x1, y1),
                source="easyocr",
                confidence=min(float(conf), base_conf),
                kind=kind,
            )
        )
    return out


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
