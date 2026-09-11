from __future__ import annotations

import re
from dataclasses import dataclass

from .rasterize import RasterPage

# Prefer a feet mark after the number so a stray "0" between "=" and "10'" is ignored.
_SCALE_PATTERNS = [
    re.compile(
        r"""scale\s*[:\-]?\s*1\s*["”]?\s*=\s*(\d+(?:\.\d+)?)\s*(?:'|’|ft|feet)""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""1\s*["”]\s*=\s*(\d+(?:\.\d+)?)\s*(?:'|’|ft|feet)""",
        re.IGNORECASE,
    ),
]


@dataclass
class ScaleDetection:
    ft_per_inch: float
    raw_text: str
    source: str
    confidence: float


def detect_scale(page: RasterPage) -> ScaleDetection:
    """Detect engineering scale (feet per drawing inch) from sheet text."""
    spans = page.text_spans

    for s in spans:
        hit = _match_scale(s["text"], "pdf_span")
        if hit:
            return hit

    # CAD sheets often split "SCALE: 1\" =" and "10'" across neighboring spans
    for s in spans:
        t = s["text"]
        if not re.search(r"scale|1\s*[\"”]\s*=", t, re.I):
            continue
        cx = (s["bbox_px"][0] + s["bbox_px"][2]) / 2
        cy = (s["bbox_px"][1] + s["bbox_px"][3]) / 2
        nearby = [(0.0, t)]
        for other in spans:
            if other is s:
                continue
            ox = (other["bbox_px"][0] + other["bbox_px"][2]) / 2
            oy = (other["bbox_px"][1] + other["bbox_px"][3]) / 2
            dist = abs(ox - cx) + abs(oy - cy)
            if dist < 160:
                nearby.append((dist, other["text"]))
        nearby.sort(key=lambda item: item[0])
        merged = " ".join(text for _, text in nearby[:12])
        hit = _match_scale(merged, "pdf_nearby")
        if hit:
            return hit

    blob = " ".join(s["text"] for s in spans)
    hit = _match_scale(blob, "pdf_blob")
    if hit:
        return hit

    return ScaleDetection(
        ft_per_inch=10.0,
        raw_text='assumed 1" = 10\' (not found on sheet)',
        source="default",
        confidence=0.2,
    )


def _match_scale(text: str, source: str) -> ScaleDetection | None:
    for pat in _SCALE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        ft = float(m.group(1))
        if 1.0 <= ft <= 200.0:
            conf = 0.95 if "scale" in text.lower() else 0.85
            return ScaleDetection(
                ft_per_inch=ft,
                raw_text=m.group(0).strip(),
                source=source,
                confidence=conf,
            )
    return None
