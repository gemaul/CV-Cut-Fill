from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
import numpy as np


@dataclass
class RasterPage:
    image_bgr: np.ndarray
    width_px: int
    height_px: int
    dpi: int
    page_width_in: float
    page_height_in: float
    pdf_path: Path
    page_index: int
    # Native PDF text spans for elevation extraction (PDF coords → later mapped)
    text_spans: list[dict]


def rasterize_pdf(
    pdf_path: str | Path,
    *,
    page_index: int = 0,
    dpi: int = 150,
) -> RasterPage:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= doc.page_count:
        raise ValueError(f"page_index {page_index} out of range (0..{doc.page_count - 1})")

    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    # PyMuPDF is RGB; OpenCV expects BGR
    image_bgr = img[:, :, ::-1].copy()

    text_spans: list[dict] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                text_spans.append(
                    {
                        "text": text,
                        "bbox_pdf": (x0, y0, x1, y1),
                        "bbox_px": (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom),
                        "source": "pdf_text",
                    }
                )

    rect = page.rect
    return RasterPage(
        image_bgr=image_bgr,
        width_px=pix.width,
        height_px=pix.height,
        dpi=dpi,
        page_width_in=rect.width / 72.0,
        page_height_in=rect.height / 72.0,
        pdf_path=pdf_path,
        page_index=page_index,
        text_spans=text_spans,
    )
