from __future__ import annotations

import cv2
import numpy as np

from .stages.associate import Association
from .stages.contours_curved import Polyline
from .stages.elevations_ocr import ElevationCallout
from .stages.segment import SheetSegments

CYAN = (255, 220, 0)
GREEN = (60, 200, 60)
ORANGE = (0, 140, 255)
MAGENTA = (255, 0, 255)
BLUE = (255, 120, 40)
PURPLE = (200, 80, 200)
YELLOW = (0, 200, 255)


def compose_overlay(
    image_bgr: np.ndarray,
    *,
    polylines: list[Polyline],
    elevations: list[ElevationCallout],
    associations: list[Association],
    segments: SheetSegments | None = None,
    layers: dict[str, bool] | None = None,
) -> np.ndarray:
    layers = {
        "contours": True,
        "elevations": True,
        "symbols": True,
        "associations": True,
        "segments": True,
        **(layers or {}),
    }
    out = image_bgr.copy()
    out = cv2.addWeighted(out, 0.65, np.full_like(out, 30), 0.35, 0)

    if layers.get("segments", True) and segments is not None:
        for bbox, color in (
            (segments.drawing_bbox, BLUE),
            (segments.legend_bbox, PURPLE),
            (segments.title_block_bbox, YELLOW),
        ):
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)

    if layers.get("contours", True):
        for poly in polylines:
            pts = poly.points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                out, [pts], isClosed=False, color=CYAN, thickness=2, lineType=cv2.LINE_AA
            )

    if layers.get("associations", True):
        for a in associations:
            p1 = (int(a.elevation.x), int(a.elevation.y))
            p2 = (int(a.point_xy[0]), int(a.point_xy[1]))
            cv2.line(out, p1, p2, MAGENTA, 2, cv2.LINE_AA)

    if layers.get("elevations", True):
        for e in elevations:
            x0, y0, x1, y1 = map(int, e.bbox)
            cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), GREEN, 2)
            label = f"{e.value_ft:.2f}"
            cv2.putText(
                out,
                label,
                (x0, max(14, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                GREEN,
                1,
                cv2.LINE_AA,
            )

    if layers.get("symbols", True):
        for e in elevations:
            cv2.circle(out, (int(e.x), int(e.y)), 7, ORANGE, 2, cv2.LINE_AA)

    return out
