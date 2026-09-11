from __future__ import annotations

import cv2
import numpy as np

from .stages.associate import Association
from .stages.contours_curved import Polyline
from .stages.elevations_ocr import ElevationCallout

# BGR colors
CYAN = (255, 220, 0)
GREEN = (80, 200, 80)
ORANGE = (0, 140, 255)
MAGENTA = (255, 0, 255)


def compose_overlay(
    image_bgr: np.ndarray,
    *,
    polylines: list[Polyline],
    elevations: list[ElevationCallout],
    associations: list[Association],
    layers: dict[str, bool] | None = None,
) -> np.ndarray:
    layers = {
        "contours": True,
        "elevations": True,
        "symbols": True,
        "associations": True,
        **(layers or {}),
    }
    out = image_bgr.copy()
    out = cv2.addWeighted(out, 0.75, np.full_like(out, 255), 0.25, 0)

    if layers.get("contours", True):
        for poly in polylines:
            pts = poly.points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                out, [pts], isClosed=False, color=CYAN, thickness=1, lineType=cv2.LINE_AA
            )

    if layers.get("associations", True):
        for a in associations:
            p1 = (int(a.elevation.x), int(a.elevation.y))
            p2 = (int(a.point_xy[0]), int(a.point_xy[1]))
            cv2.line(out, p1, p2, MAGENTA, 1, cv2.LINE_AA)

    if layers.get("elevations", True):
        for e in elevations:
            x0, y0, x1, y1 = map(int, e.bbox)
            cv2.rectangle(out, (x0, y0), (x1, y1), GREEN, 1)
            label = f"{e.value_ft:.2f}"
            cv2.putText(
                out,
                label,
                (x0, max(12, y0 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                GREEN,
                1,
                cv2.LINE_AA,
            )

    if layers.get("symbols", True):
        for e in elevations:
            cv2.circle(out, (int(e.x), int(e.y)), 5, ORANGE, 1, cv2.LINE_AA)

    return out
