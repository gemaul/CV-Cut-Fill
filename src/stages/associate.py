from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contours_curved import Polyline
from .elevations_ocr import ElevationCallout


@dataclass
class Association:
    elevation: ElevationCallout
    polyline_index: int
    point_xy: tuple[float, float]
    distance_px: float


def associate_elevations(
    elevations: list[ElevationCallout],
    polylines: list[Polyline],
    *,
    max_dist_px: float = 40.0,
) -> list[Association]:
    if not elevations or not polylines:
        return []

    associations: list[Association] = []
    for elev in elevations:
        best_i = -1
        best_d = float("inf")
        best_pt = (elev.x, elev.y)
        for i, poly in enumerate(polylines):
            d, pt = _distance_to_polyline(elev.x, elev.y, poly.points)
            if d < best_d:
                best_d = d
                best_i = i
                best_pt = pt
        if best_i >= 0 and best_d <= max_dist_px:
            associations.append(
                Association(
                    elevation=elev,
                    polyline_index=best_i,
                    point_xy=best_pt,
                    distance_px=best_d,
                )
            )
    return associations


def _distance_to_polyline(
    x: float, y: float, points: np.ndarray
) -> tuple[float, tuple[float, float]]:
    if len(points) == 0:
        return float("inf"), (x, y)
    if len(points) == 1:
        p = points[0]
        d = float(np.hypot(x - p[0], y - p[1]))
        return d, (float(p[0]), float(p[1]))

    best_d = float("inf")
    best_pt = (float(points[0, 0]), float(points[0, 1]))
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        d, pt = _point_to_segment(x, y, p1, p2)
        if d < best_d:
            best_d = d
            best_pt = pt
    return best_d, best_pt


def _point_to_segment(
    x: float, y: float, p1: np.ndarray, p2: np.ndarray
) -> tuple[float, tuple[float, float]]:
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    leng2 = vx * vx + vy * vy
    if leng2 < 1e-9:
        return float(np.hypot(x - p1[0], y - p1[1])), (float(p1[0]), float(p1[1]))
    t = ((x - p1[0]) * vx + (y - p1[1]) * vy) / leng2
    t = max(0.0, min(1.0, t))
    px, py = p1[0] + t * vx, p1[1] + t * vy
    return float(np.hypot(x - px, y - py)), (float(px), float(py))
