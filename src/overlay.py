from __future__ import annotations

import cv2
import numpy as np

from .stages.associate import Association
from .stages.contours_curved import Polyline
from .stages.elevations_ocr import ElevationCallout
from .stages.segment import SheetSegments
from .stages.site_structure import SiteStructure

CYAN = (255, 220, 0)
GREEN = (60, 200, 60)
ORANGE = (0, 140, 255)
MAGENTA = (255, 0, 255)
EXISTING = (180, 180, 255)
PROPOSED = (255, 200, 0)
PROPERTY = (0, 0, 220)
BUILDING = (0, 140, 255)


def compose_overlay(
    image_bgr: np.ndarray,
    *,
    polylines: list[Polyline],
    elevations: list[ElevationCallout],
    associations: list[Association],
    segments: SheetSegments | None = None,
    site: SiteStructure | None = None,
    layers: dict[str, bool] | None = None,
) -> np.ndarray:
    layers = {
        "contours": True,
        "elevations": True,
        "symbols": True,
        "associations": True,
        "segments": True,
        "property": True,
        "building": True,
        **(layers or {}),
    }
    out = image_bgr.copy()
    out = cv2.addWeighted(out, 0.7, np.full_like(out, 30), 0.3, 0)

    if layers.get("segments", True) and segments is not None:
        for region in segments.regions:
            if region.role in ("site", "topo_zone"):
                # Drawn via property/building/zone-specific styling below / as light boxes
                if region.role == "topo_zone" and layers.get("segments", True):
                    x0, y0, x1, y1 = region.bbox
                    cv2.rectangle(out, (x0, y0), (x1, y1), _hex_to_bgr(region.color), 1)
                continue
            x0, y0, x1, y1 = region.bbox
            color = _hex_to_bgr(region.color)
            thickness = 3 if region.role == "drawing" else 2
            cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
            cv2.putText(
                out,
                region.name,
                (x0 + 6, max(18, y0 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

    if layers.get("property", True) and site is not None:
        prop_strokes = [p for p in (site.property_lines or []) if len(p.points) >= 2]
        if prop_strokes:
            for poly in prop_strokes:
                pts = poly.points.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    out, [pts], isClosed=False, color=PROPERTY, thickness=3, lineType=cv2.LINE_AA
                )

    if layers.get("building", True) and site is not None:
        bld_strokes = [p for p in (site.building_lines or []) if len(p.points) >= 2]
        if bld_strokes:
            for poly in bld_strokes:
                pts = poly.points.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    out, [pts], isClosed=False, color=BUILDING, thickness=3, lineType=cv2.LINE_AA
                )

    if layers.get("contours", True):
        for poly in polylines:
            if poly.kind in ("structure", "property", "building"):
                color = PROPERTY if poly.kind == "property" else BUILDING
                if poly.kind == "structure":
                    continue
            else:
                color = EXISTING if poly.kind == "existing" else PROPOSED
            pts = poly.points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                out, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA
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
            cv2.putText(
                out,
                f"{e.value_ft:.2f}",
                (x0, max(14, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                GREEN,
                1,
                cv2.LINE_AA,
            )

    if layers.get("symbols", True):
        for e in elevations:
            cv2.circle(out, (int(e.x), int(e.y)), 7, ORANGE, 2, cv2.LINE_AA)

    return out


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
