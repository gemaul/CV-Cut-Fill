from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .contours_curved import Polyline
from .elevations_ocr import ElevationCallout
from .legend import LegendEntry
from .pdf_vectors import extract_property_and_building_strokes
from .rasterize import RasterPage
from .segment import Region, SheetSegments, point_in_bbox


@dataclass
class SiteStructure:
    """Semantic site layout detected before contour extraction."""

    property_polygon: np.ndarray | None  # (N, 2) float xy page coords
    property_bbox: tuple[int, int, int, int] | None
    building_polygon: np.ndarray | None
    building_bbox: tuple[int, int, int, int] | None
    zones: list[Region] = field(default_factory=list)
    property_lines: list[Polyline] = field(default_factory=list)
    building_lines: list[Polyline] = field(default_factory=list)
    notes: str = ""

    def interior_mask(self, height: int, width: int) -> np.ndarray:
        """True inside property and outside building (page coords)."""
        mask = np.zeros((height, width), dtype=bool)
        if self.property_polygon is not None and len(self.property_polygon) >= 3:
            poly = self.property_polygon.astype(np.int32).reshape(-1, 1, 2)
            canvas = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(canvas, [poly], 255)
            mask = canvas > 0
        elif self.property_bbox is not None:
            x0, y0, x1, y1 = self.property_bbox
            mask[y0:y1, x0:x1] = True
        if self.building_polygon is not None and len(self.building_polygon) >= 3:
            poly = self.building_polygon.astype(np.int32).reshape(-1, 1, 2)
            canvas = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(canvas, [poly], 255)
            mask &= canvas == 0
        elif self.building_bbox is not None:
            x0, y0, x1, y1 = self.building_bbox
            mask[y0:y1, x0:x1] = False
        return mask


def detect_site_structure(
    page: RasterPage,
    segments: SheetSegments,
    *,
    elevations: list[ElevationCallout] | None = None,
    legend: list[LegendEntry] | None = None,
    zone_tile: tuple[int, int] = (480, 400),
    sheet_role: str = "auto",
) -> SiteStructure:
    """Detect property / building from PDF vector strokes (not text hulls).

    Building is skipped entirely for ``sheet_role=existing`` (e.g. V-101).
    """
    elevations = elevations or []
    legend = legend or []
    x0, y0, x1, y1 = segments.drawing_bbox
    draw = page.image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
    dh, dw = gray.shape

    detect_building = sheet_role != "existing"
    prop_page, bld_page = extract_property_and_building_strokes(
        page,
        detect_building=detect_building,
        drawing_bbox_px=segments.drawing_bbox,
    )

    # Convert page-space strokes → drawing-local for zone building
    def to_local(lines: list[Polyline]) -> list[Polyline]:
        out: list[Polyline] = []
        for poly in lines:
            pts = poly.points.copy()
            pts[:, 0] -= x0
            pts[:, 1] -= y0
            out.append(
                Polyline(
                    points=pts,
                    length_px=poly.length_px,
                    source=poly.source,
                    kind=poly.kind,
                    straightness=poly.straightness,
                    mean_abs_turn=poly.mean_abs_turn,
                )
            )
        return out

    prop_lines_local = to_local(prop_page)
    anchors = _collect_anchors(page, segments.drawing_bbox, elevations)
    prop_poly_local = _polygon_from_traced_lines(prop_lines_local, dw, dh)

    bld_poly_local = None
    bld_bbox_local = None
    bld_lines_local: list[Polyline] = []
    if detect_building and bld_page:
        bld_lines_local = to_local(bld_page)
        bld_poly_local, bld_bbox_local, bld_lines_local = _refine_building_with_rooms(
            bld_lines_local, anchors, dw, dh
        )

    zones_local = _build_interior_zones(
        dw,
        dh,
        prop_poly_local,
        bld_poly_local,
        bld_bbox_local,
        tile=zone_tile,
    )

    def lift_poly(poly: np.ndarray | None) -> np.ndarray | None:
        if poly is None or len(poly) == 0:
            return None
        out = poly.astype(np.float32).copy()
        out[:, 0] += x0
        out[:, 1] += y0
        return out

    def lift_bbox(
        bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        a, b, c, d = bbox
        return (a + x0, b + y0, c + x0, d + y0)

    def lift_lines(lines: list[Polyline]) -> list[Polyline]:
        out: list[Polyline] = []
        for poly in lines:
            pts = poly.points.copy()
            pts[:, 0] += x0
            pts[:, 1] += y0
            out.append(
                Polyline(
                    points=pts,
                    length_px=poly.length_px,
                    source=poly.source,
                    kind=poly.kind,
                    straightness=poly.straightness,
                    mean_abs_turn=poly.mean_abs_turn,
                )
            )
        return out

    # Property / building lines already in page space
    prop_lines = prop_page
    bld_lines = lift_lines(bld_lines_local) if bld_lines_local else []
    prop_poly = lift_poly(prop_poly_local)
    bld_poly = lift_poly(bld_poly_local)

    prop_bbox = None
    if prop_lines:
        pts = np.vstack([p.points for p in prop_lines])
        prop_bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
    elif prop_poly is not None:
        xs, ys = prop_poly[:, 0], prop_poly[:, 1]
        prop_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    zones: list[Region] = []
    for i, zb in enumerate(zones_local):
        zx0, zy0, zx1, zy1 = zb
        zones.append(
            Region(
                name=f"topo_zone_{i+1}",
                bbox=(zx0 + x0, zy0 + y0, zx1 + x0, zy1 + y0),
                color="#86EFAC",
                role="topo_zone",
            )
        )

    notes = (
        f"property={'pdf' if prop_lines else 'miss'}(n={len(prop_lines)}); "
        f"building={'pdf' if bld_lines else 'skip' if not detect_building else 'miss'}"
        f"(n={len(bld_lines)}); zones={len(zones)}; role={sheet_role}"
    )
    _ = (gray, legend)
    return SiteStructure(
        property_polygon=prop_poly,
        property_bbox=prop_bbox,
        building_polygon=bld_poly,
        building_bbox=lift_bbox(bld_bbox_local),
        zones=zones,
        property_lines=prop_lines,
        building_lines=bld_lines,
        notes=notes,
    )


def filter_polylines_by_site_and_legend(
    polylines: list[Polyline],
    site: SiteStructure,
    *,
    page_shape: tuple[int, int],
    legend: list[LegendEntry] | None = None,
    min_inside_frac: float = 0.55,
) -> tuple[list[Polyline], list[Polyline]]:
    """Keep topo-like strokes inside property / outside building; drop key mismatches.

    Returns (kept_topo, rejected).
    """
    legend = legend or []
    h, w = page_shape
    interior = site.interior_mask(h, w)
    has_interior = bool(interior.any())

    reject_styles = _reject_style_names(legend)
    kept: list[Polyline] = []
    rejected: list[Polyline] = []

    for poly in polylines:
        if poly.kind in ("property", "building", "structure"):
            rejected.append(poly)
            continue

        style = _stroke_style(poly)
        if style in reject_styles:
            rejected.append(poly)
            continue

        # Long axis-aligned solids near building → building walls, not contours
        if _looks_like_building_wall(poly, site):
            rejected.append(poly)
            continue
        if _looks_like_property_stroke(poly, style):
            rejected.append(poly)
            continue

        if has_interior:
            frac = _fraction_inside(poly, interior)
            if frac < min_inside_frac:
                rejected.append(poly)
                continue

        kept.append(poly)

    return kept, rejected


def merge_structure_into_segments(
    segments: SheetSegments, site: SiteStructure
) -> SheetSegments:
    """Append property / building / topo-zone regions onto sheet segments for UI."""
    regions = list(segments.regions)
    if site.property_bbox is not None:
        regions.append(
            Region("property", site.property_bbox, "#EF4444", "site")
        )
    if site.building_bbox is not None:
        regions.append(
            Region("building", site.building_bbox, "#F97316", "site")
        )
    regions.extend(site.zones)
    note = segments.notes
    if site.notes:
        note = f"{note} | {site.notes}"
    return SheetSegments(
        width=segments.width,
        height=segments.height,
        drawing_bbox=segments.drawing_bbox,
        legend_bbox=segments.legend_bbox,
        notes_bbox=segments.notes_bbox,
        title_block_bbox=segments.title_block_bbox,
        scale_bbox=segments.scale_bbox,
        regions=regions,
        notes=note,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _legend_has(legend: list[LegendEntry], key: str) -> bool:
    return any(key in e.name.upper() for e in legend)


def _reject_style_names(legend: list[LegendEntry]) -> set[str]:
    """Styles that should not be treated as topo contours."""
    out = {"property_dashdot", "building_solid"}
    for e in legend:
        u = e.name.upper()
        if "PROPERTY" in u:
            out.add("property_dashdot")
        if "BUILDING LINE" in u or u.strip() in {"B.L.", "BL"}:
            out.add("building_solid")
        if "SIDEWALK" in u or "HEAVY" in u:
            out.add("hatch")
        if "CURB CUT" in u:
            out.add("symbol")
    return out


def _collect_anchors(
    page: RasterPage,
    drawing_bbox: tuple[int, int, int, int],
    elevations: list[ElevationCallout],
) -> list[tuple[str, float, float]]:
    x0, y0, x1, y1 = drawing_bbox
    anchors: list[tuple[str, float, float]] = []
    for s in page.text_spans:
        bx0, by0, bx1, by1 = s["bbox_px"]
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0
        if not point_in_bbox(cx, cy, drawing_bbox):
            continue
        # Local drawing coords
        lx, ly = cx - x0, cy - y0
        t = s["text"].strip()
        u = t.upper()
        if re.fullmatch(r"\d{3}", t):
            anchors.append(("room", lx, ly))
        elif "BUILDING LINE" in u:
            anchors.append(("bl", lx, ly))
        elif any(k in u for k in ("SIDEWALK", "CURB", "LANDSCAPE BUFFER")):
            anchors.append(("edge", lx, ly))
        elif re.search(r"\d\+\d{2}", t):
            anchors.append(("sta", lx, ly))
        elif any(
            k in u
            for k in (
                "CLASSROOM",
                "LOBBY",
                "OFFICE",
                "TEACHER",
                "TOILET",
                "MECHANICAL",
                "JANITOR",
                "CORRIDOR",
            )
        ):
            anchors.append(("room_label", lx, ly))

    for e in elevations:
        if point_in_bbox(e.x, e.y, drawing_bbox):
            anchors.append(("elev", e.x - x0, e.y - y0))
    return anchors


def _detect_property(
    gray: np.ndarray,
    anchors: list[tuple[str, float, float]],
    *,
    legend_hint: bool,
) -> tuple[np.ndarray | None, list[Polyline]]:
    """Deprecated path — property now comes from extract_style_line_strokes."""
    _ = (gray, anchors, legend_hint)
    return None, []


def _polygon_from_traced_lines(
    lines: list[Polyline], width: int, height: int
) -> np.ndarray | None:
    """Optional soft enclosure from traced property strokes (for zones/mask only)."""
    if len(lines) < 3:
        return None
    pts = np.vstack([p.points for p in lines]).astype(np.float32)
    if len(pts) < 5:
        return None
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, max(4.0, 0.02 * peri), True)
    poly = approx.reshape(-1, 2).astype(np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
    if len(poly) < 3:
        return None
    return poly


def _refine_building_with_rooms(
    wall_lines: list[Polyline],
    anchors: list[tuple[str, float, float]],
    width: int,
    height: int,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, list[Polyline]]:
    """Keep thick wall strokes only when room labels confirm a building exists."""
    rooms = np.array(
        [[a[1], a[2]] for a in anchors if a[0] in ("room", "room_label")],
        dtype=np.float32,
    )
    if len(rooms) < 3 or not wall_lines:
        return None, None, []

    cent = rooms.mean(axis=0)
    dist = np.linalg.norm(rooms - cent, axis=1)
    med = float(np.median(dist)) if len(dist) else 200.0
    keep = rooms[dist <= max(180.0, med * 2.6)]
    if len(keep) < 3:
        keep = rooms

    pad = 100
    rx0 = int(max(0, keep[:, 0].min() - pad))
    ry0 = int(max(0, keep[:, 1].min() - pad))
    rx1 = int(min(width - 1, keep[:, 0].max() + pad))
    ry1 = int(min(height - 1, keep[:, 1].max() + pad))

    near_walls: list[Polyline] = []
    for poly in wall_lines:
        pts = poly.points
        inside = (
            (pts[:, 0] >= rx0 - 30)
            & (pts[:, 0] <= rx1 + 30)
            & (pts[:, 1] >= ry0 - 30)
            & (pts[:, 1] <= ry1 + 30)
        )
        if float(inside.mean()) < 0.35:
            continue
        if poly.length_px < 40:
            continue
        near_walls.append(poly)

    if len(near_walls) < 3:
        return None, None, []

    wpts = np.vstack([p.points for p in near_walls]).astype(np.float32)
    hull = cv2.convexHull(wpts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, max(3.0, 0.02 * peri), True)
    poly = approx.reshape(-1, 2).astype(np.float32)

    covered = 0
    for x, y in keep:
        if (
            cv2.pointPolygonTest(
                poly.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), False
            )
            >= 0
        ):
            covered += 1
    if covered < max(2, int(0.45 * len(keep))):
        # Still show wall strokes near rooms even if hull is imperfect
        xs = np.concatenate([p.points[:, 0] for p in near_walls])
        ys = np.concatenate([p.points[:, 1] for p in near_walls])
        bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )
        return None, bbox, near_walls

    xs, ys = poly[:, 0], poly[:, 1]
    bbox = (
        int(max(0, xs.min())),
        int(max(0, ys.min())),
        int(min(width - 1, xs.max())),
        int(min(height - 1, ys.max())),
    )
    return poly, bbox, near_walls


def _detect_building_from_strokes(
    gray: np.ndarray,
    wall_lines: list[Polyline],
    anchors: list[tuple[str, float, float]],
    *,
    legend_hint: bool,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, list[Polyline]]:
    """Deprecated — use _refine_building_with_rooms."""
    _ = (gray, legend_hint)
    h, w = gray.shape[:2]
    return _refine_building_with_rooms(wall_lines, anchors, w, h)


def _detect_building(
    gray: np.ndarray,
    anchors: list[tuple[str, float, float]],
    *,
    legend_hint: bool,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, list[Polyline]]:
    """Deprecated path — building now comes from stroke tracing."""
    _ = (gray, anchors, legend_hint)
    return None, None, []


def _build_interior_zones(
    width: int,
    height: int,
    prop_poly: np.ndarray | None,
    bld_poly: np.ndarray | None,
    bld_bbox: tuple[int, int, int, int] | None,
    *,
    tile: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    mask = np.zeros((height, width), dtype=np.uint8)
    if prop_poly is not None and len(prop_poly) >= 3:
        cv2.fillPoly(mask, [prop_poly.astype(np.int32).reshape(-1, 1, 2)], 255)
    else:
        mask[:, :] = 255

    if bld_poly is not None and len(bld_poly) >= 3:
        cv2.fillPoly(mask, [bld_poly.astype(np.int32).reshape(-1, 1, 2)], 0)
    elif bld_bbox is not None:
        x0, y0, x1, y1 = bld_bbox
        mask[y0:y1, x0:x1] = 0

    tw, th = tile
    zones: list[tuple[int, int, int, int]] = []
    for yy in range(0, height, th):
        for xx in range(0, width, tw):
            cell = mask[yy : min(height, yy + th), xx : min(width, xx + tw)]
            if cell.size == 0 or float(cell.mean()) < 40:
                continue
            ys, xs = np.where(cell > 0)
            z = (
                int(xx + xs.min()),
                int(yy + ys.min()),
                int(xx + xs.max()),
                int(yy + ys.max()),
            )
            if z[2] - z[0] > 100 and z[3] - z[1] > 100:
                zones.append(z)
    return zones


def _sample_perimeter_style_lines(
    gray: np.ndarray, poly: np.ndarray, *, kind: str
) -> list[Polyline]:
    """Turn polygon edges into Polyline records for overlay / filtering."""
    if poly is None or len(poly) < 2:
        return []
    pts = poly.reshape(-1, 2).astype(np.float32)
    out: list[Polyline] = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        length = float(np.linalg.norm(b - a))
        if length < 40:
            continue
        # densify for style sampling
        n = max(8, int(length // 12))
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n)
        edge = np.column_stack([xs, ys]).astype(np.float32)
        pl = Polyline(
            points=edge,
            length_px=length,
            source="site_structure",
            kind=kind,
        )
        _annotate_simple(pl)
        out.append(pl)
    return out


def _annotate_simple(poly: Polyline) -> None:
    pts = poly.points
    if len(pts) < 2:
        poly.straightness = 1.0
        poly.mean_abs_turn = 0.0
        return
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    poly.straightness = chord / max(poly.length_px, 1e-3)
    turns = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        a1 = math.atan2(float(v1[1]), float(v1[0]))
        a2 = math.atan2(float(v2[1]), float(v2[0]))
        d = abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi)
        turns.append(d)
    poly.mean_abs_turn = float(np.mean(turns)) if turns else 0.0


def _stroke_style(poly: Polyline) -> str:
    """Coarse stroke class used to match legend key categories."""
    if poly.mean_abs_turn >= 0.12:
        return "topo_curve"
    if poly.straightness > 0.92 and poly.length_px > 120:
        # Dashiness is unknown without image; use kind if already set
        if poly.kind == "existing":
            return "contour_dash"
        if poly.kind in ("property",):
            return "property_dashdot"
        if poly.kind in ("building", "structure"):
            return "building_solid"
        if poly.straightness > 0.98:
            return "building_solid"
        return "property_dashdot"
    return "topo_curve"


def _looks_like_building_wall(poly: Polyline, site: SiteStructure) -> bool:
    if site.building_bbox is None:
        return False
    if poly.mean_abs_turn >= 0.08:
        return False
    if poly.straightness < 0.93 or poly.length_px < 80:
        return False
    x0, y0, x1, y1 = site.building_bbox
    # Expand slightly — walls sit on the footprint edge
    pad = 40
    bx0, by0, bx1, by1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    pts = poly.points
    inside = (
        (pts[:, 0] >= bx0)
        & (pts[:, 0] <= bx1)
        & (pts[:, 1] >= by0)
        & (pts[:, 1] <= by1)
    )
    return float(inside.mean()) > 0.6


def _looks_like_property_stroke(poly: Polyline, style: str) -> bool:
    if style != "property_dashdot":
        return False
    return poly.length_px > 200 and poly.straightness > 0.9


def _fraction_inside(poly: Polyline, mask: np.ndarray) -> float:
    h, w = mask.shape
    pts = poly.points
    if len(pts) == 0:
        return 0.0
    xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
    return float(mask[ys, xs].mean())
