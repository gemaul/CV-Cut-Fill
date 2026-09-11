from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from .associate import Association
from .contours_curved import Polyline
from .elevations_ocr import ElevationCallout
from .rasterize import RasterPage
from .site_structure import SiteStructure
from .surfaces import SurfaceBundle


@dataclass
class SheetSurfacePoints:
    """Elevation samples in sheet-local feet (origin = property / drawing center)."""

    proposed_xyz: np.ndarray  # (N,3) feet
    existing_xyz: np.ndarray  # (M,3) feet
    origin_px: tuple[float, float]
    ft_per_px: float
    property_size_ft: tuple[float, float]
    note: str


def collect_sheet_surface_points(
    *,
    page: RasterPage,
    site: SiteStructure,
    associations: list[Association],
    elevations: list[ElevationCallout],
    polylines: list[Polyline],
    scale_ft_per_inch: float,
    sheet_role: str = "auto",
) -> SheetSurfacePoints:
    """Gather proposed/existing XYZ samples in feet relative to the property center."""
    ft_per_px = scale_ft_per_inch / page.dpi
    origin = _origin_px(site, page)
    prop_w, prop_h = _property_size_ft(site, ft_per_px, page)

    proposed: list[tuple[float, float, float]] = []
    existing: list[tuple[float, float, float]] = []

    def to_ft(x: float, y: float) -> tuple[float, float]:
        return ((x - origin[0]) * ft_per_px, (y - origin[1]) * ft_per_px)

    if sheet_role == "existing":
        # V-101 (and similar): all callouts feed the existing surface
        for e in elevations:
            xf, yf = to_ft(e.x, e.y)
            existing.append((xf, yf, e.value_ft))
        for a in associations:
            z = a.elevation.value_ft
            poly = polylines[a.polyline_index] if a.polyline_index < len(polylines) else None
            samples = [a.point_xy]
            if poly is not None:
                samples.extend(poly.points[:: max(1, len(poly.points) // 8)])
            for p in samples:
                xf, yf = to_ft(float(p[0]), float(p[1]))
                existing.append((xf, yf, z))
        # If associations are sparse, still seed each contour label onto the
        # nearest polyline within a generous radius (existing contour labels sit
        # on/near the line but detection gaps are common).
        if elevations and polylines and len(associations) < max(3, len(elevations) // 2):
            from .associate import associate_elevations

            loose = associate_elevations(elevations, polylines, max_dist_px=280.0)
            for a in loose:
                z = a.elevation.value_ft
                poly = polylines[a.polyline_index]
                for p in poly.points[:: max(1, len(poly.points) // 10)]:
                    xf, yf = to_ft(float(p[0]), float(p[1]))
                    existing.append((xf, yf, z))
        note = (
            f"existing-sheet samples={len(existing)} "
            f"(elev={len(elevations)}, assoc={len(associations)}); "
            f"scale={scale_ft_per_inch:g} ft/in"
        )
        return SheetSurfacePoints(
            proposed_xyz=np.zeros((0, 3), dtype=np.float64),
            existing_xyz=_as_xyz(existing),
            origin_px=origin,
            ft_per_px=ft_per_px,
            property_size_ft=(prop_w, prop_h),
            note=note,
        )

    # Proposed / grading sheet (C-201)
    for e in elevations:
        xf, yf = to_ft(e.x, e.y)
        if e.kind == "existing_match":
            existing.append((xf, yf, e.value_ft))
        else:
            # Finished-grade spots, FFE, weak decimals → proposed
            proposed.append((xf, yf, e.value_ft))

    if associations and polylines:
        for a in associations:
            poly = polylines[a.polyline_index]
            z = a.elevation.value_ft
            bucket = existing if poly.kind == "existing" else proposed
            samples = list(poly.points[:: max(1, len(poly.points) // 6)])
            samples.append(np.array(a.point_xy, dtype=np.float64))
            for p in samples:
                xf, yf = to_ft(float(p[0]), float(p[1]))
                bucket.append((xf, yf, z))

    note = (
        f"grading-sheet proposed={len(proposed)} existing_on_sheet={len(existing)}; "
        f"scale={scale_ft_per_inch:g} ft/in"
    )
    return SheetSurfacePoints(
        proposed_xyz=_as_xyz(proposed),
        existing_xyz=_as_xyz(existing),
        origin_px=origin,
        ft_per_px=ft_per_px,
        property_size_ft=(prop_w, prop_h),
        note=note,
    )


def fuse_surfaces(
    *,
    proposed_sheet: SheetSurfacePoints,
    existing_sheet: SheetSurfacePoints | None = None,
    grid_n: int = 140,
    supplement_existing_from_grading: bool = True,
) -> SurfaceBundle:
    """Build proposed from C-201 and existing from V-101 in a shared site-foot grid.

    Cut/fill uses ``proposed − existing`` (fill where proposed is higher).
    """
    prop = proposed_sheet.proposed_xyz
    if len(prop) < 3:
        empty = np.full((grid_n, grid_n), np.nan, dtype=np.float64)
        return SurfaceBundle(
            existing=empty,
            proposed=empty,
            cell_size_ft=1.0,
            origin_xy_ft=(0.0, 0.0),
            point_count=0,
            ft_per_px=proposed_sheet.ft_per_px,
            method_note="insufficient proposed points from grading sheet",
            proposed_count=0,
            existing_count=0,
        )

    existing_parts: list[np.ndarray] = []
    notes = [proposed_sheet.note]

    if existing_sheet is not None and len(existing_sheet.existing_xyz) >= 1:
        aligned = _align_existing_to_proposed(existing_sheet, proposed_sheet)
        existing_parts.append(aligned)
        notes.append(existing_sheet.note + " [aligned→grading frame]")

    if supplement_existing_from_grading and len(proposed_sheet.existing_xyz) >= 1:
        existing_parts.append(proposed_sheet.existing_xyz)
        notes.append(
            f"plus grading-sheet existing samples={len(proposed_sheet.existing_xyz)}"
        )

    if existing_parts:
        ex = np.vstack(existing_parts)
    else:
        ex = np.zeros((0, 3), dtype=np.float64)

    # Limit grid to the overlap of proposed samples and property footprint
    pw, ph = proposed_sheet.property_size_ft
    half_w, half_h = max(pw * 0.55, 40.0), max(ph * 0.55, 40.0)
    prop = prop[
        (np.abs(prop[:, 0]) <= half_w * 1.15) & (np.abs(prop[:, 1]) <= half_h * 1.15)
    ]
    if len(prop) < 3:
        prop = proposed_sheet.proposed_xyz

    if len(ex):
        ex = ex[
            (np.abs(ex[:, 0]) <= half_w * 1.25) & (np.abs(ex[:, 1]) <= half_h * 1.25)
        ]

    all_xy = prop[:, :2]
    if len(ex):
        all_xy = np.vstack([all_xy, ex[:, :2]])

    xmin, xmax = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    ymin, ymax = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    pad = 0.04 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_n),
        np.linspace(ymin, ymax, grid_n),
    )
    cell_size = float(max((xmax - xmin) / (grid_n - 1), (ymax - ymin) / (grid_n - 1)))

    proposed = _interpolate_ft(prop, grid_x, grid_y)

    if len(ex) >= 3:
        existing = _interpolate_ft(ex, grid_x, grid_y)
        if np.isnan(existing).any():
            existing = np.where(np.isnan(existing), proposed - 0.15, existing)
        method = (
            "FUSED: proposed←C-201; existing←V-101 (+ grading ME/dashed if any). "
            + " | ".join(notes)
        )
    else:
        existing = proposed - 0.25
        method = (
            "FUSED fallback: existing=proposed−0.25ft (V-101 samples < 3). "
            + " | ".join(notes)
        )

    # Mask cells outside a soft property ellipse so margins don't dominate CY
    rr = (grid_x / max(half_w, 1e-3)) ** 2 + (grid_y / max(half_h, 1e-3)) ** 2
    outside = rr > 1.05
    proposed = np.where(outside, np.nan, proposed)
    existing = np.where(outside, np.nan, existing)

    return SurfaceBundle(
        existing=existing,
        proposed=proposed,
        cell_size_ft=cell_size,
        origin_xy_ft=(xmin, ymin),
        point_count=len(prop) + len(ex),
        ft_per_px=proposed_sheet.ft_per_px,
        method_note=method,
        proposed_count=len(prop),
        existing_count=len(ex),
    )


def _interpolate_ft(pts: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Interpolate points already expressed in feet."""
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    surf = griddata(np.column_stack([xs, ys]), zs, (grid_x, grid_y), method="linear")
    if np.isnan(surf).any():
        fill = griddata(
            np.column_stack([xs, ys]), zs, (grid_x, grid_y), method="nearest"
        )
        surf = np.where(np.isnan(surf), fill, surf)
    return surf


def _align_existing_to_proposed(
    existing: SheetSurfacePoints, proposed: SheetSurfacePoints
) -> np.ndarray:
    """Map existing-sheet feet into the grading-sheet frame.

    Uses uniform scale from property bbox size ratio + shared property-centered origin.
    (Both sheets already use property-center local feet, so primarily a scale fix when
    property detections differ; translation stays ~0.)
    """
    pts = existing.existing_xyz.copy()
    if len(pts) == 0:
        return pts

    ew, eh = existing.property_size_ft
    pw, ph = proposed.property_size_ft
    scales = []
    if ew > 1 and pw > 1:
        scales.append(pw / ew)
    if eh > 1 and ph > 1:
        scales.append(ph / eh)
    s = float(np.median(scales)) if scales else 1.0
    # Clamp extreme mis-detects
    s = float(np.clip(s, 0.55, 1.8))
    pts[:, 0] *= s
    pts[:, 1] *= s
    return pts


def _origin_px(site: SiteStructure, page: RasterPage) -> tuple[float, float]:
    if site.property_bbox is not None:
        x0, y0, x1, y1 = site.property_bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if site.building_bbox is not None:
        x0, y0, x1, y1 = site.building_bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return (page.width_px / 2.0, page.height_px / 2.0)


def _property_size_ft(
    site: SiteStructure, ft_per_px: float, page: RasterPage
) -> tuple[float, float]:
    if site.property_bbox is not None:
        x0, y0, x1, y1 = site.property_bbox
        return ((x1 - x0) * ft_per_px, (y1 - y0) * ft_per_px)
    return (page.width_px * ft_per_px * 0.5, page.height_px * ft_per_px * 0.5)


def _as_xyz(pts: list[tuple[float, float, float]]) -> np.ndarray:
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)
