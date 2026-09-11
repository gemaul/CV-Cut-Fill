from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from .associate import Association
from .contours_curved import Polyline
from .elevations_ocr import ElevationCallout
from .rasterize import RasterPage


@dataclass
class SurfaceBundle:
    existing: np.ndarray
    proposed: np.ndarray
    cell_size_ft: float
    origin_xy_ft: tuple[float, float]
    point_count: int
    ft_per_px: float
    method_note: str
    proposed_count: int = 0
    existing_count: int = 0


def build_surfaces(
    *,
    page: RasterPage,
    associations: list[Association],
    elevations: list[ElevationCallout],
    polylines: list[Polyline],
    scale_ft_per_inch: float = 20.0,
    grid_n: int = 120,
) -> SurfaceBundle:
    """Build existing/proposed grids from classified elevations + contour kinds."""
    ft_per_px = scale_ft_per_inch / page.dpi

    proposed_pts: list[tuple[float, float, float]] = []
    existing_pts: list[tuple[float, float, float]] = []

    # Seed from elevations by kind
    for e in elevations:
        if e.kind in ("ffe",):
            # FFE is a pad level for proposed, not dense existing topo
            proposed_pts.append((e.x, e.y, e.value_ft))
        elif e.kind == "existing_match":
            existing_pts.append((e.x, e.y, e.value_ft))
        elif e.kind in ("spot", "weak"):
            proposed_pts.append((e.x, e.y, e.value_ft))

    # Associations: push value onto polyline; inherit contour class when present
    if associations and polylines:
        for a in associations:
            poly = polylines[a.polyline_index]
            z = a.elevation.value_ft
            samples = poly.points[:: max(1, len(poly.points) // 6)]
            bucket = existing_pts if poly.kind == "existing" else proposed_pts
            for p in samples:
                bucket.append((float(p[0]), float(p[1]), z))
            # also the association foot
            bucket.append((a.point_xy[0], a.point_xy[1], z))

    if len(proposed_pts) < 3 and elevations:
        proposed_pts = [(e.x, e.y, e.value_ft) for e in elevations]

    if len(proposed_pts) < 3:
        empty = np.full((grid_n, grid_n), np.nan, dtype=np.float64)
        return SurfaceBundle(
            existing=empty,
            proposed=empty,
            cell_size_ft=1.0,
            origin_xy_ft=(0.0, 0.0),
            point_count=len(proposed_pts) + len(existing_pts),
            ft_per_px=ft_per_px,
            method_note="insufficient elevation points",
            proposed_count=len(proposed_pts),
            existing_count=len(existing_pts),
        )

    prop = np.array(proposed_pts, dtype=np.float64)
    all_xy = prop[:, :2]
    if existing_pts:
        ex = np.array(existing_pts, dtype=np.float64)
        all_xy = np.vstack([all_xy, ex[:, :2]])

    xs = all_xy[:, 0] * ft_per_px
    ys = all_xy[:, 1] * ft_per_px
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    pad = 0.05 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_n),
        np.linspace(ymin, ymax, grid_n),
    )
    cell_size = float(max((xmax - xmin) / (grid_n - 1), (ymax - ymin) / (grid_n - 1)))

    proposed = _interpolate(prop, ft_per_px, grid_x, grid_y)

    if len(existing_pts) >= 3:
        existing = _interpolate(np.array(existing_pts, dtype=np.float64), ft_per_px, grid_x, grid_y)
        # Fill NaNs in existing from a gentle bias of proposed (not a flat plane)
        if np.isnan(existing).any():
            existing = np.where(
                np.isnan(existing),
                proposed - 0.15,  # slight cut bias only where unknown
                existing,
            )
        method = (
            f"proposed={len(proposed_pts)} pts; existing={len(existing_pts)} pts "
            f"(ME/dashed); scale={scale_ft_per_inch:g} ft/in"
        )
    else:
        # No reliable existing samples: use proposed minus small strip/subgrade offset
        # rather than a global 25th-percentile plane.
        existing = proposed - 0.25
        method = (
            f"proposed={len(proposed_pts)} pts; existing=proposed-0.25ft fallback "
            f"(no ME/dashed samples); scale={scale_ft_per_inch:g} ft/in"
        )

    return SurfaceBundle(
        existing=existing,
        proposed=proposed,
        cell_size_ft=cell_size,
        origin_xy_ft=(xmin, ymin),
        point_count=len(proposed_pts) + len(existing_pts),
        ft_per_px=ft_per_px,
        method_note=method,
        proposed_count=len(proposed_pts),
        existing_count=len(existing_pts),
    )


def _interpolate(
    pts: np.ndarray, ft_per_px: float, grid_x: np.ndarray, grid_y: np.ndarray
) -> np.ndarray:
    xs = pts[:, 0] * ft_per_px
    ys = pts[:, 1] * ft_per_px
    zs = pts[:, 2]
    surf = griddata(np.column_stack([xs, ys]), zs, (grid_x, grid_y), method="linear")
    if np.isnan(surf).any():
        fill = griddata(
            np.column_stack([xs, ys]), zs, (grid_x, grid_y), method="nearest"
        )
        surf = np.where(np.isnan(surf), fill, surf)
    return surf
