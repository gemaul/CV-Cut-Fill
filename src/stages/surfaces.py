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
    existing: np.ndarray  # 2D grid elevations (ft)
    proposed: np.ndarray
    cell_size_ft: float
    origin_xy_ft: tuple[float, float]
    point_count: int
    ft_per_px: float
    method_note: str


def build_surfaces(
    *,
    page: RasterPage,
    associations: list[Association],
    elevations: list[ElevationCallout],
    polylines: list[Polyline],
    scale_ft_per_inch: float = 20.0,
    grid_n: int = 120,
) -> SurfaceBundle:
    """Build existing/proposed elevation grids from associated callouts.

    Heuristic for MVP:
    - Proposed surface = interpolate elevation callouts (prefer associated points)
    - Existing surface = proposed shifted toward a lower reference plane
      (25th percentile of elevations), representing pre-grade / stripped terrain
      when we cannot yet split existing vs proposed contours reliably.
    """
    ft_per_px = scale_ft_per_inch / page.dpi

    pts: list[tuple[float, float, float]] = []
    if associations:
        for a in associations:
            pts.append((a.point_xy[0], a.point_xy[1], a.elevation.value_ft))
    else:
        for e in elevations:
            pts.append((e.x, e.y, e.value_ft))

    # Also sprinkle elevations along associated polylines for smoother surfaces
    if associations and polylines:
        for a in associations:
            poly = polylines[a.polyline_index]
            for p in poly.points[:: max(1, len(poly.points) // 8)]:
                pts.append((float(p[0]), float(p[1]), a.elevation.value_ft))

    if len(pts) < 3:
        empty = np.full((grid_n, grid_n), np.nan, dtype=np.float64)
        return SurfaceBundle(
            existing=empty,
            proposed=empty,
            cell_size_ft=1.0,
            origin_xy_ft=(0.0, 0.0),
            point_count=len(pts),
            ft_per_px=ft_per_px,
            method_note="insufficient elevation points",
        )

    arr = np.array(pts, dtype=np.float64)
    xs_ft = arr[:, 0] * ft_per_px
    ys_ft = arr[:, 1] * ft_per_px
    zs = arr[:, 2]

    xmin, xmax = xs_ft.min(), xs_ft.max()
    ymin, ymax = ys_ft.min(), ys_ft.max()
    pad = 0.05 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_n),
        np.linspace(ymin, ymax, grid_n),
    )
    cell_size = float(max((xmax - xmin) / (grid_n - 1), (ymax - ymin) / (grid_n - 1)))

    proposed = griddata(
        np.column_stack([xs_ft, ys_ft]),
        zs,
        (grid_x, grid_y),
        method="linear",
    )
    # Fill holes with nearest
    if np.isnan(proposed).any():
        fill = griddata(
            np.column_stack([xs_ft, ys_ft]),
            zs,
            (grid_x, grid_y),
            method="nearest",
        )
        proposed = np.where(np.isnan(proposed), fill, proposed)

    z_ref = float(np.percentile(zs, 25))
    # Existing: blend proposed toward lower reference (stripped/existing proxy)
    existing = 0.65 * z_ref + 0.35 * proposed

    return SurfaceBundle(
        existing=existing,
        proposed=proposed,
        cell_size_ft=cell_size,
        origin_xy_ft=(float(xmin), float(ymin)),
        point_count=len(pts),
        ft_per_px=ft_per_px,
        method_note=(
            "proposed=interpolated callouts; existing=heuristic blend to 25th-pct "
            f"plane ({z_ref:.2f} ft); scale={scale_ft_per_inch:g} ft/in"
        ),
    )
