from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surfaces import SurfaceBundle


@dataclass
class CutFillResult:
    cut_cy: float
    fill_cy: float
    net_cy: float
    ok: bool
    message: str
    cell_count: int


def compute_cut_fill(surfaces: SurfaceBundle) -> CutFillResult:
    """Deterministic grid cut/fill between existing and proposed surfaces."""
    existing = surfaces.existing
    proposed = surfaces.proposed
    if existing.size == 0 or np.isnan(existing).all() or np.isnan(proposed).all():
        return CutFillResult(
            cut_cy=0.0,
            fill_cy=0.0,
            net_cy=0.0,
            ok=False,
            message="No surface grid available for cut/fill.",
            cell_count=0,
        )

    delta = proposed - existing  # + = fill (raise grade), - = cut (lower grade)
    valid = np.isfinite(delta)
    if not valid.any():
        return CutFillResult(
            cut_cy=0.0,
            fill_cy=0.0,
            net_cy=0.0,
            ok=False,
            message="Surface grid contained no finite cells.",
            cell_count=0,
        )

    cell_area = surfaces.cell_size_ft ** 2
    cut_cf = float((-delta[valid & (delta < 0)]).sum() * cell_area)
    fill_cf = float((delta[valid & (delta > 0)]).sum() * cell_area)
    cut_cy = cut_cf / 27.0
    fill_cy = fill_cf / 27.0
    return CutFillResult(
        cut_cy=cut_cy,
        fill_cy=fill_cy,
        net_cy=fill_cy - cut_cy,
        ok=True,
        message="Deterministic grid cut/fill (proposed − existing).",
        cell_count=int(valid.sum()),
    )
