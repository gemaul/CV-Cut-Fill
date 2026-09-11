from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .overlay import compose_overlay
from .stages.associate import Association, associate_elevations
from .stages.contours_curved import Polyline, extract_curved_contours
from .stages.cut_fill import CutFillResult, compute_cut_fill
from .stages.elevations_ocr import ElevationCallout, extract_elevations
from .stages.evaluate import EvaluationResult, evaluate
from .stages.rasterize import RasterPage, rasterize_pdf
from .stages.surfaces import SurfaceBundle, build_surfaces


@dataclass
class PipelineResult:
    page: RasterPage
    polylines: list[Polyline]
    elevations: list[ElevationCallout]
    associations: list[Association]
    surfaces: SurfaceBundle
    volumes: CutFillResult
    evaluation: EvaluationResult | None
    overlay_bgr: np.ndarray
    stage_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def run_pipeline(
    pdf_path: str | Path,
    *,
    page_index: int = 0,
    dpi: int = 150,
    scale_ft_per_inch: float = 20.0,
    ground_truth: dict | None = None,
    use_ocr: bool = True,
    overlay_layers: dict[str, bool] | None = None,
) -> PipelineResult:
    warnings: list[str] = []
    status = {
        "contours": "pending",
        "elevations": "pending",
        "symbols": "stub",
        "existing_vs_proposed": "heuristic",
        "legend": "stub",
        "association": "pending",
        "cut_fill": "pending",
    }

    page = rasterize_pdf(pdf_path, page_index=page_index, dpi=dpi)

    polylines = extract_curved_contours(page.image_bgr)
    status["contours"] = "done" if polylines else "empty"
    if not polylines:
        warnings.append("No curved contour polylines detected.")

    elevations = extract_elevations(page, use_ocr=use_ocr)
    status["elevations"] = "done" if elevations else "empty"
    if not elevations:
        warnings.append("No elevation callouts detected.")

    associations = associate_elevations(elevations, polylines)
    status["association"] = "done" if associations else "empty"

    surfaces = build_surfaces(
        page=page,
        associations=associations,
        elevations=elevations,
        polylines=polylines,
        scale_ft_per_inch=scale_ft_per_inch,
    )
    if surfaces.point_count < 3:
        warnings.append(
            "Fewer than 3 elevation-linked points; volume estimate is unreliable."
        )

    volumes = compute_cut_fill(surfaces)
    status["cut_fill"] = "done" if volumes.ok else "failed"
    if not volumes.ok:
        warnings.append(volumes.message)

    evaluation = evaluate(volumes, ground_truth) if ground_truth else None

    overlay_bgr = compose_overlay(
        page.image_bgr,
        polylines=polylines,
        elevations=elevations,
        associations=associations,
        layers=overlay_layers,
    )

    return PipelineResult(
        page=page,
        polylines=polylines,
        elevations=elevations,
        associations=associations,
        surfaces=surfaces,
        volumes=volumes,
        evaluation=evaluation,
        overlay_bgr=overlay_bgr,
        stage_status=status,
        warnings=warnings,
    )
