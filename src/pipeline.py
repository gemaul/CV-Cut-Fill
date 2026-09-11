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
from .stages.legend import LegendEntry, extract_legend
from .stages.rasterize import RasterPage, rasterize_pdf
from .stages.scale_detect import ScaleDetection, detect_scale
from .stages.segment import SheetSegments, point_in_bbox, segment_sheet
from .stages.surfaces import SurfaceBundle, build_surfaces


@dataclass
class PipelineResult:
    page: RasterPage
    segments: SheetSegments
    scale: ScaleDetection
    legend: list[LegendEntry]
    polylines: list[Polyline]
    elevations: list[ElevationCallout]
    associations: list[Association]
    surfaces: SurfaceBundle
    volumes: CutFillResult
    evaluation: EvaluationResult | None
    overlay_bgr: np.ndarray
    sheet_label: str = ""
    stage_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def run_pipeline(
    pdf_path: str | Path,
    *,
    page_index: int = 0,
    dpi: int = 150,
    scale_ft_per_inch: float | None = None,
    ground_truth: dict | None = None,
    use_ocr: bool = False,
    overlay_layers: dict[str, bool] | None = None,
    sheet_label: str = "",
    compute_volumes: bool = True,
) -> PipelineResult:
    warnings: list[str] = []
    status = {
        "contours": "pending",
        "elevations": "pending",
        "symbols": "heuristic",
        "existing_vs_proposed": "pending",
        "legend": "pending",
        "scale": "pending",
        "segments": "pending",
        "association": "pending",
        "cut_fill": "pending",
    }

    page = rasterize_pdf(pdf_path, page_index=page_index, dpi=dpi)

    segments = segment_sheet(page)
    status["segments"] = "done"

    scale = detect_scale(page)
    status["scale"] = "done" if scale.source != "default" else "default"
    if scale.source == "default":
        warnings.append(f"Scale not found on sheet; using {scale.raw_text}.")
    effective_scale = (
        float(scale_ft_per_inch) if scale_ft_per_inch is not None else scale.ft_per_inch
    )

    legend = extract_legend(page, segments)
    status["legend"] = "done" if legend else "empty"

    x0, y0, x1, y1 = segments.drawing_bbox
    drawing = page.image_bgr[y0:y1, x0:x1]
    local_polys = extract_curved_contours(drawing)
    polylines: list[Polyline] = []
    for poly in local_polys:
        pts = poly.points.copy()
        pts[:, 0] += x0
        pts[:, 1] += y0
        polylines.append(
            Polyline(
                points=pts,
                length_px=poly.length_px,
                source=poly.source,
                kind=poly.kind,
                straightness=poly.straightness,
                mean_abs_turn=poly.mean_abs_turn,
            )
        )
    status["contours"] = "done" if polylines else "empty"
    n_exist = sum(1 for p in polylines if p.kind == "existing")
    n_prop = sum(1 for p in polylines if p.kind == "proposed")
    status["existing_vs_proposed"] = f"existing={n_exist}, proposed={n_prop}"

    elevations = extract_elevations(
        page,
        use_ocr=use_ocr,
        drawing_bbox=segments.drawing_bbox,
    )
    drawing_elevations = [
        e for e in elevations if point_in_bbox(e.x, e.y, segments.drawing_bbox)
    ]
    status["elevations"] = "done" if drawing_elevations else "empty"
    if not drawing_elevations:
        warnings.append("No high-quality elevation callouts in drawing viewport.")

    associations = associate_elevations(drawing_elevations, polylines)
    status["association"] = "done" if associations else "empty"
    if drawing_elevations and associations:
        rate = len(associations) / max(len(drawing_elevations), 1)
        if rate < 0.35:
            warnings.append(
                f"Low elevation↔contour association rate ({rate:.0%})."
            )

    surfaces = build_surfaces(
        page=page,
        associations=associations,
        elevations=drawing_elevations,
        polylines=polylines,
        scale_ft_per_inch=effective_scale,
    )
    if surfaces.existing_count < 3:
        warnings.append(
            "Few/no existing-surface samples (ME/dashed); using proposed−0.25ft fallback."
        )

    if compute_volumes:
        volumes = compute_cut_fill(surfaces)
        status["cut_fill"] = "done" if volumes.ok else "failed"
        if not volumes.ok:
            warnings.append(volumes.message)
        evaluation = evaluate(volumes, ground_truth) if ground_truth else None
    else:
        volumes = CutFillResult(
            cut_cy=0.0,
            fill_cy=0.0,
            net_cy=0.0,
            ok=False,
            message="Volumes not computed for this sheet (context only).",
            cell_count=0,
        )
        status["cut_fill"] = "skipped"
        evaluation = None

    overlay_bgr = compose_overlay(
        page.image_bgr,
        polylines=polylines,
        elevations=drawing_elevations,
        associations=associations,
        segments=segments,
        layers=overlay_layers,
    )

    return PipelineResult(
        page=page,
        segments=segments,
        scale=scale,
        legend=legend,
        polylines=polylines,
        elevations=drawing_elevations,
        associations=associations,
        surfaces=surfaces,
        volumes=volumes,
        evaluation=evaluation,
        overlay_bgr=overlay_bgr,
        sheet_label=sheet_label,
        stage_status=status,
        warnings=warnings,
    )
