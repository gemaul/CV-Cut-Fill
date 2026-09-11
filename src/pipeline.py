from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .overlay import compose_overlay
from .stages.associate import Association, associate_elevations
from .stages.contours_curved import Polyline
from .stages.contours_seeded import extract_seeded_contours
from .stages.cut_fill import CutFillResult, compute_cut_fill
from .stages.elevations_ocr import ElevationCallout, extract_elevations
from .stages.evaluate import EvaluationResult, evaluate
from .stages.fuse_surfaces import collect_sheet_surface_points, fuse_surfaces
from .stages.legend import LegendEntry, extract_legend
from .stages.line_styles import (
    apply_style_kinds,
    load_builtin_templates,
    refine_templates_from_legend,
)
from .stages.rasterize import RasterPage, rasterize_pdf
from .stages.scale_detect import ScaleDetection, detect_scale
from .stages.segment import SheetSegments, drawing_mask, point_in_bbox, segment_sheet
from .stages.site_structure import (
    SiteStructure,
    detect_site_structure,
    merge_structure_into_segments,
)
from .stages.surfaces import SurfaceBundle, build_surfaces


@dataclass
class PipelineResult:
    page: RasterPage
    segments: SheetSegments
    site: SiteStructure
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
    sheet_role: str = "auto"
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
    sheet_role: str = "auto",
    compute_volumes: bool = True,
) -> PipelineResult:
    """Run perception (+ optional single-sheet volumes).

    ``sheet_role``:
      - ``proposed``: grading plan (C-201) — finished-grade spots + proposed contours
      - ``existing``: existing conditions (V-101) — surveyed contours/labels
      - ``auto``: single-sheet mode (try-your-own)
    """
    warnings: list[str] = []
    status = {
        "contours": "pending",
        "elevations": "pending",
        "symbols": "heuristic",
        "existing_vs_proposed": "pending",
        "legend": "pending",
        "scale": "pending",
        "segments": "pending",
        "property": "pending",
        "building": "pending",
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

    elevations = extract_elevations(
        page,
        use_ocr=use_ocr,
        drawing_bbox=segments.drawing_bbox,
        exclude_bboxes=segments.chrome_bboxes,
        sheet_role=sheet_role,
    )
    drawing_elevations = [
        e for e in elevations if point_in_bbox(e.x, e.y, segments.drawing_bbox)
    ]
    status["elevations"] = "done" if drawing_elevations else "empty"
    if not drawing_elevations:
        warnings.append("No high-quality elevation callouts in drawing viewport.")

    site = detect_site_structure(
        page,
        segments,
        elevations=drawing_elevations,
        legend=legend,
        sheet_role=sheet_role,
    )
    status["property"] = (
        "done"
        if site.property_lines or site.property_polygon is not None
        else "empty"
    )
    status["building"] = (
        "done"
        if site.building_lines
        or site.building_polygon is not None
        else "empty"
    )
    if not site.property_lines and site.property_polygon is None:
        warnings.append("Property boundary not traced from legend line style.")
    if status["building"] == "empty" and sheet_role == "proposed":
        warnings.append("Building footprint not confidently detected.")

    segments = merge_structure_into_segments(segments, site)

    x0, y0, x1, y1 = segments.drawing_bbox
    drawing = page.image_bgr[y0:y1, x0:x1]
    # Chrome-only mask (drawing minus legend/notes/title) — do NOT hard-clip to property.
    page_mask = drawing_mask(segments)
    local_mask = page_mask[y0:y1, x0:x1]
    if not local_mask.any():
        local_mask = np.ones(drawing.shape[:2], dtype=bool)

    local_polys, contour_stats = extract_seeded_contours(
        drawing,
        drawing_elevations,
        elev_offset_xy=(float(x0), float(y0)),
        sheet_role=sheet_role,
        allow_mask=local_mask,
        use_fallback=True,
    )
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

    # Pre-seeded MAP LEGEND styles: existing dashed / proposed solid / property dash-dot-dot
    style_templates = refine_templates_from_legend(load_builtin_templates(), legend)
    before_n = len(polylines)
    polylines = apply_style_kinds(
        polylines,
        page.image_bgr,
        templates=style_templates,
        sheet_role=sheet_role,
    )
    style_dropped = before_n - len(polylines)

    # Soft post-filter: drop residual structure/property/building kinds
    cleaned: list[Polyline] = []
    rejected: list[Polyline] = []
    for poly in polylines:
        if poly.kind in ("structure", "property", "building"):
            rejected.append(poly)
            continue
        if poly.kind not in ("existing", "proposed"):
            poly = Polyline(
                points=poly.points,
                length_px=poly.length_px,
                source=poly.source,
                kind="proposed" if sheet_role != "existing" else "existing",
                straightness=poly.straightness,
                mean_abs_turn=poly.mean_abs_turn,
            )
        cleaned.append(poly)
    polylines = cleaned

    overlay_extras = list(site.property_lines) + list(site.building_lines)
    status["contours"] = "done" if polylines else "empty"
    if rejected or style_dropped:
        warnings.append(
            f"Dropped {len(rejected) + style_dropped} non-contour strokes "
            f"(property-line/structure via legend styles)."
        )
    n_exist = sum(1 for p in polylines if p.kind == "existing")
    n_prop = sum(1 for p in polylines if p.kind == "proposed")
    status["existing_vs_proposed"] = (
        f"existing={n_exist}, proposed={n_prop}; "
        f"seeded={contour_stats.seeds_hit}/{contour_stats.seeds_tried}, "
        f"closed={contour_stats.closed_count}, open={contour_stats.open_count}, "
        f"mean_len={contour_stats.mean_length_px:.0f}px, "
        f"fallback={contour_stats.fallback_count}, "
        f"style_drop={style_dropped}"
    )
    warnings.append(
        f"Contours: seeds_hit={contour_stats.seeds_hit}, "
        f"closed={contour_stats.closed_count}, "
        f"mean_length={contour_stats.mean_length_px:.0f}px, "
        f"fallback={contour_stats.fallback_count}, "
        f"legend_styles=existing/proposed/property."
    )

    associations = associate_elevations(
        drawing_elevations,
        polylines,
        max_dist_px=160.0 if sheet_role == "existing" else 48.0,
    )
    status["association"] = "done" if associations else "empty"
    if drawing_elevations and associations:
        rate = len(associations) / max(len(drawing_elevations), 1)
        if rate < 0.35:
            warnings.append(
                f"Low elevation↔contour association rate ({rate:.0%})."
            )

    # Single-sheet surfaces (used for try-your-own, or as interim before fusion)
    surfaces = build_surfaces(
        page=page,
        associations=associations,
        elevations=drawing_elevations,
        polylines=polylines,
        scale_ft_per_inch=effective_scale,
    )
    if sheet_role == "existing":
        status["cut_fill"] = "deferred"
        volumes = CutFillResult(
            cut_cy=0.0,
            fill_cy=0.0,
            net_cy=0.0,
            ok=False,
            message="Existing sheet contributes the existing surface; volumes fused with grading plan.",
            cell_count=0,
        )
        evaluation = None
    elif compute_volumes:
        if surfaces.existing_count < 3:
            warnings.append(
                "Few/no existing-surface samples on this sheet alone; "
                "prefer fusing with V-101 existing conditions."
            )
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
        polylines=polylines + overlay_extras,
        elevations=drawing_elevations,
        associations=associations,
        segments=segments,
        site=site,
        layers=overlay_layers,
    )

    return PipelineResult(
        page=page,
        segments=segments,
        site=site,
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
        sheet_role=sheet_role,
        stage_status=status,
        warnings=warnings,
    )


def fuse_cut_fill(
    grading: PipelineResult,
    existing: PipelineResult,
    *,
    ground_truth: dict | None = None,
) -> PipelineResult:
    """Proposed surface from C-201 + existing surface from V-101 → cut/fill.

    Returns an updated copy of ``grading`` with fused surfaces/volumes/evaluation.
    """
    g_scale = grading.scale.ft_per_inch
    e_scale = existing.scale.ft_per_inch

    proposed_pts = collect_sheet_surface_points(
        page=grading.page,
        site=grading.site,
        associations=grading.associations,
        elevations=grading.elevations,
        polylines=grading.polylines,
        scale_ft_per_inch=g_scale,
        sheet_role="proposed",
    )
    existing_pts = collect_sheet_surface_points(
        page=existing.page,
        site=existing.site,
        associations=existing.associations,
        elevations=existing.elevations,
        polylines=existing.polylines,
        scale_ft_per_inch=e_scale,
        sheet_role="existing",
    )

    surfaces = fuse_surfaces(
        proposed_sheet=proposed_pts,
        existing_sheet=existing_pts,
        supplement_existing_from_grading=True,
    )
    volumes = compute_cut_fill(surfaces)
    volumes = CutFillResult(
        cut_cy=volumes.cut_cy,
        fill_cy=volumes.fill_cy,
        net_cy=volumes.net_cy,
        ok=volumes.ok,
        message=(
            "Fused cut/fill: proposed←C-201 (contours+spots), "
            "existing←V-101 (surveyed contours/labels). "
            f"{volumes.message}"
        ),
        cell_count=volumes.cell_count,
    )
    evaluation = evaluate(volumes, ground_truth) if ground_truth else None

    warnings = list(grading.warnings)
    warnings.append(
        f"Fused surfaces: proposed_pts={surfaces.proposed_count}, "
        f"existing_pts={surfaces.existing_count}."
    )
    if surfaces.existing_count < 3:
        warnings.append("Fused existing surface still sparse after V-101 merge.")

    status = dict(grading.stage_status)
    status["cut_fill"] = "fused" if volumes.ok else "failed"
    status["existing_vs_proposed"] = (
        f"fused existing_pts={surfaces.existing_count}, "
        f"proposed_pts={surfaces.proposed_count}"
    )

    return PipelineResult(
        page=grading.page,
        segments=grading.segments,
        site=grading.site,
        scale=grading.scale,
        legend=grading.legend,
        polylines=grading.polylines,
        elevations=grading.elevations,
        associations=grading.associations,
        surfaces=surfaces,
        volumes=volumes,
        evaluation=evaluation,
        overlay_bgr=grading.overlay_bgr,
        sheet_label=grading.sheet_label,
        sheet_role="proposed",
        stage_status=status,
        warnings=warnings,
    )
