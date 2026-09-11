from __future__ import annotations

import cv2
import numpy as np
import plotly.graph_objects as go

from .stages.associate import Association
from .stages.contours_curved import Polyline
from .stages.elevations_ocr import ElevationCallout
from .stages.segment import SheetSegments


def build_interactive_figure(
    image_bgr: np.ndarray,
    *,
    polylines: list[Polyline],
    elevations: list[ElevationCallout],
    associations: list[Association],
    segments: SheetSegments | None = None,
    layers: dict[str, bool] | None = None,
    focus_drawing: bool = True,
) -> go.Figure:
    """Zoomable Plotly viewer with toggleable detection layers."""
    layers = {
        "contours": True,
        "elevations": True,
        "symbols": True,
        "associations": True,
        "segments": True,
        **(layers or {}),
    }

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Slightly darken so overlays pop
    rgb = (rgb.astype(np.float32) * 0.82).clip(0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]

    fig = go.Figure()
    fig.add_trace(
        go.Image(z=rgb, hoverinfo="skip", name="Plan")
    )

    if layers.get("contours", True) and polylines:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for poly in polylines:
            for x, y in poly.points:
                xs.append(float(x))
                ys.append(float(y))
            xs.append(None)
            ys.append(None)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#00E5FF", width=2),
                name="Curved contours",
                hoverinfo="skip",
            )
        )

    if layers.get("associations", True) and associations:
        xs = []
        ys = []
        for a in associations:
            xs.extend([a.elevation.x, a.point_xy[0], None])
            ys.extend([a.elevation.y, a.point_xy[1], None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#FF00AA", width=1.5),
                name="Associations",
                hoverinfo="skip",
            )
        )

    if layers.get("elevations", True) and elevations:
        fig.add_trace(
            go.Scatter(
                x=[e.x for e in elevations],
                y=[e.y for e in elevations],
                mode="markers+text",
                marker=dict(size=9, color="#22C55E", line=dict(width=1, color="white")),
                text=[f"{e.value_ft:.2f}" for e in elevations],
                textposition="top center",
                textfont=dict(size=10, color="#14532D"),
                name="Elevations",
                customdata=[e.text for e in elevations],
                hovertemplate="%{text} ft<br>%{customdata}<extra></extra>",
            )
        )

    if layers.get("symbols", True) and elevations:
        fig.add_trace(
            go.Scatter(
                x=[e.x for e in elevations],
                y=[e.y for e in elevations],
                mode="markers",
                marker=dict(
                    size=14,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2, color="#F97316"),
                ),
                name="Spot markers",
                hoverinfo="skip",
            )
        )

    shapes = []
    if layers.get("segments", True) and segments is not None:
        for bbox, color, label in (
            (segments.drawing_bbox, "#3B82F6", "drawing"),
            (segments.legend_bbox, "#A855F7", "legend"),
            (segments.title_block_bbox, "#EAB308", "title"),
        ):
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color=color, width=2, dash="dot"),
                    fillcolor="rgba(0,0,0,0)",
                    name=label,
                )
            )

    x_range = [0, w]
    y_range = [h, 0]  # image coordinates
    if focus_drawing and segments is not None:
        x0, y0, x1, y1 = segments.drawing_bbox
        pad = 20
        x_range = [max(0, x0 - pad), min(w, x1 + pad)]
        y_range = [min(h, y1 + pad), max(0, y0 - pad)]

    fig.update_layout(
        shapes=shapes,
        margin=dict(l=0, r=0, t=30, b=0),
        height=720,
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        title=dict(text="Scroll to zoom · drag to pan · toggle layers in legend", font=dict(size=13)),
    )
    fig.update_xaxes(visible=False, range=x_range, fixedrange=False)
    fig.update_yaxes(visible=False, range=y_range, fixedrange=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(uirevision="cv-cut-fill-map")
    return fig
