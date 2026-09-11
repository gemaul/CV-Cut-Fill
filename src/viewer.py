from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .stages.associate import Association
from .stages.contours_curved import Polyline
from .stages.elevations_ocr import ElevationCallout
from .stages.segment import SheetSegments
from .stages.site_structure import SiteStructure


def build_interactive_figure(
    image_bgr: np.ndarray,
    *,
    polylines: list[Polyline],
    elevations: list[ElevationCallout],
    associations: list[Association],
    segments: SheetSegments | None = None,
    site: SiteStructure | None = None,
    layers: dict[str, bool] | None = None,
    focus_drawing: bool = True,
    max_display_width: int = 1600,
) -> go.Figure:
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

    disp_bgr, scale = _downscale(image_bgr, max_display_width)
    rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
    rgb = (rgb.astype(np.float32) * 0.85).clip(0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]

    def sx(x: float) -> float:
        return float(x) * scale

    def sy(y: float) -> float:
        return float(y) * scale

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, w],
            y=[0, h],
            mode="markers",
            marker=dict(size=1, opacity=0),
            showlegend=False,
            hoverinfo="skip",
            name="anchor",
        )
    )

    if layers.get("property", True) and site is not None:
        prop_strokes = [p for p in (site.property_lines or []) if len(p.points) >= 2]
        if prop_strokes:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for poly in prop_strokes:
                for x, y in poly.points:
                    xs.append(sx(float(x)))
                    ys.append(sy(float(y)))
                xs.append(None)
                ys.append(None)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#EF4444", width=3),
                    name="Property boundary",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    if layers.get("building", True) and site is not None:
        bld_strokes = [p for p in (site.building_lines or []) if len(p.points) >= 2]
        if bld_strokes:
            xs = []
            ys = []
            for poly in bld_strokes:
                for x, y in poly.points:
                    xs.append(sx(float(x)))
                    ys.append(sy(float(y)))
                xs.append(None)
                ys.append(None)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#F97316", width=3),
                    name="Building",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    if layers.get("contours", True) and polylines:
        # Remap unexpected kinds so Plotly never emits a nameless/"undefined" series.
        buckets: dict[str, list[Polyline]] = {
            "existing": [],
            "proposed": [],
        }
        for poly in polylines:
            kind = (poly.kind or "").strip().lower()
            if kind in ("structure", "property", "building"):
                continue
            if kind == "existing":
                buckets["existing"].append(poly)
            else:
                # proposed | topo | anything else → proposed (named)
                buckets["proposed"].append(poly)

        for kind, color, label in (
            ("existing", "#C4B5FD", "Existing contours"),
            ("proposed", "#22D3EE", "Proposed contours"),
        ):
            xs: list[float | None] = []
            ys: list[float | None] = []
            for poly in buckets[kind]:
                if len(poly.points) < 2:
                    continue
                for x, y in poly.points:
                    xs.append(sx(float(x)))
                    ys.append(sy(float(y)))
                xs.append(None)
                ys.append(None)
            if len(xs) > 1:
                fig.add_trace(
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        name=label,
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )

    if layers.get("associations", True) and associations:
        xs, ys = [], []
        for a in associations:
            xs.extend([sx(a.elevation.x), sx(a.point_xy[0]), None])
            ys.extend([sy(a.elevation.y), sy(a.point_xy[1]), None])
        fig.add_trace(
            go.Scattergl(
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
            go.Scattergl(
                x=[sx(e.x) for e in elevations],
                y=[sy(e.y) for e in elevations],
                mode="markers+text",
                marker=dict(size=8, color="#22C55E", line=dict(width=1, color="white")),
                text=[f"{e.value_ft:.2f}" for e in elevations],
                textposition="top center",
                textfont=dict(size=9, color="#14532D"),
                name="Elevations",
                customdata=[[e.text, e.kind] for e in elevations],
                hovertemplate="%{text} ft (%{customdata[1]})<br>%{customdata[0]}<extra></extra>",
                showlegend=True,
            )
        )

    if layers.get("symbols", True) and elevations:
        fig.add_trace(
            go.Scattergl(
                x=[sx(e.x) for e in elevations],
                y=[sy(e.y) for e in elevations],
                mode="markers",
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2, color="#F97316"),
                ),
                name="Spot markers",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    shapes = []
    if layers.get("segments", True) and segments is not None:
        for region in segments.regions:
            # Property/building drawn as polygons above; skip duplicate boxes
            if region.name in ("property", "building"):
                continue
            x0, y0, x1, y1 = region.bbox
            width = 1 if region.role == "topo_zone" else 2
            dash = "dot" if region.role != "topo_zone" else "dash"
            shapes.append(
                dict(
                    type="rect",
                    x0=sx(x0),
                    y0=sy(y0),
                    x1=sx(x1),
                    y1=sy(y1),
                    line=dict(color=region.color, width=width, dash=dash),
                    fillcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    name="",
                )
            )

    x_range = [0, w]
    y_range = [h, 0]
    if focus_drawing and segments is not None:
        focus = segments.drawing_bbox
        if site is not None and site.property_bbox is not None:
            focus = site.property_bbox
        fx0, fy0, fx1, fy1 = focus
        pad = 20 * scale
        x_range = [max(0, sx(fx0) - pad), min(w, sx(fx1) + pad)]
        y_range = [min(h, sy(fy1) + pad), max(0, sy(fy0) - pad)]

    fig.add_layout_image(
        dict(
            source=_rgb_to_jpeg_data_uri(rgb),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            sizing="stretch",
            layer="below",
        )
    )

    fig.update_layout(
        shapes=shapes,
        margin=dict(l=0, r=0, t=48, b=0),
        height=720,
        dragmode="pan",
        # Explicit empty strings — Plotly serializes title=None as {} which
        # renders as the literal text "undefined" in the interactive legend.
        title=dict(text=""),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.9)",
            title=dict(text=""),
            itemsizing="constant",
            traceorder="normal",
        ),
        uirevision="cv-cut-fill-map-v6",
    )
    # Belt-and-suspenders for Streamlit/plotly.js legend title
    fig.update_layout(legend_title_text="")
    fig.layout.title.text = ""
    if fig.layout.legend is not None:
        fig.layout.legend.title.text = ""
    fig.update_xaxes(visible=False, range=x_range, fixedrange=False)
    fig.update_yaxes(
        visible=False, range=y_range, fixedrange=False, scaleanchor="x", scaleratio=1
    )
    return fig


def _downscale(image_bgr: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr, 1.0
    scale = max_width / w
    out = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, scale


def _rgb_to_jpeg_data_uri(rgb: np.ndarray, quality: int = 72) -> str:
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
