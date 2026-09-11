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


def build_interactive_figure(
    image_bgr: np.ndarray,
    *,
    polylines: list[Polyline],
    elevations: list[ElevationCallout],
    associations: list[Association],
    segments: SheetSegments | None = None,
    layers: dict[str, bool] | None = None,
    focus_drawing: bool = True,
    max_display_width: int = 1600,
) -> go.Figure:
    """Zoomable Plotly viewer with toggleable detection layers.

    Uses a compressed JPEG layout image (not go.Image pixel arrays) so layer
    toggles do not re-send tens of MB and crash the browser/websocket.
    """
    layers = {
        "contours": True,
        "elevations": True,
        "symbols": True,
        "associations": True,
        "segments": True,
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
    # Invisible anchor trace so axes exist even with no overlays
    fig.add_trace(
        go.Scatter(
            x=[0, w],
            y=[0, h],
            mode="markers",
            marker=dict(size=1, opacity=0),
            showlegend=False,
            hoverinfo="skip",
            name="_anchor",
        )
    )

    if layers.get("contours", True) and polylines:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for poly in polylines:
            for x, y in poly.points:
                xs.append(sx(x))
                ys.append(sy(y))
            xs.append(None)
            ys.append(None)
        if len(xs) > 1:
            fig.add_trace(
                go.Scattergl(
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
                customdata=[e.text for e in elevations],
                hovertemplate="%{text} ft<br>%{customdata}<extra></extra>",
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
            )
        )

    shapes = []
    if layers.get("segments", True) and segments is not None:
        for bbox, color in (
            (segments.drawing_bbox, "#3B82F6"),
            (segments.legend_bbox, "#A855F7"),
            (segments.title_block_bbox, "#EAB308"),
        ):
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            shapes.append(
                dict(
                    type="rect",
                    x0=sx(x0),
                    y0=sy(y0),
                    x1=sx(x1),
                    y1=sy(y1),
                    line=dict(color=color, width=2, dash="dot"),
                    fillcolor="rgba(0,0,0,0)",
                )
            )

    x_range = [0, w]
    y_range = [h, 0]
    if focus_drawing and segments is not None:
        x0, y0, x1, y1 = segments.drawing_bbox
        pad = 20 * scale
        x_range = [max(0, sx(x0) - pad), min(w, sx(x1) + pad)]
        y_range = [min(h, sy(y1) + pad), max(0, sy(y0) - pad)]

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
        margin=dict(l=0, r=0, t=28, b=0),
        height=720,
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.9)",
        ),
        title=dict(
            text="Scroll to zoom · drag to pan · use layer checkboxes",
            font=dict(size=13),
        ),
        uirevision="cv-cut-fill-map",
    )
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
