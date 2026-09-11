from __future__ import annotations

"""Extract property / building strokes from native PDF vector drawings.

Civil PDFs often draw dashed property lines as many short collinear segments.
We map those into page pixels, restrict to the drawing viewport, then chain.
"""

import math

import cv2
import numpy as np
import pymupdf

from .contours_curved import Polyline
from .rasterize import RasterPage


def extract_property_and_building_strokes(
    page: RasterPage,
    *,
    detect_building: bool = True,
    drawing_bbox_px: tuple[int, int, int, int] | None = None,
) -> tuple[list[Polyline], list[Polyline]]:
    """Return (property_lines, building_lines) in page pixel coordinates."""
    doc = pymupdf.open(page.pdf_path)
    pdf_page = doc[page.page_index]
    zoom = page.dpi / 72.0
    rot = pdf_page.rotation_matrix
    bbox = drawing_bbox_px or (0, 0, page.width_px, page.height_px)

    segs_px = _collect_segments_px(pdf_page, zoom, rot, bbox)

    prop_lines = _property_from_segments(segs_px)
    bld_lines: list[Polyline] = []
    if detect_building:
        bld_lines = _building_from_segments(segs_px)

    prop_lines = _keep_longest_outer(prop_lines, min_keep=4)
    bld_lines = sorted(bld_lines, key=lambda p: -p.length_px)[:50]
    return prop_lines, bld_lines


def _collect_segments_px(
    pdf_page: pymupdf.Page,
    zoom: float,
    rot: pymupdf.Matrix,
    bbox: tuple[int, int, int, int],
) -> list[dict]:
    x0, y0, x1, y1 = bbox
    pad = 8
    segs: list[dict] = []
    for d in pdf_page.get_drawings():
        color = d.get("color")
        if color is None:
            continue
        if float(min(color)) > 0.55:
            continue
        width = float(d.get("width") or 0.0)
        for it in d.get("items") or []:
            if it[0] != "l":
                continue
            p1, p2 = it[1], it[2]
            length_pdf = float(math.hypot(p1.x - p2.x, p1.y - p2.y))
            if length_pdf < 0.8:
                continue
            a = _pdf_to_px(np.array([p1.x, p1.y]), zoom, rot)
            b = _pdf_to_px(np.array([p2.x, p2.y]), zoom, rot)
            mid = (a + b) * 0.5
            if not (
                x0 - pad <= mid[0] <= x1 + pad and y0 - pad <= mid[1] <= y1 + pad
            ):
                continue
            length = float(np.linalg.norm(b - a))
            if length < 1.2:
                continue
            segs.append(
                {
                    "p1": a.astype(np.float64),
                    "p2": b.astype(np.float64),
                    "len": length,
                    "w": width,
                    "color": tuple(float(c) for c in color),
                    "ang": float(np.degrees(np.atan2(b[1] - a[1], b[0] - a[0])) % 180.0),
                    "used": False,
                }
            )
    return segs


def _property_from_segments(segs: list[dict]) -> list[Polyline]:
    """Black / near-black dashed chains with long-short length variation."""
    cand = [
        s
        for s in segs
        if s["color"][0] <= 0.15
        and s["w"] <= 1.4
        and s["len"] <= 90.0
    ]
    chains = _chain_collinear(cand, ang_tol=7.0, dist_tol=3.5, gap_max=40.0)
    out: list[Polyline] = []
    for members in chains:
        score = _property_chain_score(members)
        if score < 2.2:
            continue
        a, b, length, _ang = _chain_endpoints(members)
        if length < 60.0:
            continue
        out.append(_segment_polyline(a, b, kind="property", length=length))
    out.sort(key=lambda p: -p.length_px)
    return out[:40]


def _building_from_segments(segs: list[dict]) -> list[Polyline]:
    """Thick dark continuous strokes — wall candidates."""
    cand = [
        s
        for s in segs
        if s["color"][0] <= 0.2 and s["w"] >= 1.2 and s["len"] >= 10.0
    ]
    chains = _chain_collinear(cand, ang_tol=6.0, dist_tol=4.0, gap_max=10.0)
    out: list[Polyline] = []
    for members in chains:
        a, b, length, ang = _chain_endpoints(members)
        if length < 45.0:
            continue
        axis = min(_ang_diff(ang, 0.0), _ang_diff(ang, 90.0))
        if axis > 15.0:
            continue
        out.append(_segment_polyline(a, b, kind="building", length=length))
    out.sort(key=lambda p: -p.length_px)
    return out


def _property_chain_score(members: list[dict]) -> float:
    if len(members) < 3:
        return 0.0
    lengths = np.array([m["len"] for m in members], dtype=np.float64)
    med = float(np.median(lengths))
    if med <= 0:
        return 0.0
    cv = float(lengths.std()) / max(float(lengths.mean()), 1e-3)
    shorts = lengths[lengths <= med * 1.3]
    longs = lengths[lengths >= max(med * 2.0, med + 4.0)]
    if len(longs) < 1 or len(shorts) < 2:
        shorts = lengths[lengths <= 12.0]
        longs = lengths[lengths >= 18.0]
    if len(longs) < 1 or len(shorts) < 2:
        # Still allow multi-gap dashed black chains (uniform ticks less likely black)
        if len(members) >= 6 and cv > 0.15:
            return 2.0 + min(1.0, len(members) / 20.0)
        return 0.0
    ratio = float(np.median(longs) / max(float(np.median(shorts)), 0.1))
    score = 0.0
    if 2.0 <= ratio <= 10.0:
        score += 2.0
    else:
        return 0.0
    if len(members) >= 4:
        score += 0.8
    if cv >= 0.25:
        score += 0.5
    return score


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _chain_collinear(
    segments: list[dict],
    *,
    ang_tol: float,
    dist_tol: float,
    gap_max: float,
) -> list[list[dict]]:
    for s in segments:
        s["used"] = False
    chains: list[list[dict]] = []
    for seed in segments:
        if seed["used"]:
            continue
        seed["used"] = True
        members = [seed]
        changed = True
        while changed:
            changed = False
            mean_ang = float(np.median([m["ang"] for m in members]))
            rad = np.radians(mean_ang)
            direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            pts = np.array([p for m in members for p in (m["p1"], m["p2"])])
            origin = pts.mean(axis=0)
            ts = [float(np.dot(p - origin, direction)) for p in pts]
            tmin, tmax = min(ts), max(ts)
            for other in segments:
                if other["used"]:
                    continue
                if _ang_diff(other["ang"], mean_ang) > ang_tol:
                    continue
                lat = max(
                    abs(float(np.dot(other["p1"] - origin, normal))),
                    abs(float(np.dot(other["p2"] - origin, normal))),
                )
                if lat > dist_tol:
                    continue
                t1 = float(np.dot(other["p1"] - origin, direction))
                t2 = float(np.dot(other["p2"] - origin, direction))
                otmin, otmax = min(t1, t2), max(t1, t2)
                if otmax < tmin - gap_max or otmin > tmax + gap_max:
                    continue
                other["used"] = True
                members.append(other)
                changed = True
                break
        if sum(m["len"] for m in members) >= 40.0:
            chains.append(members)
    return chains


def _chain_endpoints(
    members: list[dict],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean_ang = float(np.median([m["ang"] for m in members]))
    rad = np.radians(mean_ang)
    direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
    pts = np.array([p for m in members for p in (m["p1"], m["p2"])], dtype=np.float64)
    origin = pts.mean(axis=0)
    ts = np.array([float(np.dot(p - origin, direction)) for p in pts])
    order = np.argsort(ts)
    a = origin + direction * ts[order[0]]
    b = origin + direction * ts[order[-1]]
    return a, b, float(ts[order[-1]] - ts[order[0]]), mean_ang


def _pdf_to_px(
    pt: np.ndarray, zoom: float, rot: pymupdf.Matrix
) -> np.ndarray:
    p = pymupdf.Point(float(pt[0]), float(pt[1])) * rot
    return np.array([p.x * zoom, p.y * zoom], dtype=np.float32)


def _segment_polyline(
    a: np.ndarray, b: np.ndarray, *, kind: str, length: float
) -> Polyline:
    n = max(8, int(length // 10))
    xs = np.linspace(float(a[0]), float(b[0]), n)
    ys = np.linspace(float(a[1]), float(b[1]), n)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    return Polyline(
        points=pts,
        length_px=float(length),
        source="pdf_vector",
        kind=kind,
        straightness=1.0,
        mean_abs_turn=0.0,
    )


def _keep_longest_outer(lines: list[Polyline], *, min_keep: int) -> list[Polyline]:
    if len(lines) <= min_keep:
        return sorted(lines, key=lambda p: -p.length_px)
    lines = sorted(lines, key=lambda p: -p.length_px)
    pts = np.vstack([p.points[[0, -1]] for p in lines]).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    kept: list[Polyline] = []
    for poly in lines:
        mid = poly.points[len(poly.points) // 2]
        d = cv2.pointPolygonTest(
            hull.reshape(-1, 1, 2), (float(mid[0]), float(mid[1])), True
        )
        if d >= -60.0 or poly.length_px >= 280:
            kept.append(poly)
        if len(kept) >= 28:
            break
    return kept or lines[:min_keep]
