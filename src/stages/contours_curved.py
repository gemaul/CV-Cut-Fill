from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Polyline:
    points: np.ndarray  # (N, 2) float xy in pixels
    length_px: float
    source: str = "skeleton"
    kind: str = "topo"  # topo | structure | existing | proposed
    straightness: float = 0.0
    mean_abs_turn: float = 0.0


def extract_curved_contours(
    image_bgr: np.ndarray,
    *,
    min_length_px: float = 80.0,
    max_polylines: int = 400,
    work_max_dim: int = 1600,
) -> list[Polyline]:
    """Extract ink strokes, then keep topo-like curves (drop walls/stalls)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1.0
    if max(h, w) > work_max_dim:
        scale = work_max_dim / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    wh, ww = gray.shape

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)
    area_limit = (wh * ww) * 0.02
    min_area = max(20, int(40 * scale * scale))
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= area_limit:
            cleaned[labels == i] = 255

    thin = cv2.erode(cleaned, kernel, iterations=1)
    skel = skeletonize(thin > 0).astype(np.uint8) * 255

    min_len = min_length_px * scale
    polylines = _trace_skeleton(skel, min_length_px=min_len, gray=gray)

    if scale != 1.0:
        inv = 1.0 / scale
        for poly in polylines:
            poly.points = poly.points * inv
            poly.length_px = poly.length_px * inv

    # Classify + filter architectural junk
    kept: list[Polyline] = []
    for poly in polylines:
        _annotate_geometry(poly)
        poly.kind = _classify_polyline(poly, gray_full=None, scale=1.0)
        if poly.kind == "structure":
            continue
        kept.append(poly)

    # Dash vs solid on full-res crop using local gray sampling
    for poly in kept:
        poly.kind = _classify_existing_proposed(poly, image_bgr)

    kept.sort(key=lambda p: p.length_px, reverse=True)
    return kept[:max_polylines]


def filter_topo_polylines(polylines: list[Polyline]) -> list[Polyline]:
    return [p for p in polylines if p.kind in ("topo", "existing", "proposed")]


def _annotate_geometry(poly: Polyline) -> None:
    pts = poly.points
    if len(pts) < 2:
        poly.straightness = 1.0
        poly.mean_abs_turn = 0.0
        return
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    poly.straightness = chord / max(poly.length_px, 1e-3)

    turns = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        a1 = math.atan2(float(v1[1]), float(v1[0]))
        a2 = math.atan2(float(v2[1]), float(v2[0]))
        d = abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi)
        turns.append(d)
    poly.mean_abs_turn = float(np.mean(turns)) if turns else 0.0


def _classify_polyline(poly: Polyline, gray_full: np.ndarray | None, scale: float) -> str:
    """Reject long axis-aligned nearly-straight runs (walls, stalls, grids)."""
    pts = poly.points
    if len(pts) < 2:
        return "structure"

    # Dominant direction
    v = pts[-1] - pts[0]
    ang = abs(math.degrees(math.atan2(float(v[1]), float(v[0])))) % 180
    axis_aligned = min(ang, abs(90 - ang), abs(180 - ang)) < 12

    if poly.straightness > 0.92 and axis_aligned and poly.length_px > 120:
        return "structure"
    if poly.straightness > 0.97 and poly.length_px > 200:
        return "structure"
    # Short very straight ticks
    if poly.straightness > 0.98 and poly.mean_abs_turn < 0.05:
        return "structure"
    return "topo"


def _classify_existing_proposed(poly: Polyline, image_bgr: np.ndarray) -> str:
    """Heuristic dash detection: sample along polyline for ink gaps → existing."""
    if poly.kind == "structure":
        return "structure"
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pts = poly.points
    if len(pts) < 4:
        return "proposed"

    # Sample ~40 points along path
    idxs = np.linspace(0, len(pts) - 1, num=min(40, len(pts))).astype(int)
    ink = []
    for i in idxs:
        x = int(np.clip(pts[i, 0], 0, w - 1))
        y = int(np.clip(pts[i, 1], 0, h - 1))
        # local darkness
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        patch = gray[y0:y1, x0:x1]
        ink.append(float(patch.mean()) < 170)

    if len(ink) < 8:
        return "proposed"
    # Transitions between ink/no-ink suggest dashes
    transitions = sum(1 for a, b in zip(ink, ink[1:]) if a != b)
    ink_ratio = sum(ink) / len(ink)
    if transitions >= 6 and 0.25 < ink_ratio < 0.85:
        return "existing"
    return "proposed"


def _trace_skeleton(
    skel: np.ndarray, *, min_length_px: float, gray: np.ndarray
) -> list[Polyline]:
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    if len(xs) > 120_000:
        step = int(np.ceil(len(xs) / 120_000))
        xs = xs[::step]
        ys = ys[::step]

    points = set(zip(xs.tolist(), ys.tolist()))
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def degree(pt: tuple[int, int]) -> int:
        x, y = pt
        return sum((x + dx, y + dy) in points for dx, dy in neighbors)

    endpoints = [p for p in points if degree(p) == 1]
    visited: set[tuple[int, int]] = set()
    polylines: list[Polyline] = []
    seeds = endpoints + [p for p in list(points) if p not in endpoints]

    for start in seeds:
        if start in visited or start not in points:
            continue
        path = [start]
        visited.add(start)
        cur = start
        while True:
            x, y = cur
            nxts = [
                (x + dx, y + dy)
                for dx, dy in neighbors
                if (x + dx, y + dy) in points and (x + dx, y + dy) not in visited
            ]
            if not nxts:
                break
            if len(path) >= 2:
                prev = path[-2]
                vx, vy = cur[0] - prev[0], cur[1] - prev[1]

                def score(n: tuple[int, int]) -> float:
                    return -(vx * (n[0] - cur[0]) + vy * (n[1] - cur[1]))

                nxts.sort(key=score)
            cur = nxts[0]
            visited.add(cur)
            path.append(cur)

        if len(path) < 2:
            continue
        arr = np.array(path, dtype=np.float32)
        length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
        if length < min_length_px:
            continue
        approx = cv2.approxPolyDP(arr.reshape(-1, 1, 2), epsilon=1.5, closed=False)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        polylines.append(Polyline(points=pts, length_px=length))

    return polylines
