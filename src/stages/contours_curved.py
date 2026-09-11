from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Polyline:
    points: np.ndarray  # (N, 2) float xy in pixels
    length_px: float
    source: str = "skeleton"


def extract_curved_contours(
    image_bgr: np.ndarray,
    *,
    min_length_px: float = 80.0,
    max_polylines: int = 400,
    work_max_dim: int = 1600,
) -> list[Polyline]:
    """Extract curved ink strokes as polylines via adaptive threshold + skeleton.

    Runs skeletonization on a downscaled image for speed, then scales polylines
    back to full resolution.
    """
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
    polylines = _trace_skeleton(skel, min_length_px=min_len)
    # Scale back to full-res coordinates
    if scale != 1.0:
        inv = 1.0 / scale
        for poly in polylines:
            poly.points = poly.points * inv
            poly.length_px = poly.length_px * inv

    polylines.sort(key=lambda p: p.length_px, reverse=True)
    return polylines[:max_polylines]


def _trace_skeleton(skel: np.ndarray, *, min_length_px: float) -> list[Polyline]:
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    # Cap skeleton pixels for pathological dense sheets
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
