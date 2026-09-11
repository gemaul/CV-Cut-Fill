from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import sato
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
    min_length_px: float = 55.0,
    max_polylines: int = 500,
    work_max_dim: int = 1800,
    allow_mask: np.ndarray | None = None,
    drop_structure: bool = True,
) -> list[Polyline]:
    """Extract topographic contours via ridge detection (Sato) + skeleton tracing.

    Adaptive threshold alone misses faint mid-gray contour ink. Sato black-ridge
    filtering targets thin dark strokes, then dash gaps are closed before tracing.

    ``allow_mask`` (H×W bool/uint8, same size as ``image_bgr``) restricts ink to
    property-interior zones after building/property structure has been removed.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1.0
    mask_work = None
    if allow_mask is not None:
        if allow_mask.shape[:2] != (h, w):
            raise ValueError("allow_mask must match image_bgr spatial size")
        mask_work = (allow_mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    if max(h, w) > work_max_dim:
        scale = work_max_dim / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask_work is not None:
            mask_work = cv2.resize(
                mask_work, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
    wh, ww = gray.shape

    ink = _ridge_ink_mask(gray)
    if mask_work is not None:
        ink = cv2.bitwise_and(ink, mask_work)

    # Bridge dashed existing contours
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2,
    )

    # Drop giant filled components (solid hatches)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cleaned = np.zeros_like(ink)
    area_limit = (wh * ww) * 0.04
    min_area = max(10, int(18 * scale * scale))
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= area_limit:
            cleaned[labels == i] = 255

    skel = skeletonize(cleaned > 0).astype(np.uint8) * 255
    polylines = _trace_skeleton(skel, min_length_px=min_length_px * scale)

    if scale != 1.0:
        inv = 1.0 / scale
        for poly in polylines:
            poly.points = poly.points * inv
            poly.length_px = poly.length_px * inv

    kept: list[Polyline] = []
    for poly in polylines:
        _annotate_geometry(poly)
        cls = _classify_polyline(poly)
        if drop_structure and cls == "structure":
            continue
        if cls == "structure":
            poly.kind = "structure"
        else:
            poly.kind = _classify_existing_proposed(poly, image_bgr)
        kept.append(poly)

    # Prefer longer + curvier strokes (site contours over tiny nicks)
    kept.sort(
        key=lambda p: (p.length_px * (1.0 + 2.0 * p.mean_abs_turn)),
        reverse=True,
    )
    return kept[:max_polylines]


def filter_topo_polylines(polylines: list[Polyline]) -> list[Polyline]:
    return [p for p in polylines if p.kind in ("topo", "existing", "proposed")]


def _ridge_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Sato ridges for faint contours, OR'd with a soft adaptive mask for bold ink."""
    g = gray.astype(np.float32) / 255.0
    ridge = sato(g, sigmas=(0.8, 1.2, 1.8, 2.5), black_ridges=True)
    # Keep strongest ridge responses (thin dark lines)
    thr = float(np.percentile(ridge, 90))
    ridge_mask = (ridge >= thr).astype(np.uint8) * 255

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    # Only keep thin dark strokes from adaptive (erode heavy fills)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    return cv2.bitwise_or(ridge_mask, dark)


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


def _classify_polyline(poly: Polyline) -> str:
    pts = poly.points
    if len(pts) < 2:
        return "structure"

    # Strong curvature ⇒ keep as topo even if overall chord is straight-ish
    if poly.mean_abs_turn >= 0.10:
        return "topo"

    v = pts[-1] - pts[0]
    ang = abs(math.degrees(math.atan2(float(v[1]), float(v[0])))) % 180
    axis_aligned = min(ang, abs(90 - ang), abs(180 - ang)) < 10

    # Parking stalls / walls: long, straight, axis-aligned
    if poly.straightness > 0.93 and axis_aligned and poly.length_px > 90:
        return "structure"
    if poly.straightness > 0.985 and poly.length_px > 140:
        return "structure"
    return "topo"


def _classify_existing_proposed(poly: Polyline, image_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pts = poly.points
    if len(pts) < 6:
        return "proposed"

    idxs = np.linspace(0, len(pts) - 1, num=min(50, len(pts))).astype(int)
    ink = []
    for i in idxs:
        x = int(np.clip(pts[i, 0], 0, w - 1))
        y = int(np.clip(pts[i, 1], 0, h - 1))
        patch = gray[max(0, y - 1) : min(h, y + 2), max(0, x - 1) : min(w, x + 2)]
        ink.append(float(patch.mean()) < 185)

    transitions = sum(1 for a, b in zip(ink, ink[1:]) if a != b)
    ink_ratio = sum(ink) / max(len(ink), 1)
    if transitions >= 5 and 0.2 < ink_ratio < 0.88:
        return "existing"
    return "proposed"


def _trace_skeleton(skel: np.ndarray, *, min_length_px: float) -> list[Polyline]:
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    if len(xs) > 200_000:
        step = int(np.ceil(len(xs) / 200_000))
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
        approx = cv2.approxPolyDP(arr.reshape(-1, 1, 2), epsilon=1.1, closed=False)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        polylines.append(Polyline(points=pts, length_px=length))

    return polylines
