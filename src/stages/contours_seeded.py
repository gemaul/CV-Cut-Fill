from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .contours_curved import (
    Polyline,
    _annotate_geometry,
    _classify_existing_proposed,
    _classify_polyline,
    _ridge_ink_mask,
    extract_curved_contours,
)
from .elevations_ocr import ElevationCallout


@dataclass
class SeededContourStats:
    seeds_tried: int = 0
    seeds_hit: int = 0
    closed_count: int = 0
    open_count: int = 0
    mean_length_px: float = 0.0
    fallback_count: int = 0


def extract_seeded_contours(
    image_bgr: np.ndarray,
    elevations: list[ElevationCallout],
    *,
    elev_offset_xy: tuple[float, float] = (0.0, 0.0),
    sheet_role: str = "auto",
    allow_mask: np.ndarray | None = None,
    min_length_px: float = 40.0,
    seed_search_px: float = 48.0,
    spot_seed_max_dist_px: float = 14.0,
    close_tol_px: float = 14.0,
    work_max_dim: int = 2200,
    use_fallback: bool = True,
    max_fallback: int = 80,
) -> tuple[list[Polyline], SeededContourStats]:
    """Trace contours by seeding from elevation callouts and following ridge ink.

    Elevations are in full-page coordinates; ``elev_offset_xy`` is the top-left of
    ``image_bgr`` in page space (drawing crop origin). ``allow_mask`` should be
    chrome-only (drawing viewport), never a hard property interior mask.
    """
    stats = SeededContourStats()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ox, oy = elev_offset_xy

    scale = 1.0
    mask_work = None
    if allow_mask is not None:
        if allow_mask.shape[:2] != (h, w):
            raise ValueError("allow_mask must match image_bgr spatial size")
        mask_work = (allow_mask.astype(np.uint8) > 0).astype(np.uint8) * 255

    work_gray = gray
    if max(h, w) > work_max_dim:
        scale = work_max_dim / max(h, w)
        work_gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        if mask_work is not None:
            mask_work = cv2.resize(
                mask_work, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )

    ink = _ridge_ink_mask(work_gray)
    if mask_work is not None:
        ink = cv2.bitwise_and(ink, mask_work)

    # Same ridge bridging for both sheets (V-101 settings also help C-201 continuity)
    close_iters = 4
    close_k = 5
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=close_iters,
    )

    wh, ww = work_gray.shape
    # Only drop tiny speckles — do NOT drop large components. Aggressive morph-close
    # often merges topo into one big blob; an upper area cut would wipe all contour ink.
    num, labels, stats_cc, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cleaned = np.zeros_like(ink)
    min_area = max(6, int(10 * scale * scale))
    for i in range(1, num):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    skel = skeletonize(cleaned > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    skel_pts = set(zip(xs.tolist(), ys.tolist()))

    seeds = _select_seeds(
        elevations,
        skel_pts,
        scale=scale,
        offset_xy=(ox, oy),
        search_px=seed_search_px * scale,
        spot_max_dist=spot_seed_max_dist_px * scale,
        sheet_role=sheet_role,
        image_shape=(wh, ww),
    )
    stats.seeds_tried = len(elevations)
    stats.seeds_hit = len(seeds)

    consumed: set[tuple[int, int]] = set()
    polylines: list[Polyline] = []
    z_by_poly: list[float | None] = []

    # Prefer higher-confidence contour-label seeds first
    seeds.sort(key=lambda s: (0 if s[2] == "existing_match" else 1, s[3]))

    for sx, sy, kind, z, _dist in seeds:
        start = (int(sx), int(sy))
        if start in consumed or start not in skel_pts:
            # snap again to nearest free skeleton pixel
            start2 = _nearest_skel(start, skel_pts, consumed, max_dist=seed_search_px * scale)
            if start2 is None:
                continue
            start = start2

        path = _trace_both_ways(
            start,
            skel_pts,
            consumed,
            close_tol_px=close_tol_px * scale,
        )
        if path is None or len(path) < 2:
            continue

        arr = np.array(path, dtype=np.float32)
        length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
        if length < min_length_px * scale:
            continue

        closed = (
            float(np.linalg.norm(arr[0] - arr[-1])) <= close_tol_px * scale
            and length > 3.0 * close_tol_px * scale
        )
        if closed:
            stats.closed_count += 1
        else:
            stats.open_count += 1

        approx = cv2.approxPolyDP(arr.reshape(-1, 1, 2), epsilon=1.2, closed=closed)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if scale != 1.0:
            pts = pts / scale
            length = length / scale

        poly = Polyline(points=pts, length_px=length, source="seeded")
        _annotate_geometry(poly)
        if _classify_polyline(poly) == "structure" and sheet_role == "auto":
            continue

        if sheet_role == "existing":
            poly.kind = "existing"
        else:
            poly.kind = _classify_existing_proposed(poly, image_bgr)
            # Contour-label seeds prefer existing when dashy; otherwise keep style class
            if kind == "existing_match" and poly.kind == "proposed":
                # integer labels on grading sheets are often existing contour elevs
                if abs(z - round(z)) < 1e-6:
                    poly.kind = "existing"

        polylines.append(poly)
        z_by_poly.append(z)

    # Merge same-Z polylines that share endpoints / overlap
    polylines = _merge_same_z(polylines, z_by_poly)

    if use_fallback:
        fallback = extract_curved_contours(
            image_bgr,
            allow_mask=allow_mask,
            drop_structure=False,
            min_length_px=min_length_px * 1.4,
            max_polylines=max_fallback,
            work_max_dim=work_max_dim,
        )
        # Keep fallback strokes that don't heavily overlap seeded ones
        for poly in fallback:
            if sheet_role == "existing":
                poly.kind = "existing"
            if _is_redundant(poly, polylines):
                continue
            if poly.length_px < min_length_px * 1.5:
                continue
            poly.source = "skeleton_fallback"
            polylines.append(poly)
            stats.fallback_count += 1

    if polylines:
        stats.mean_length_px = float(np.mean([p.length_px for p in polylines]))
    polylines.sort(
        key=lambda p: (p.length_px * (1.0 + 2.0 * p.mean_abs_turn)),
        reverse=True,
    )
    return polylines, stats


def _select_seeds(
    elevations: list[ElevationCallout],
    skel_pts: set[tuple[int, int]],
    *,
    scale: float,
    offset_xy: tuple[float, float],
    search_px: float,
    spot_max_dist: float,
    sheet_role: str,
    image_shape: tuple[int, int],
) -> list[tuple[int, int, str, float, float]]:
    """Return (sx, sy, kind, z, dist) in work-image coordinates."""
    if not skel_pts:
        return []
    ox, oy = offset_xy
    wh, ww = image_shape
    # Grid for fast nearest lookup
    skel_arr = np.array(list(skel_pts), dtype=np.float32)
    seeds: list[tuple[int, int, str, float, float]] = []

    for e in elevations:
        lx = (e.x - ox) * scale
        ly = (e.y - oy) * scale
        if not (0 <= lx < ww and 0 <= ly < wh):
            continue

        # Same generous seed radius as V-101 — spots still need to reach nearby ridge ink
        is_label = e.kind == "existing_match" or (
            sheet_role == "existing" and abs(e.value_ft - round(e.value_ft)) < 1e-6
        )
        max_d = search_px if (is_label or sheet_role in ("existing", "proposed")) else spot_max_dist

        d = np.hypot(skel_arr[:, 0] - lx, skel_arr[:, 1] - ly)
        i = int(np.argmin(d))
        dist = float(d[i])
        if dist > max_d:
            continue
        sx, sy = int(skel_arr[i, 0]), int(skel_arr[i, 1])
        seeds.append((sx, sy, e.kind, float(e.value_ft), dist))
    return seeds


def _nearest_skel(
    pt: tuple[int, int],
    skel_pts: set[tuple[int, int]],
    consumed: set[tuple[int, int]],
    *,
    max_dist: float,
) -> tuple[int, int] | None:
    x0, y0 = pt
    best = None
    best_d = max_dist
    r = int(max_dist) + 1
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            q = (x0 + dx, y0 + dy)
            if q not in skel_pts or q in consumed:
                continue
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best = q
    return best


def _trace_both_ways(
    start: tuple[int, int],
    skel_pts: set[tuple[int, int]],
    consumed: set[tuple[int, int]],
    *,
    close_tol_px: float,
) -> list[tuple[int, int]] | None:
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def walk(origin: tuple[int, int], first_step: tuple[int, int] | None) -> list[tuple[int, int]]:
        path = [origin]
        local_vis = {origin}
        cur = origin
        prev = None
        if first_step is not None:
            path.append(first_step)
            local_vis.add(first_step)
            prev = origin
            cur = first_step
        while True:
            x, y = cur
            cand = [
                (x + dx, y + dy)
                for dx, dy in neighbors
                if (x + dx, y + dy) in skel_pts and (x + dx, y + dy) not in local_vis
            ]
            # Allow stepping onto start to close the loop
            close_hits = [
                (x + dx, y + dy)
                for dx, dy in neighbors
                if (x + dx, y + dy) == origin and len(path) > 8
            ]
            if close_hits and float(np.linalg.norm(np.array(cur) - np.array(origin))) <= close_tol_px * 1.5:
                path.append(origin)
                break
            if not cand:
                break
            if prev is not None:
                vx, vy = cur[0] - prev[0], cur[1] - prev[1]

                def score(n: tuple[int, int]) -> float:
                    return -(vx * (n[0] - cur[0]) + vy * (n[1] - cur[1]))

                cand.sort(key=score)
            nxt = cand[0]
            local_vis.add(nxt)
            path.append(nxt)
            prev = cur
            cur = nxt
            if len(path) > 20000:
                break
        return path

    # First steps from start
    x0, y0 = start
    firsts = [
        (x0 + dx, y0 + dy)
        for dx, dy in neighbors
        if (x0 + dx, y0 + dy) in skel_pts and (x0 + dx, y0 + dy) not in consumed
    ]
    if not firsts:
        return None

    arm_a = walk(start, firsts[0])
    # Second direction: pick a first neighbor not on arm_a early segment
    early = set(arm_a[: min(12, len(arm_a))])
    second = [p for p in firsts[1:] if p not in early]
    if second:
        arm_b = walk(start, second[0])
        # Combine: reverse arm_b (excluding start) + arm_a
        path = list(reversed(arm_b[1:])) + arm_a
    else:
        path = arm_a

    for p in path:
        consumed.add(p)
    return path


def _merge_same_z(
    polylines: list[Polyline], z_by_poly: list[float | None]
) -> list[Polyline]:
    if not polylines:
        return []
    # Group by rounded Z when available
    groups: dict[float | None, list[int]] = {}
    for i, z in enumerate(z_by_poly):
        key = None if z is None else round(float(z), 2)
        groups.setdefault(key, []).append(i)

    kept: list[Polyline] = []
    used: set[int] = set()
    for key, idxs in groups.items():
        if key is None or len(idxs) == 1:
            for i in idxs:
                if i not in used:
                    kept.append(polylines[i])
                    used.add(i)
            continue
        # Keep longest; drop others that share proximity with it
        idxs_sorted = sorted(idxs, key=lambda i: polylines[i].length_px, reverse=True)
        primary = polylines[idxs_sorted[0]]
        kept.append(primary)
        used.add(idxs_sorted[0])
        for i in idxs_sorted[1:]:
            if _is_redundant(polylines[i], [primary]):
                used.add(i)
            elif i not in used:
                kept.append(polylines[i])
                used.add(i)
    for i, poly in enumerate(polylines):
        if i not in used:
            kept.append(poly)
    return kept


def _is_redundant(poly: Polyline, existing: list[Polyline], *, frac: float = 0.45) -> bool:
    if not existing or len(poly.points) < 2:
        return False
    pts = poly.points
    sample = pts[:: max(1, len(pts) // 20)]
    for other in existing:
        if len(other.points) < 2:
            continue
        # Quick bbox reject
        if (
            pts[:, 0].max() < other.points[:, 0].min() - 20
            or pts[:, 0].min() > other.points[:, 0].max() + 20
            or pts[:, 1].max() < other.points[:, 1].min() - 20
            or pts[:, 1].min() > other.points[:, 1].max() + 20
        ):
            continue
        near = 0
        for p in sample:
            d = np.min(np.linalg.norm(other.points - p, axis=1))
            if d < 10.0:
                near += 1
        if near / max(len(sample), 1) >= frac:
            return True
    return False
