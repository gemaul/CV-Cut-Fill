from __future__ import annotations

"""Pre-seeded line-style templates from the MAP LEGEND.

User-provided reference styles:
  - EXISTING CONTOUR: dashed gray line with embedded elevation number
  - PROPOSED CONTOUR: solid black line with embedded elevation number
  - PROPERTY LINE: long-dash / short / short (dash-dot-dot) — NOT a contour
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .contours_curved import Polyline
from .legend import LegendEntry

_STYLE_DIR = Path(__file__).resolve().parents[2] / "samples" / "styles"


@dataclass
class LineStyleTemplate:
    name: str  # existing_contour | proposed_contour | property_line
    dash_signature: str  # solid | dashed | dash_dot_dot
    ink_ratio_range: tuple[float, float]
    transition_range: tuple[float, float]  # transitions per sample point
    mean_gray_max: float  # darker = lower
    notes: str = ""


def load_builtin_templates() -> dict[str, LineStyleTemplate]:
    """Templates derived from legend swatches (+ defaults if files missing)."""
    templates: dict[str, LineStyleTemplate] = {
        "existing_contour": LineStyleTemplate(
            name="existing_contour",
            dash_signature="dashed",
            ink_ratio_range=(0.25, 0.78),
            transition_range=(0.12, 0.55),
            mean_gray_max=200.0,
            notes="Dashed gray with elevation break (EXISTING CONTOUR)",
        ),
        "proposed_contour": LineStyleTemplate(
            name="proposed_contour",
            dash_signature="solid",
            ink_ratio_range=(0.72, 1.0),
            transition_range=(0.0, 0.12),
            mean_gray_max=170.0,
            notes="Solid dark with elevation break (PROPOSED CONTOUR)",
        ),
        "property_line": LineStyleTemplate(
            name="property_line",
            dash_signature="dash_dot_dot",
            ink_ratio_range=(0.35, 0.82),
            transition_range=(0.10, 0.45),
            mean_gray_max=160.0,
            notes="Long-dash short short (PROPERTY LINE) — exclude from contours",
        ),
    }

    # Refine from on-disk user screenshots when present
    for key, filename in (
        ("existing_contour", "existing_contour.png"),
        ("proposed_contour", "proposed_contour.png"),
        ("property_line", "property_line.png"),
    ):
        path = _STYLE_DIR / filename
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        sig = _profile_from_swatch(img)
        if sig is None:
            continue
        dash, ink_ratio, trans_rate, mean_gray = sig
        # Number breaks in contour swatches must not flip proposed→dashed
        if key == "proposed_contour":
            dash = "solid"
            ink_lo, ink_hi = 0.55, 1.0
            tr_lo, tr_hi = 0.0, 0.14
        elif key == "existing_contour":
            dash = "dashed"
            ink_lo, ink_hi = max(0.15, ink_ratio - 0.25), min(1.0, ink_ratio + 0.25)
            tr_lo, tr_hi = max(0.08, trans_rate - 0.1), min(0.8, trans_rate + 0.2)
        else:
            dash = "dash_dot_dot"
            ink_lo, ink_hi = max(0.25, ink_ratio - 0.25), min(1.0, ink_ratio + 0.25)
            tr_lo, tr_hi = max(0.05, trans_rate - 0.12), min(0.8, trans_rate + 0.18)
        t = templates[key]
        templates[key] = LineStyleTemplate(
            name=t.name,
            dash_signature=dash,
            ink_ratio_range=(ink_lo, ink_hi),
            transition_range=(tr_lo, tr_hi),
            mean_gray_max=max(mean_gray + 40.0, t.mean_gray_max),
            notes=t.notes + f" [from {filename}]",
        )
    return templates


def refine_templates_from_legend(
    templates: dict[str, LineStyleTemplate],
    legend: list[LegendEntry],
) -> dict[str, LineStyleTemplate]:
    """Optionally tighten templates using cropped MAP LEGEND icons."""
    out = dict(templates)
    for e in legend:
        if e.icon_rgb is None:
            continue
        u = e.name.upper()
        key = None
        if "EXISTING CONTOUR" in u:
            key = "existing_contour"
        elif "PROPOSED CONTOUR" in u:
            key = "proposed_contour"
        elif "PROPERTY" in u and "LINE" in u:
            key = "property_line"
        if key is None or key not in out:
            continue
        bgr = cv2.cvtColor(e.icon_rgb, cv2.COLOR_RGB2BGR)
        sig = _profile_from_swatch(bgr)
        if sig is None:
            continue
        dash, ink_ratio, trans_rate, mean_gray = sig
        base = out[key]
        out[key] = LineStyleTemplate(
            name=base.name,
            dash_signature=dash or base.dash_signature,
            ink_ratio_range=(
                max(0.05, min(base.ink_ratio_range[0], ink_ratio - 0.2)),
                min(1.0, max(base.ink_ratio_range[1], ink_ratio + 0.2)),
            ),
            transition_range=(
                max(0.0, min(base.transition_range[0], trans_rate - 0.1)),
                min(0.8, max(base.transition_range[1], trans_rate + 0.15)),
            ),
            mean_gray_max=max(base.mean_gray_max, mean_gray + 35.0),
            notes=base.notes + " +legend icon",
        )
    return out


def classify_stroke_style(
    poly: Polyline,
    image_bgr: np.ndarray,
    templates: dict[str, LineStyleTemplate] | None = None,
) -> str:
    """Return existing_contour | proposed_contour | property_line | other."""
    templates = templates or load_builtin_templates()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    profile = _sample_polyline_profile(poly, gray)
    if profile is None:
        return "other"
    ink_ratio, trans_rate, mean_gray, run_pattern = profile

    scores: dict[str, float] = {}
    for key, t in templates.items():
        score = 0.0
        lo, hi = t.ink_ratio_range
        if lo <= ink_ratio <= hi:
            score += 1.5
        else:
            score -= abs(ink_ratio - np.clip(ink_ratio, lo, hi)) * 3.0

        tlo, thi = t.transition_range
        if tlo <= trans_rate <= thi:
            score += 1.5
        else:
            score -= abs(trans_rate - np.clip(trans_rate, tlo, thi)) * 4.0

        if mean_gray <= t.mean_gray_max:
            score += 0.5
        else:
            score -= 0.5

        # Pattern prior
        if t.dash_signature == "solid" and run_pattern == "solid":
            score += 1.2
        elif t.dash_signature == "dashed" and run_pattern == "dashed":
            score += 1.2
        elif t.dash_signature == "dash_dot_dot" and run_pattern == "dash_dot_dot":
            score += 2.0
        elif t.dash_signature == "dash_dot_dot" and run_pattern == "dashed":
            score += 0.3

        # Property lines are typically straighter / longer
        if key == "property_line":
            if poly.straightness > 0.92 and poly.length_px > 80:
                score += 0.8
            if poly.mean_abs_turn > 0.14:
                score -= 1.0
        if key in ("existing_contour", "proposed_contour"):
            if poly.mean_abs_turn >= 0.06 or poly.straightness < 0.98:
                score += 0.4

        scores[key] = score

    best = max(scores, key=scores.get)
    if scores[best] < 1.2:
        return "other"
    return best


def apply_style_kinds(
    polylines: list[Polyline],
    image_bgr: np.ndarray,
    *,
    templates: dict[str, LineStyleTemplate] | None = None,
    sheet_role: str = "auto",
) -> list[Polyline]:
    """Set polyline.kind from legend style templates; drop property-line strokes."""
    templates = templates or load_builtin_templates()
    out: list[Polyline] = []
    for poly in polylines:
        style = classify_stroke_style(poly, image_bgr, templates)
        if style == "property_line":
            # Keep as property for overlays if needed, but not as topo contour
            poly.kind = "property"
            continue
        if sheet_role == "existing":
            poly.kind = "existing"
        elif style == "existing_contour":
            poly.kind = "existing"
        elif style == "proposed_contour":
            poly.kind = "proposed"
        elif poly.kind not in ("existing", "proposed"):
            poly.kind = "proposed" if sheet_role != "existing" else "existing"
        out.append(poly)
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _profile_from_swatch(
    bgr: np.ndarray,
) -> tuple[str, float, float, float] | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Crop to content (drop white padding) so legend screenshots profile cleanly
    ink_mask = gray < 210
    if not ink_mask.any():
        return None
    ys, xs = np.where(ink_mask)
    y0, y1 = max(0, ys.min() - 2), min(gray.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(gray.shape[1], xs.max() + 3)
    gray = gray[y0:y1, x0:x1]
    h, w = gray.shape
    if h < 4 or w < 8:
        return None
    y_a, y_b = int(h * 0.35), int(h * 0.65) + 1
    band = gray[y_a:y_b, :]
    row = band.mean(axis=0)
    ink = row < 200
    if ink.mean() < 0.05:
        ink = row < 220
    if not ink.any():
        return None
    ink_ratio = float(ink.mean())
    transitions = float(np.sum(ink[1:] != ink[:-1])) / max(len(ink) - 1, 1)
    mean_gray = float(row[ink].mean()) if ink.any() else float(row.mean())
    dash = _runs_to_signature(ink)
    return dash, ink_ratio, transitions, mean_gray


def _runs_to_signature(ink: np.ndarray) -> str:
    runs: list[tuple[bool, int]] = []
    cur = bool(ink[0])
    n = 1
    for v in ink[1:]:
        v = bool(v)
        if v == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur = v
            n = 1
    runs.append((cur, n))
    ink_runs = [length for is_ink, length in runs if is_ink and length >= 2]
    gap_runs = [length for is_ink, length in runs if (not is_ink) and length >= 2]
    if len(ink_runs) <= 1:
        return "solid"
    # A solid contour with an elevation number has ~2 long ink runs and one wide gap
    if len(ink_runs) == 2 and len(gap_runs) <= 2:
        return "solid"
    lengths = np.array(ink_runs, dtype=np.float64)
    med = float(np.median(lengths))
    if med <= 0:
        return "dashed"
    rel = lengths / med
    longs = np.sum(rel >= 1.6)
    shorts = np.sum(rel <= 0.75)
    if longs >= 1 and shorts >= 2 and len(ink_runs) >= 3:
        return "dash_dot_dot"
    if len(ink_runs) >= 3:
        return "dashed"
    return "dashed"


def transitions_from_runs(runs: list[tuple[bool, int]]) -> int:
    return max(0, len(runs) - 1)


def extract_style_line_strokes(
    image_bgr: np.ndarray,
    *,
    allow_mask: np.ndarray | None = None,
    min_length_px: float = 90.0,
) -> dict[str, list[Polyline]]:
    """Trace ink with LSD, chain collinear dashes, classify by legend style.

    Returns polylines that follow drawn strokes (not text-anchor hulls):
      - ``property_line``: dash-dot-dot PROPERTY LINE
      - ``building_line``: long solid near-orthogonal wall candidates
      - ``other``: unused (omitted)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if allow_mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = allow_mask.astype(bool)
        if mask.shape != (h, w):
            raise ValueError("allow_mask must match image size")

    ink = ((gray < 155) & mask).astype(np.uint8) * 255
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw = lsd.detect(ink)[0]
    if raw is None:
        return {"property_line": [], "building_line": []}

    segments: list[dict] = []
    for ln in raw.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, ln)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 8:
            continue
        ang = float(np.degrees(np.atan2(y2 - y1, x2 - x1)) % 180.0)
        segments.append(
            {
                "p1": np.array([x1, y1], dtype=np.float64),
                "p2": np.array([x2, y2], dtype=np.float64),
                "len": length,
                "ang": ang,
                "used": False,
            }
        )

    chains = _chain_collinear_segments(segments)
    property_lines: list[Polyline] = []
    building_lines: list[Polyline] = []
    margin = 28

    for members in chains:
        a, b, clen, ang = _chain_endpoints(members)
        if clen < min_length_px:
            continue
        # Skip sheet-frame strokes glued to the drawing crop edge
        if _segment_near_border(a, b, w, h, margin=margin):
            continue

        prop_score = _property_pattern_score(gray, a, b)
        solid_score = _solid_wall_score(gray, a, b, ang)

        if prop_score >= 2.4:
            property_lines.append(
                _snap_segment_to_ink(gray, a, b, kind="property", length=clen)
            )
        elif solid_score >= 2.4:
            building_lines.append(
                _snap_segment_to_ink(gray, a, b, kind="building", length=clen)
            )

    property_lines = _keep_outer_property_ring(property_lines)
    property_lines.sort(key=lambda p: p.length_px, reverse=True)
    building_lines.sort(key=lambda p: p.length_px, reverse=True)
    return {
        "property_line": property_lines[:40],
        "building_line": building_lines[:60],
    }


def _segment_polyline(
    a: np.ndarray, b: np.ndarray, *, kind: str, length: float
) -> Polyline:
    n = max(8, int(length // 12))
    xs = np.linspace(float(a[0]), float(b[0]), n)
    ys = np.linspace(float(a[1]), float(b[1]), n)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    chord = float(np.linalg.norm(b - a))
    return Polyline(
        points=pts,
        length_px=float(length),
        source="lsd_chain",
        kind=kind,
        straightness=chord / max(length, 1e-3),
        mean_abs_turn=0.0,
    )


def _snap_segment_to_ink(
    gray: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    kind: str,
    length: float,
    search: int = 3,
) -> Polyline:
    """Densify the segment and snap each sample onto the darkest nearby ink."""
    h, w = gray.shape
    n = max(12, int(length // 6))
    xs = np.linspace(float(a[0]), float(b[0]), n)
    ys = np.linspace(float(a[1]), float(b[1]), n)
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return _segment_polyline(a, b, kind=kind, length=length)
    tangent = direction / norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    snapped: list[list[float]] = []
    for x, y in zip(xs, ys):
        best = (x, y)
        best_g = 255.0
        for k in range(-search, search + 1):
            px = x + normal[0] * k
            py = y + normal[1] * k
            xi = int(np.clip(round(px), 0, w - 1))
            yi = int(np.clip(round(py), 0, h - 1))
            g = float(gray[yi, xi])
            if g < best_g:
                best_g = g
                best = (float(xi), float(yi))
        # Only accept snaps that actually hit ink
        if best_g < 175:
            snapped.append([best[0], best[1]])
        else:
            snapped.append([x, y])
    pts = np.array(snapped, dtype=np.float32)
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    # Recompute length along path
    seglen = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return Polyline(
        points=pts,
        length_px=max(seglen, chord),
        source="lsd_chain_snapped",
        kind=kind,
        straightness=chord / max(seglen, 1e-3),
        mean_abs_turn=0.0,
    )


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _project(pt: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(pt - origin, direction))


def _chain_collinear_segments(
    segments: list[dict],
    *,
    ang_tol: float = 7.0,
    dist_tol: float = 5.0,
    gap_max: float = 48.0,
) -> list[list[dict]]:
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
            ts = [_project(p, origin, direction) for p in pts]
            tmin, tmax = min(ts), max(ts)
            for other in segments:
                if other["used"]:
                    continue
                if _ang_diff(other["ang"], mean_ang) > ang_tol:
                    continue
                lat = max(
                    abs(_project(other["p1"], origin, normal)),
                    abs(_project(other["p2"], origin, normal)),
                )
                if lat > dist_tol:
                    continue
                t1 = _project(other["p1"], origin, direction)
                t2 = _project(other["p2"], origin, direction)
                otmin, otmax = min(t1, t2), max(t1, t2)
                if otmax < tmin - gap_max or otmin > tmax + gap_max:
                    continue
                other["used"] = True
                members.append(other)
                changed = True
                break
        if sum(m["len"] for m in members) >= 60:
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
    ts = np.array([_project(p, origin, direction) for p in pts])
    order = np.argsort(ts)
    a = origin + direction * ts[order[0]]
    b = origin + direction * ts[order[-1]]
    return a, b, float(ts[order[-1]] - ts[order[0]]), mean_ang


def _segment_near_border(
    a: np.ndarray, b: np.ndarray, w: int, h: int, *, margin: int
) -> bool:
    pts = np.vstack([a, b])
    return bool(
        ((pts[:, 0] < margin) | (pts[:, 0] > w - margin) | (pts[:, 1] < margin) | (pts[:, 1] > h - margin)).any()
    )


def _sample_segment_ink(
    gray: np.ndarray, a: np.ndarray, b: np.ndarray, *, step: float = 1.2
) -> np.ndarray:
    length = float(np.linalg.norm(b - a))
    n = max(50, int(length / step))
    xs = np.linspace(float(a[0]), float(b[0]), n)
    ys = np.linspace(float(a[1]), float(b[1]), n)
    h, w = gray.shape
    flags: list[bool] = []
    for x, y in zip(xs, ys):
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        patch = gray[max(0, yi - 1) : min(h, yi + 2), max(0, xi - 1) : min(w, xi + 2)]
        flags.append(float(patch.min()) < 165.0)
    return np.array(flags, dtype=bool)


def _ink_runs(ink_arr: np.ndarray) -> list[tuple[bool, int]]:
    runs: list[tuple[bool, int]] = []
    cur = bool(ink_arr[0])
    n = 1
    for v in ink_arr[1:]:
        v = bool(v)
        if v == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur = v
            n = 1
    runs.append((cur, n))
    return runs


def _property_pattern_score(gray: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Score long-dash / short / short (PROPERTY LINE) along a traced segment."""
    ink_arr = _sample_segment_ink(gray, a, b)
    if ink_arr.size < 40:
        return 0.0
    ir = float(ink_arr.mean())
    runs = _ink_runs(ink_arr)
    if runs and not runs[0][0]:
        runs = runs[1:]
    if runs and not runs[-1][0]:
        runs = runs[:-1]
    ink_runs = [L for is_ink, L in runs if is_ink]
    gap_runs = [L for is_ink, L in runs if not is_ink]
    if len(ink_runs) < 3 or len(gap_runs) < 2:
        return 0.0

    lengths = np.array(ink_runs, dtype=np.float64)
    mn = float(np.percentile(lengths, 25))
    if mn < 1:
        return 0.0
    rel = lengths / mn
    n_long = int(np.sum(rel >= 2.8))
    n_short = int(np.sum(rel <= 1.7))
    # Uniform dashed contours (existing) → reject
    cv = float(lengths.std()) / max(float(lengths.mean()), 1.0)
    if n_long == 0 or cv < 0.35:
        return 0.0

    shorts = lengths[rel <= 1.7]
    longs = lengths[rel >= 2.8]
    if len(shorts) < 2 or len(longs) < 1:
        return 0.0
    ratio = float(np.median(longs) / max(float(np.median(shorts)), 1.0))
    # Legend PROPERTY LINE template is ~4× (long vs short)
    if not (2.6 <= ratio <= 6.5):
        return 0.0

    score = 1.5
    if 0.30 <= ir <= 0.75:
        score += 1.0
    elif 0.25 <= ir <= 0.82:
        score += 0.35
    else:
        return 0.0
    # Mean gray on ink samples only (gaps are white and would inflate the mean)
    xs = np.linspace(float(a[0]), float(b[0]), 48)
    ys = np.linspace(float(a[1]), float(b[1]), 48)
    h, w = gray.shape
    dark = []
    for x, y in zip(xs, ys):
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        g = float(gray[yi, xi])
        if g < 170:
            dark.append(g)
    if len(dark) < 6:
        return 0.0
    mean_g = float(np.mean(dark))
    # Reject very light gray contour ink; keep black property strokes
    if mean_g > 155:
        return 0.0
    if mean_g < 110:
        score += 0.5
    if gap_runs:
        g = np.array(gap_runs, dtype=np.float64)
        gmean = float(g.mean())
        if 2.0 <= gmean <= 40.0 and float(g.std()) / max(gmean, 1.0) < 1.1:
            score += 0.5
        else:
            score -= 0.3
    if len(ink_runs) >= 5:
        score += 0.4
    if 3.0 <= ratio <= 5.5:
        score += 0.6
    return score


def _solid_wall_score(
    gray: np.ndarray, a: np.ndarray, b: np.ndarray, ang: float
) -> float:
    """Score long solid orthognal strokes as building-wall candidates."""
    length = float(np.linalg.norm(b - a))
    if length < 140:
        return 0.0
    # Prefer axis-aligned walls
    axis = min(_ang_diff(ang, 0.0), _ang_diff(ang, 90.0))
    if axis > 10.0:
        return 0.0
    ink_arr = _sample_segment_ink(gray, a, b, step=1.5)
    ir = float(ink_arr.mean())
    if ir < 0.88:
        return 0.0
    # Walls are dark — reject mid-gray parking stripes / light screening
    h, w = gray.shape
    xs = np.linspace(float(a[0]), float(b[0]), 40)
    ys = np.linspace(float(a[1]), float(b[1]), 40)
    samples = []
    for x, y in zip(xs, ys):
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        samples.append(float(gray[yi, xi]))
    mean_g = float(np.mean(samples))
    if mean_g > 90:
        return 0.0
    runs = _ink_runs(ink_arr)
    ink_runs = [L for is_ink, L in runs if is_ink]
    if not ink_runs:
        return 0.0
    longest = max(ink_runs)
    if longest < 0.75 * len(ink_arr):
        return 0.0
    score = 1.6 + min(1.0, length / 400.0)
    if axis <= 5.0:
        score += 0.5
    if ir >= 0.94 and mean_g < 70:
        score += 0.5
    return score


def _keep_outer_property_ring(lines: list[Polyline]) -> list[Polyline]:
    """Drop interior false hits; keep strokes near the outer envelope of candidates."""
    if len(lines) <= 4:
        return lines
    pts = np.vstack([p.points[[0, -1]] for p in lines]).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) < 3:
        return lines
    # Distance from each segment midpoint to hull edges
    kept: list[Polyline] = []
    for poly in lines:
        mid = poly.points[len(poly.points) // 2]
        d = cv2.pointPolygonTest(hull.reshape(-1, 1, 2), (float(mid[0]), float(mid[1])), True)
        # Outside or near boundary (within ~40px inside)
        if d >= -40.0:
            kept.append(poly)
            continue
        # Also keep very long strokes even if slightly inset
        if poly.length_px >= 500 and d >= -120.0:
            kept.append(poly)
    return kept or lines


def _sample_polyline_profile(
    poly: Polyline, gray: np.ndarray
) -> tuple[float, float, float, str] | None:
    h, w = gray.shape
    pts = poly.points
    if len(pts) < 2:
        return None
    n = min(80, max(12, int(poly.length_px // 4)))
    idxs = np.linspace(0, len(pts) - 1, num=n).astype(int)
    ink_flags: list[bool] = []
    grays: list[float] = []
    for i in idxs:
        x = int(np.clip(pts[i, 0], 0, w - 1))
        y = int(np.clip(pts[i, 1], 0, h - 1))
        patch = gray[max(0, y - 1) : min(h, y + 2), max(0, x - 1) : min(w, x + 2)]
        g = float(patch.mean())
        grays.append(g)
        ink_flags.append(g < 185)

    ink_arr = np.array(ink_flags, dtype=bool)
    ink_ratio = float(ink_arr.mean())
    trans_rate = float(np.sum(ink_arr[1:] != ink_arr[:-1])) / max(len(ink_arr) - 1, 1)
    mean_gray = float(np.mean([g for g, f in zip(grays, ink_flags) if f])) if any(ink_flags) else float(np.mean(grays))
    run_pattern = _runs_to_signature(ink_arr)
    return ink_ratio, trans_rate, mean_gray, run_pattern
