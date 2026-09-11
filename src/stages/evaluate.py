from __future__ import annotations

from dataclasses import dataclass

from .cut_fill import CutFillResult


@dataclass
class EvaluationResult:
    truth_cut_cy: float
    truth_fill_cy: float
    pred_cut_cy: float
    pred_fill_cy: float
    cut_abs_error: float
    fill_abs_error: float
    cut_pct_error: float | None
    fill_pct_error: float | None
    net_abs_error: float
    within_tolerance: bool
    tolerance_pct: float


def evaluate(
    volumes: CutFillResult,
    ground_truth: dict,
    *,
    tolerance_pct: float = 5.0,
) -> EvaluationResult:
    truth_cut = float(ground_truth["cut_cy"])
    truth_fill = float(ground_truth["fill_cy"])
    pred_cut = float(volumes.cut_cy)
    pred_fill = float(volumes.fill_cy)

    cut_abs = abs(pred_cut - truth_cut)
    fill_abs = abs(pred_fill - truth_fill)
    cut_pct = (cut_abs / truth_cut * 100.0) if truth_cut else None
    fill_pct = (fill_abs / truth_fill * 100.0) if truth_fill else None
    net_truth = truth_fill - truth_cut
    net_pred = pred_fill - pred_cut

    pcts = [p for p in (cut_pct, fill_pct) if p is not None]
    within = bool(pcts) and all(p <= tolerance_pct for p in pcts)

    return EvaluationResult(
        truth_cut_cy=truth_cut,
        truth_fill_cy=truth_fill,
        pred_cut_cy=pred_cut,
        pred_fill_cy=pred_fill,
        cut_abs_error=cut_abs,
        fill_abs_error=fill_abs,
        cut_pct_error=cut_pct,
        fill_pct_error=fill_pct,
        net_abs_error=abs(net_pred - net_truth),
        within_tolerance=within,
        tolerance_pct=tolerance_pct,
    )
