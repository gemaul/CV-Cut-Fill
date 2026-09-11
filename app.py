"""CV-Cut-Fill — detect plan entities, compute deterministic cut/fill, score vs truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.overlay import compose_overlay
from src.pipeline import run_pipeline
from src.stages.evaluate import evaluate

EXAMPLE_DIR = ROOT / "samples" / "example"
EXAMPLE_PDF = EXAMPLE_DIR / "plan.pdf"
GROUND_TRUTH_PATH = EXAMPLE_DIR / "ground_truth.json"

ARCHITECTURE = [
    ("Curved contours", "Stroke → skeleton → polylines", "contours"),
    ("Spot elevations / symbols", "Heuristic near OCR/text", "symbols"),
    ("Elevation callouts", "PDF text + optional EasyOCR", "elevations"),
    ("Existing vs proposed", "Heuristic surface blend", "existing_vs_proposed"),
    ("Legend / notes", "Stub in MVP", "legend"),
    ("Elevation↔geometry link", "Nearest polyline", "association"),
    ("Cut/fill volumes", "Deterministic grid", "cut_fill"),
]


def load_ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text())


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="CV-Cut-Fill",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CV-Cut-Fill")
    st.caption(
        "Detect + highlight plan entities → deterministic cut/fill. "
        "Example mode scores against the Zirl Palmer volume report."
    )

    mode = st.radio(
        "Mode",
        ["Example (with ground truth)", "Try Your Own"],
        horizontal=True,
    )
    example_mode = mode.startswith("Example")

    with st.sidebar:
        st.header("Layers")
        show_contours = st.checkbox("Curved contours", value=True)
        show_elevations = st.checkbox("Elevation callouts", value=True)
        show_symbols = st.checkbox("Spot markers", value=True)
        show_assoc = st.checkbox("Associations", value=True)
        st.divider()
        dpi = st.slider("Raster DPI", 100, 200, 140, 10)
        scale = st.number_input(
            "Drawing scale (ft per inch)",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="C-201 graphic scale is typically 1\" = 20'. Adjust if needed.",
        )
        use_ocr = st.checkbox(
            "Fallback EasyOCR if few PDF elevations",
            value=False,
            help="Slower; enable if vector text extraction finds little.",
        )
        tolerance = st.slider("Score tolerance (%)", 1, 50, 5)
        run = st.button("Run detection", type="primary", use_container_width=True)

    pdf_path: Path | None = None
    ground_truth = None

    if example_mode:
        st.info(
            f"Static example: **Grading Plan C-201** · truth **cut "
            f"{load_ground_truth()['cut_cy']} / fill {load_ground_truth()['fill_cy']} CY** "
            "(Subgrade vs. Stripped)."
        )
        pdf_path = EXAMPLE_PDF
        ground_truth = load_ground_truth()
        if not pdf_path.exists():
            st.error(f"Missing example PDF at {pdf_path}")
            return
    else:
        st.warning("Try Your Own — detection + volumes only. No ground-truth scoring.")
        uploaded = st.file_uploader("Upload a plan PDF", type=["pdf"])
        if uploaded is not None:
            upload_dir = ROOT / "samples" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = upload_dir / uploaded.name
            pdf_path.write_bytes(uploaded.getbuffer())

    layers = {
        "contours": show_contours,
        "elevations": show_elevations,
        "symbols": show_symbols,
        "associations": show_assoc,
    }

    if run and pdf_path is not None:
        with st.spinner("Rasterizing, detecting contours & elevations…"):
            try:
                result = run_pipeline(
                    pdf_path,
                    dpi=dpi,
                    scale_ft_per_inch=float(scale),
                    ground_truth=ground_truth,
                    use_ocr=use_ocr,
                    overlay_layers=layers,
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return
        st.session_state["result"] = result
        st.session_state["example_mode"] = example_mode
        st.session_state["tolerance"] = tolerance

    result = st.session_state.get("result")
    if result is None:
        st.write("Choose a mode and click **Run detection**.")
        _architecture_footer({})
        return

    # Re-compose overlay if layers changed after run
    result.overlay_bgr = compose_overlay(
        result.page.image_bgr,
        polylines=result.polylines,
        elevations=result.elevations,
        associations=result.associations,
        layers=layers,
    )
    if example_mode and ground_truth is not None:
        result.evaluation = evaluate(
            result.volumes, ground_truth, tolerance_pct=float(tolerance)
        )

    left, right = st.columns([2.4, 1], gap="large")
    with left:
        st.subheader("Detections")
        st.image(bgr_to_rgb(result.overlay_bgr), use_container_width=True)
        with st.expander("Color legend"):
            st.markdown(
                "- **Cyan** — curved contour polylines\n"
                "- **Green** — elevation callouts\n"
                "- **Orange** — spot markers near elevations\n"
                "- **Magenta** — elevation → contour links"
            )

    with right:
        st.subheader("Volumes")
        v = result.volumes
        c1, c2, c3 = st.columns(3)
        c1.metric("Cut (CY)", f"{v.cut_cy:,.0f}")
        c2.metric("Fill (CY)", f"{v.fill_cy:,.0f}")
        c3.metric("Net fill−cut", f"{v.net_cy:,.0f}")
        st.caption(v.message)
        st.caption(result.surfaces.method_note)

        st.subheader("Detections")
        st.write(
            {
                "contour_polylines": len(result.polylines),
                "elevations": len(result.elevations),
                "associations": len(result.associations),
                "surface_points": result.surfaces.point_count,
            }
        )

        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        if example_mode and result.evaluation is not None:
            ev = result.evaluation
            st.subheader("Score vs ground truth")
            badge = "✅ Within tolerance" if ev.within_tolerance else "❌ Outside tolerance"
            st.markdown(f"**{badge}** (≤ {ev.tolerance_pct:g}%)")
            st.table(
                {
                    "": ["Cut", "Fill", "Net |error|"],
                    "Truth (CY)": [
                        f"{ev.truth_cut_cy:,.0f}",
                        f"{ev.truth_fill_cy:,.0f}",
                        "—",
                    ],
                    "Pred (CY)": [
                        f"{ev.pred_cut_cy:,.0f}",
                        f"{ev.pred_fill_cy:,.0f}",
                        "—",
                    ],
                    "Abs err": [
                        f"{ev.cut_abs_error:,.0f}",
                        f"{ev.fill_abs_error:,.0f}",
                        f"{ev.net_abs_error:,.0f}",
                    ],
                    "% err": [
                        f"{ev.cut_pct_error:.1f}%" if ev.cut_pct_error is not None else "—",
                        f"{ev.fill_pct_error:.1f}%" if ev.fill_pct_error is not None else "—",
                        "—",
                    ],
                }
            )
            st.caption(
                "Truth from AGTEK-style Total Regions on Subgrade vs. Stripped. "
                "Day-one CV on one sheet is not expected to match closely."
            )
        elif not example_mode:
            st.info("Evaluation only available on the example plan.")

        with st.expander("Elevation callouts"):
            rows = [
                {
                    "value_ft": e.value_ft,
                    "source": e.source,
                    "text": e.text,
                    "conf": round(e.confidence, 2),
                }
                for e in result.elevations[:100]
            ]
            st.dataframe(rows, use_container_width=True)

    _architecture_footer(result.stage_status)


def _architecture_footer(status: dict) -> None:
    st.divider()
    st.markdown("#### Pipeline map")
    cols = st.columns(len(ARCHITECTURE))
    for col, (problem, approach, key) in zip(cols, ARCHITECTURE):
        state = status.get(key, "—")
        col.markdown(f"**{problem}**")
        col.caption(approach)
        col.code(state, language=None)


if __name__ == "__main__":
    main()
