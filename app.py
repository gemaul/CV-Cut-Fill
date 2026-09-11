"""CV-Cut-Fill — detect plan entities, compute deterministic cut/fill, score vs truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline
from src.stages.evaluate import evaluate
from src.viewer import build_interactive_figure

EXAMPLE_DIR = ROOT / "samples" / "example"
EXAMPLE_PDF = EXAMPLE_DIR / "plan.pdf"
GROUND_TRUTH_PATH = EXAMPLE_DIR / "ground_truth.json"

ARCHITECTURE = [
    ("Sheet segments", "Drawing / legend / title rails", "segments"),
    ("Scale", "Parse 1\" = N' from sheet", "scale"),
    ("Curved contours", "Skeleton polylines in drawing", "contours"),
    ("Elevation callouts", "PDF text (+ optional OCR)", "elevations"),
    ("Legend symbols", "Parse MAP LEGEND names", "legend"),
    ("Elevation↔geometry", "Nearest polyline", "association"),
    ("Cut/fill volumes", "Deterministic grid", "cut_fill"),
]


def load_ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text())


def main() -> None:
    st.set_page_config(
        page_title="CV-Cut-Fill",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CV-Cut-Fill")
    st.caption(
        "Segment the sheet, detect scale + legend, highlight entities, "
        "then compute deterministic cut/fill."
    )

    mode = st.radio(
        "Mode",
        ["Example (with ground truth)", "Try Your Own"],
        horizontal=True,
    )
    example_mode = mode.startswith("Example")

    with st.sidebar:
        st.header("Run settings")
        dpi = st.slider("Raster DPI", 100, 200, 140, 10)
        use_ocr = st.checkbox(
            "Fallback EasyOCR if few PDF elevations",
            value=False,
            help="Slower; enable if vector text extraction finds little.",
        )
        tolerance = st.slider("Score tolerance (%)", 1, 50, 5)
        st.caption("Scale is auto-detected from the sheet (override only if needed).")
        override_scale = st.checkbox("Override detected scale", value=False)
        scale_override = None
        if override_scale:
            scale_override = st.number_input(
                "ft per inch",
                min_value=1.0,
                max_value=200.0,
                value=10.0,
                step=1.0,
            )
        run = st.button("Run detection", type="primary", use_container_width=True)

    pdf_path: Path | None = None
    ground_truth = None

    if example_mode:
        gt = load_ground_truth()
        st.info(
            f"Static example: **Grading Plan C-201** · truth **cut "
            f"{gt['cut_cy']} / fill {gt['fill_cy']} CY** (Subgrade vs. Stripped)."
        )
        pdf_path = EXAMPLE_PDF
        ground_truth = gt
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

    if run and pdf_path is not None:
        with st.spinner("Segmenting sheet, detecting scale/legend/contours…"):
            try:
                result = run_pipeline(
                    pdf_path,
                    dpi=dpi,
                    scale_ft_per_inch=float(scale_override)
                    if scale_override is not None
                    else None,
                    ground_truth=ground_truth,
                    use_ocr=use_ocr,
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return
        st.session_state["result"] = result
        st.session_state["example_mode"] = example_mode

    result = st.session_state.get("result")
    if result is None:
        st.write("Choose a mode and click **Run detection**.")
        _architecture_footer({})
        return

    # ---- Processed outputs: scale + legend ----
    meta_l, meta_r = st.columns([1, 1.2])
    with meta_l:
        st.subheader("Detected scale")
        st.metric("ft / inch", f"{result.scale.ft_per_inch:g}")
        st.write(
            {
                "raw_text": result.scale.raw_text,
                "source": result.scale.source,
                "confidence": result.scale.confidence,
                "used_for_volumes": result.surfaces.method_note,
            }
        )
        st.caption(result.segments.notes)
    with meta_r:
        st.subheader("Map legend symbols")
        if result.legend:
            # Show symbol crop + name for each legend entry
            for e in result.legend:
                c_icon, c_text = st.columns([0.22, 0.78], gap="small")
                with c_icon:
                    if e.icon_rgb is not None:
                        st.image(e.icon_rgb, width=72)
                    else:
                        st.caption("—")
                with c_text:
                    st.markdown(f"**{e.name}**")
                    st.caption(e.symbol_hint)
        else:
            st.write("No legend entries parsed.")

    # ---- Interactive map + live layer toggles ----
    st.subheader("Interactive plan")
    map_col, layer_col = st.columns([3.2, 1], gap="medium")
    with layer_col:
        st.markdown("**Layers**")
        st.caption("Toggle while viewing — no re-run needed.")
        show_contours = st.checkbox("Curved contours", value=True, key="ly_contours")
        show_elevations = st.checkbox("Elevations", value=True, key="ly_elev")
        show_symbols = st.checkbox("Spot markers", value=True, key="ly_sym")
        show_assoc = st.checkbox("Associations", value=True, key="ly_assoc")
        show_segments = st.checkbox("Sheet segments", value=True, key="ly_seg")
        focus_drawing = st.checkbox("Focus drawing viewport", value=True, key="ly_focus")
        st.divider()
        st.write(
            {
                "contours": len(result.polylines),
                "elevations": len(result.elevations),
                "associations": len(result.associations),
            }
        )

    layers = {
        "contours": show_contours,
        "elevations": show_elevations,
        "symbols": show_symbols,
        "associations": show_assoc,
        "segments": show_segments,
    }

    with map_col:
        fig = build_interactive_figure(
            result.page.image_bgr,
            polylines=result.polylines,
            elevations=result.elevations,
            associations=result.associations,
            segments=result.segments,
            layers=layers,
            focus_drawing=focus_drawing,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "lasso2d",
                    "select2d",
                    "drawopenpath",
                    "eraseshape",
                ],
            },
        )

    # ---- Volumes / score ----
    vcol, scol = st.columns(2)
    with vcol:
        st.subheader("Volumes")
        v = result.volumes
        c1, c2, c3 = st.columns(3)
        c1.metric("Cut (CY)", f"{v.cut_cy:,.0f}")
        c2.metric("Fill (CY)", f"{v.fill_cy:,.0f}")
        c3.metric("Net fill−cut", f"{v.net_cy:,.0f}")
        st.caption(v.message)
        if result.warnings:
            for w in result.warnings:
                st.warning(w)

    with scol:
        if example_mode and ground_truth is not None:
            ev = evaluate(
                result.volumes, ground_truth, tolerance_pct=float(tolerance)
            )
            st.subheader("Score vs ground truth")
            badge = "Within tolerance" if ev.within_tolerance else "Outside tolerance"
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
        elif not example_mode:
            st.info("Evaluation only available on the example plan.")

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
