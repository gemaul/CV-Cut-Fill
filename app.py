"""CV-Cut-Fill — detect plan entities, compute deterministic cut/fill, score vs truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineResult, run_pipeline
from src.stages.evaluate import evaluate
from src.viewer import build_interactive_figure

EXAMPLE_DIR = ROOT / "samples" / "example"
GRADING_PDF = EXAMPLE_DIR / "grading_plan.pdf"
EXISTING_PDF = EXAMPLE_DIR / "existing_conditions.pdf"
GROUND_TRUTH_PATH = EXAMPLE_DIR / "ground_truth.json"

ARCHITECTURE = [
    ("Sheet segments", "Drawing / legend / notes / title", "segments"),
    ("Scale", "Parse 1\" = N' from sheet", "scale"),
    ("Curved contours", "Topo only; existing vs proposed", "contours"),
    ("Elevation callouts", "Decimal + context; FFE deduped", "elevations"),
    ("Legend symbols", "Parse MAP LEGEND + icons", "legend"),
    ("Elevation↔geometry", "Nearest topo polyline", "association"),
    ("Existing vs proposed", "Classified contours/spots", "existing_vs_proposed"),
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
        "Systematically segment sheets, detect scale + legend, highlight topo entities, "
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
        )
        tolerance = st.slider("Score tolerance (%)", 1, 50, 5)
        st.caption("Scale is auto-detected per sheet.")
        override_scale = st.checkbox("Override detected scale", value=False)
        scale_override = None
        if override_scale:
            scale_override = st.number_input(
                "ft per inch", min_value=1.0, max_value=200.0, value=10.0, step=1.0
            )
        run = st.button("Run detection", type="primary", use_container_width=True)

    ground_truth = None
    upload_path: Path | None = None

    if example_mode:
        ground_truth = load_ground_truth()
        st.info(
            "Example tabs: **C-201 Grading Plan** (volumes) and "
            "**V-101 Existing Conditions** (context). "
            f"Truth **cut {ground_truth['cut_cy']} / fill {ground_truth['fill_cy']} CY**."
        )
        if not GRADING_PDF.exists() or not EXISTING_PDF.exists():
            st.error("Missing samples/example grading/existing PDFs.")
            return
    else:
        st.warning("Try Your Own — detection + volumes only. No ground-truth scoring.")
        uploaded = st.file_uploader("Upload a plan PDF", type=["pdf"])
        if uploaded is not None:
            upload_dir = ROOT / "samples" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / uploaded.name
            upload_path.write_bytes(uploaded.getbuffer())

    if run:
        scale_kw = float(scale_override) if scale_override is not None else None
        if example_mode:
            with st.spinner("Segmenting + detecting on C-201 and V-101…"):
                try:
                    grading = run_pipeline(
                        GRADING_PDF,
                        dpi=dpi,
                        scale_ft_per_inch=scale_kw,
                        ground_truth=ground_truth,
                        use_ocr=use_ocr,
                        sheet_label="C-201 Grading Plan",
                        compute_volumes=True,
                    )
                    existing = run_pipeline(
                        EXISTING_PDF,
                        dpi=dpi,
                        scale_ft_per_inch=scale_kw,
                        ground_truth=None,
                        use_ocr=use_ocr,
                        sheet_label="V-101 Existing Conditions",
                        compute_volumes=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return
            st.session_state["sheets"] = {
                "C-201 Grading Plan": grading,
                "V-101 Existing Conditions": existing,
            }
            st.session_state["volume_sheet"] = "C-201 Grading Plan"
            st.session_state["example_mode"] = True
        elif upload_path is not None:
            with st.spinner("Segmenting sheet…"):
                try:
                    result = run_pipeline(
                        upload_path,
                        dpi=dpi,
                        scale_ft_per_inch=scale_kw,
                        ground_truth=None,
                        use_ocr=use_ocr,
                        sheet_label=upload_path.name,
                        compute_volumes=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return
            st.session_state["sheets"] = {upload_path.name: result}
            st.session_state["volume_sheet"] = upload_path.name
            st.session_state["example_mode"] = False

    sheets: dict[str, PipelineResult] | None = st.session_state.get("sheets")
    if not sheets:
        st.write("Choose a mode and click **Run detection**.")
        _architecture_footer({})
        return

    example_mode = bool(st.session_state.get("example_mode", example_mode))
    volume_key = st.session_state.get("volume_sheet", next(iter(sheets)))
    volume_result = sheets[volume_key]

    tab_labels = list(sheets.keys())
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            _render_sheet_tab(
                sheets[label],
                is_volume_source=(label == volume_key),
                example_mode=example_mode,
                ground_truth=ground_truth if example_mode else None,
                tolerance=float(tolerance),
                volume_result=volume_result if label == volume_key else None,
            )


def _render_sheet_tab(
    active: PipelineResult,
    *,
    is_volume_source: bool,
    example_mode: bool,
    ground_truth: dict | None,
    tolerance: float,
    volume_result: PipelineResult | None,
) -> None:
    if is_volume_source:
        st.caption("This sheet drives **cut/fill volumes** and scoring.")
    else:
        st.caption("Context sheet — detection only (volumes skipped).")

    meta_l, meta_r = st.columns([1, 1.2])
    with meta_l:
        st.subheader("Detected scale")
        st.metric("ft / inch", f"{active.scale.ft_per_inch:g}")
        st.write(
            {
                "raw_text": active.scale.raw_text,
                "source": active.scale.source,
                "confidence": active.scale.confidence,
                "surface": active.surfaces.method_note,
            }
        )
        st.caption(active.segments.notes)
        st.write(
            {
                "regions": [r.name for r in active.segments.regions],
                "elev_kinds": _count_kinds(active.elevations),
                "contour_kinds": _count_poly_kinds(active.polylines),
            }
        )
    with meta_r:
        st.subheader("Map legend symbols")
        if active.legend:
            with st.container(height=280):
                for e in active.legend:
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

    st.subheader("Interactive plan")
    map_col, layer_col = st.columns([3.2, 1], gap="medium")
    with layer_col:
        st.markdown("**Layers**")
        # Unique keys per sheet tab
        sk = active.sheet_label or "sheet"
        show_contours = st.checkbox("Contours", value=True, key=f"{sk}_contours")
        show_elevations = st.checkbox("Elevations", value=True, key=f"{sk}_elev")
        show_symbols = st.checkbox("Spot markers", value=True, key=f"{sk}_sym")
        show_assoc = st.checkbox("Associations", value=True, key=f"{sk}_assoc")
        show_segments = st.checkbox("Sheet segments", value=True, key=f"{sk}_seg")
        focus_drawing = st.checkbox("Focus drawing", value=True, key=f"{sk}_focus")
        st.divider()
        st.write(
            {
                "contours": len(active.polylines),
                "elevations": len(active.elevations),
                "associations": len(active.associations),
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
            active.page.image_bgr,
            polylines=active.polylines,
            elevations=active.elevations,
            associations=active.associations,
            segments=active.segments,
            layers=layers,
            focus_drawing=focus_drawing,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

    if is_volume_source and volume_result is not None:
        vcol, scol = st.columns(2)
        with vcol:
            st.subheader("Volumes")
            v = volume_result.volumes
            c1, c2, c3 = st.columns(3)
            c1.metric("Cut (CY)", f"{v.cut_cy:,.0f}")
            c2.metric("Fill (CY)", f"{v.fill_cy:,.0f}")
            c3.metric("Net fill−cut", f"{v.net_cy:,.0f}")
            st.caption(v.message)
            st.caption(volume_result.surfaces.method_note)
            for w in volume_result.warnings:
                st.warning(w)
        with scol:
            if example_mode and ground_truth is not None:
                ev = evaluate(
                    volume_result.volumes,
                    ground_truth,
                    tolerance_pct=tolerance,
                )
                st.subheader("Score vs ground truth")
                badge = (
                    "Within tolerance" if ev.within_tolerance else "Outside tolerance"
                )
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
                            f"{ev.cut_pct_error:.1f}%"
                            if ev.cut_pct_error is not None
                            else "—",
                            f"{ev.fill_pct_error:.1f}%"
                            if ev.fill_pct_error is not None
                            else "—",
                            "—",
                        ],
                    }
                )
            else:
                st.info("No ground-truth scoring for this run.")

    for w in active.warnings:
        if not is_volume_source or w not in (volume_result.warnings if volume_result else []):
            st.caption(f"Note: {w}")

    _architecture_footer(active.stage_status)


def _count_kinds(elevations) -> dict:
    out: dict[str, int] = {}
    for e in elevations:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out


def _count_poly_kinds(polylines) -> dict:
    out: dict[str, int] = {}
    for p in polylines:
        out[p.kind] = out.get(p.kind, 0) + 1
    return out


def _architecture_footer(status: dict) -> None:
    st.divider()
    st.markdown("#### Pipeline map")
    cols = st.columns(len(ARCHITECTURE))
    for col, (problem, approach, key) in zip(cols, ARCHITECTURE):
        state = status.get(key, "—")
        col.markdown(f"**{problem}**")
        col.caption(approach)
        col.code(str(state), language=None)


if __name__ == "__main__":
    main()
