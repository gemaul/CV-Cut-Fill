"""CV-Cut-Fill — detect plan entities, compute deterministic cut/fill, score vs truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineResult, fuse_cut_fill, run_pipeline
from src.stages.evaluate import evaluate
from src.viewer import build_interactive_figure

EXAMPLE_DIR = ROOT / "samples" / "example"
GRADING_PDF = EXAMPLE_DIR / "grading_plan.pdf"
EXISTING_PDF = EXAMPLE_DIR / "existing_conditions.pdf"
GROUND_TRUTH_PATH = EXAMPLE_DIR / "ground_truth.json"

ARCHITECTURE = [
    ("Segmentation", "Separate the drawing from legend, notes, and title block"),
    ("Property", "Trace property-line strokes from the plan"),
    ("Building", "Trace building walls when a footprint is present"),
    ("Scale", "Read the printed scale (e.g. 1\" = 10')"),
    ("Legend key", "Parse MAP LEGEND symbols and line styles"),
    ("Contours", "Follow topographic contour ink from elevation seeds"),
    ("Elevations", "Read spot elevations and contour labels"),
    ("Association", "Link each elevation callout to nearby contour geometry"),
    ("Surfaces", "Build proposed grade from C-201 and existing grade from V-101"),
    ("Cut / fill", "Compute volumes on a grid: proposed − existing"),
]


def load_ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text())


def _run_fused_pair(
    *,
    existing_pdf: Path,
    proposed_pdf: Path,
    dpi: int,
    existing_label: str,
    proposed_label: str,
    ground_truth: dict | None,
) -> tuple[PipelineResult, PipelineResult]:
    """Same workflow for example and try-your-own: existing + proposed → fuse."""
    proposed = run_pipeline(
        proposed_pdf,
        dpi=dpi,
        ground_truth=None,
        use_ocr=False,
        sheet_label=proposed_label,
        sheet_role="proposed",
        compute_volumes=False,
    )
    existing = run_pipeline(
        existing_pdf,
        dpi=dpi,
        ground_truth=None,
        use_ocr=False,
        sheet_label=existing_label,
        sheet_role="existing",
        compute_volumes=False,
    )
    proposed = fuse_cut_fill(proposed, existing, ground_truth=ground_truth)
    return existing, proposed


def main() -> None:
    st.set_page_config(
        page_title="CV-Cut-Fill",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CV-Cut-Fill")
    st.caption(
        "C-201 builds the proposed surface; V-101 builds the existing surface; "
        "cut/fill is proposed − existing on a shared site grid."
    )

    mode = st.radio(
        "Mode",
        ["Example (with ground truth)", "Try Your Own"],
        horizontal=True,
    )
    example_mode = mode.startswith("Example")

    with st.sidebar:
        st.header("Run settings")
        dpi = st.slider("Raster DPI", 100, 200, 100, 10)
        run = st.button("Run detection", type="primary", use_container_width=True)

    ground_truth = None
    existing_upload: Path | None = None
    proposed_upload: Path | None = None

    if example_mode:
        ground_truth = load_ground_truth()
        st.info(
            "Example: **C-201** → proposed grade (contours + finished-grade spots). "
            "**V-101** → existing grade (surveyed contours/labels). "
            f"Fused volumes vs truth **cut {ground_truth['cut_cy']} / fill {ground_truth['fill_cy']} CY**."
        )
        if not GRADING_PDF.exists() or not EXISTING_PDF.exists():
            st.error("Missing samples/example grading/existing PDFs.")
            return
    else:
        st.info(
            "Same workflow as the example: upload an **existing conditions** sheet and a "
            "**proposed / grading** sheet. Existing builds the existing surface; proposed "
            "builds the proposed surface; cut/fill is proposed − existing."
        )
        up_l, up_r = st.columns(2)
        with up_l:
            existing_file = st.file_uploader(
                "Existing conditions PDF",
                type=["pdf"],
                key="upload_existing",
            )
        with up_r:
            proposed_file = st.file_uploader(
                "Proposed / grading PDF",
                type=["pdf"],
                key="upload_proposed",
            )
        upload_dir = ROOT / "samples" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        if existing_file is not None:
            existing_upload = upload_dir / f"existing_{existing_file.name}"
            existing_upload.write_bytes(existing_file.getbuffer())
        if proposed_file is not None:
            proposed_upload = upload_dir / f"proposed_{proposed_file.name}"
            proposed_upload.write_bytes(proposed_file.getbuffer())

    if run:
        if example_mode:
            with st.spinner("Existing + proposed → fused cut/fill…"):
                try:
                    existing, grading = _run_fused_pair(
                        existing_pdf=EXISTING_PDF,
                        proposed_pdf=GRADING_PDF,
                        dpi=dpi,
                        existing_label="V-101 Existing Conditions",
                        proposed_label="C-201 Grading Plan",
                        ground_truth=ground_truth,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return
            st.session_state["sheets"] = {
                "V-101 Existing Conditions": existing,
                "C-201 Grading Plan": grading,
            }
            st.session_state["volume_sheet"] = "C-201 Grading Plan"
            st.session_state["example_mode"] = True
        elif existing_upload is not None and proposed_upload is not None:
            with st.spinner("Existing + proposed → fused cut/fill…"):
                try:
                    existing, grading = _run_fused_pair(
                        existing_pdf=existing_upload,
                        proposed_pdf=proposed_upload,
                        dpi=dpi,
                        existing_label=existing_upload.name.replace("existing_", "", 1),
                        proposed_label=proposed_upload.name.replace("proposed_", "", 1),
                        ground_truth=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return
            ex_tab = existing_upload.name.replace("existing_", "", 1)
            pr_tab = proposed_upload.name.replace("proposed_", "", 1)
            if ex_tab == pr_tab:
                ex_tab = f"Existing — {ex_tab}"
                pr_tab = f"Proposed — {pr_tab}"
            st.session_state["sheets"] = {
                ex_tab: existing,
                pr_tab: grading,
            }
            st.session_state["volume_sheet"] = pr_tab
            st.session_state["example_mode"] = False
        elif not example_mode:
            st.error("Upload both an existing-conditions PDF and a proposed/grading PDF.")
            return

    sheets: dict[str, PipelineResult] | None = st.session_state.get("sheets")
    if not sheets:
        st.write("Choose a mode and click **Run detection**.")
        _pipeline_guide()
        return

    example_mode = bool(st.session_state.get("example_mode", example_mode))
    volume_key = st.session_state.get("volume_sheet", next(iter(sheets)))
    volume_result = sheets[volume_key]

    # Volumes + score are site-level (fused), not tab-specific — show above sheets.
    _render_volumes_and_score(
        volume_result,
        example_mode=example_mode,
        ground_truth=ground_truth if example_mode else None,
    )
    st.divider()
    st.markdown("#### Sheet plans")
    st.caption(
        "Fused cut/fill uses the **existing** sheet for existing grade and the "
        "**proposed / grading** sheet for proposed grade (proposed − existing). "
        "Switch tabs to inspect each sheet."
        if len(sheets) > 1
        else "Inspect detection layers on the sheet below."
    )

    tab_labels = list(sheets.keys())
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            _render_sheet_tab(sheets[label])

    _pipeline_guide()


def _render_volumes_and_score(
    volume_result: PipelineResult,
    *,
    example_mode: bool,
    ground_truth: dict | None,
) -> None:
    vcol, scol = st.columns(2)
    with vcol:
        st.subheader("Volumes")
        v = volume_result.volumes
        c1, c2, c3 = st.columns(3)
        c1.metric("Cut (CY)", f"{v.cut_cy:,.0f}")
        c2.metric("Fill (CY)", f"{v.fill_cy:,.0f}")
        c3.metric("Net fill−cut", f"{v.net_cy:,.0f}")
        st.caption("Cubic yards from proposed − existing on a shared site grid.")
    with scol:
        if example_mode and ground_truth is not None:
            ev = evaluate(
                volume_result.volumes,
                ground_truth,
                tolerance_pct=5.0,
            )
            st.subheader("Score vs ground truth")
            st.table(
                {
                    "": ["Cut", "Fill"],
                    "Truth (CY)": [
                        f"{ev.truth_cut_cy:,.0f}",
                        f"{ev.truth_fill_cy:,.0f}",
                    ],
                    "Pred (CY)": [
                        f"{ev.pred_cut_cy:,.0f}",
                        f"{ev.pred_fill_cy:,.0f}",
                    ],
                    "Abs err": [
                        f"{ev.cut_abs_error:,.0f}",
                        f"{ev.fill_abs_error:,.0f}",
                    ],
                    "% err": [
                        f"{ev.cut_pct_error:.1f}%"
                        if ev.cut_pct_error is not None
                        else "—",
                        f"{ev.fill_pct_error:.1f}%"
                        if ev.fill_pct_error is not None
                        else "—",
                    ],
                }
            )
        else:
            st.subheader("Score vs ground truth")
            st.info("No ground-truth scoring for this run.")


def _render_sheet_tab(active: PipelineResult) -> None:
    if active.sheet_role == "existing":
        st.caption(
            "Existing-conditions sheet — surveyed topo feeds the **existing** surface "
            "for fused cut/fill."
        )
    elif active.sheet_role == "proposed":
        st.caption(
            "Grading plan — proposed contours and finished-grade spots feed the "
            "**proposed** surface for fused cut/fill."
        )

    st.subheader("Interactive plan")
    st.caption("Scroll to zoom · drag to pan · use layer checkboxes")
    map_col, layer_col = st.columns([3.2, 1], gap="medium")
    with layer_col:
        st.markdown("**Layers**")
        # Unique keys per sheet tab
        sk = active.sheet_label or "sheet"
        show_contours = st.checkbox("Contours", value=True, key=f"{sk}_contours")
        show_elevations = st.checkbox("Elevations", value=True, key=f"{sk}_elev")
        show_symbols = st.checkbox("Spot markers", value=True, key=f"{sk}_sym")
        show_property = st.checkbox(
            "Property boundary",
            value=bool(active.site.property_lines),
            key=f"{sk}_prop_v2",
        )
        if active.sheet_role == "existing":
            show_building = False
            st.caption("Building: not applicable on V-101 / existing sheet")
        else:
            show_building = st.checkbox(
                "Building",
                value=bool(active.site.building_lines),
                key=f"{sk}_bld_v2",
            )
        show_assoc = st.checkbox("Associations", value=False, key=f"{sk}_assoc")
        show_segments = st.checkbox("Sheet segments / zones", value=True, key=f"{sk}_seg_v3")

    layers = {
        "contours": show_contours,
        "elevations": show_elevations,
        "symbols": show_symbols,
        "associations": show_assoc,
        "segments": show_segments,
        "property": show_property,
        "building": show_building,
    }
    with map_col:
        fig = build_interactive_figure(
            active.page.image_bgr,
            polylines=active.polylines,
            elevations=active.elevations,
            associations=active.associations,
            segments=active.segments,
            site=getattr(active, "site", None),
            layers=layers,
            focus_drawing=True,
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

    meta_l, meta_r = st.columns([1, 1.2])
    with meta_l:
        st.subheader("Detected scale")
        st.metric("ft / inch", f"{active.scale.ft_per_inch:g}")
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


def _pipeline_guide() -> None:
    st.divider()
    st.markdown("#### Detection flow")
    st.caption(
        "Perception finds plan entities; cut/fill math stays deterministic "
        "(proposed − existing on a shared grid)."
    )

    items = []
    for i, (title, detail) in enumerate(ARCHITECTURE, start=1):
        connector = (
            "<div style='width:2px;height:14px;background:#CBD5E1;"
            "margin:0 0 0 15px'></div>"
            if i < len(ARCHITECTURE)
            else ""
        )
        items.append(
            f"<div style='display:flex;gap:0.85rem;align-items:flex-start'>"
            f"<div style='flex:0 0 32px;width:32px;height:32px;border-radius:999px;"
            f"background:#0F172A;color:#F8FAFC;display:flex;align-items:center;"
            f"justify-content:center;font-size:0.8rem;font-weight:600'>{i}</div>"
            f"<div style='padding-top:0.2rem;padding-bottom:0.15rem'>"
            f"<div style='font-weight:600;font-size:0.95rem;color:#0F172A'>{title}</div>"
            f"<div style='font-size:0.85rem;line-height:1.4;color:#475569'>{detail}</div>"
            f"</div></div>{connector}"
        )
    st.markdown(
        "<div style='max-width:42rem;margin:0.35rem 0 0.5rem'>"
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
