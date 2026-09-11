# CV-Cut-Fill

Detect and highlight entities on a civil **grading plan**, then compute **deterministic** cut/fill volumes. Example mode scores against a known earthwork report.

## Demo corpus

| Asset | Role |
| --- | --- |
| [`samples/example/plan.pdf`](samples/example/plan.pdf) | Zirl Palmer Prep Academy — **Grading Plan C-201** |
| [`samples/example/ground_truth.json`](samples/example/ground_truth.json) | **Cut 952 CY / Fill 1319 CY** (Total Regions, Subgrade vs. Stripped) |
| [`samples/example/volume_report.xlsx`](samples/example/volume_report.xlsx) | Provenance volume report |

Optional sheets for Try Your Own are under `samples/optional/`.

## Modes

1. **Example (with ground truth)** — runs C-201, shows detection overlays, volumes, and error vs 952 / 1319 CY.
2. **Try Your Own** — upload any PDF; same detection + volumes; **no scoring**.

## Architecture (MVP)

| Problem | Approach in this MVP |
| --- | --- |
| Curved contours | Adaptive threshold → skeleton → traced polylines |
| Spot elevations | Heuristic markers near elevation text |
| Elevation values | PDF vector text first; optional EasyOCR fallback |
| Existing vs proposed | Heuristic surface blend (classifier later) |
| Legend / notes | Stub |
| Elevation↔geometry | Nearest contour polyline |
| Cut/fill | Deterministic grid (proposed − existing) |

**Not in MVP:** M-LSD / neural straight-line detectors (deferred).

Cut/fill math is never an LLM guess.

## Setup

```bash
cd CV-Cut-Fill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

### Tips

- Start with **Example** and **Run detection** (EasyOCR off is fine if PDF text is present).
- Adjust **ft per inch** if the sheet scale isn’t 1\" = 20'.
- Expect a large score gap initially — the Excel truth comes from a full region model; this MVP reads one sheet.

## Publish (Streamlit Community Cloud)

1. Push this repo to GitHub (`gemaul/CV-Cut-Fill`).
2. At [share.streamlit.io](https://share.streamlit.io), deploy `app.py` from the repo root.
3. Use Python 3.11+; install from `requirements.txt`.

## License / data

Demo drawings and volume figures are for local product exploration. Do not redistribute project PDFs beyond your authorized use.
