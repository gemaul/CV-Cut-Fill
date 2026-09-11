# CV-Cut-Fill

Detect plan entities on civil grading / existing-conditions sheets, then compute
**deterministic** cut/fill volumes (proposed − existing). Example mode scores
against a known earthwork report.

## Demo corpus

| Asset | Role |
| --- | --- |
| [`samples/example/grading_plan.pdf`](samples/example/grading_plan.pdf) | **C-201** — proposed grade |
| [`samples/example/existing_conditions.pdf`](samples/example/existing_conditions.pdf) | **V-101** — existing grade |
| [`samples/example/ground_truth.json`](samples/example/ground_truth.json) | Cut **952** / Fill **1319** CY |

## Modes

1. **Example (with ground truth)** — V-101 + C-201 fused cut/fill, scored vs truth.
2. **Try Your Own** — same workflow: upload existing + proposed PDFs, fuse volumes (no scoring).

## Local setup

```bash
cd CV-Cut-Fill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

Easiest host — free `*.streamlit.app` URL, no custom domain needed.

1. Push `main` to GitHub (already set up for `gemaul/CV-Cut-Fill`).
2. Open [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Repo: `gemaul/CV-Cut-Fill`, branch: `main`, main file: `app.py`.
4. Deploy. Live URL looks like `https://cv-cut-fill.streamlit.app`.

`packages.txt` installs OpenCV system libs on the Cloud VM.

### Optional: own domain (Fly.io / Docker)

See `Dockerfile` + `fly.toml` if you later want a custom hostname.

## License / data

Demo drawings and volume figures are for authorized product exploration only.
