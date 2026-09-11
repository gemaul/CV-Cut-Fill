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

## Production (own domain)

Streamlit Community Cloud cannot attach a custom domain. This repo ships with
**Docker + Fly.io** so you can serve the app at e.g. `https://cutfill.yourdomain.com`.

### 1. Push to GitHub

```bash
git push -u origin main
```

### 2. Deploy to Fly.io

```bash
# Install: https://fly.io/docs/hands-on/install-flyctl/
fly auth login
fly launch --no-deploy   # first time only if app name is free
fly deploy
```

App URL (temporary): `https://cv-cut-fill.fly.dev`

### 3. Attach your domain

```bash
fly certs add cutfill.yourdomain.com
```

Then at your DNS provider create the record Fly prints (usually a **CNAME** to
`cv-cut-fill.fly.dev`, or A/AAAA for apex domains). Wait for TLS:

```bash
fly certs show cutfill.yourdomain.com
```

### Docker only (any VPS)

```bash
docker build -t cv-cut-fill .
docker run -p 8501:8501 cv-cut-fill
```

Put nginx/Caddy in front with TLS for your domain.

## License / data

Demo drawings and volume figures are for authorized product exploration only.
