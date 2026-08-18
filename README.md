# VoteYourWay — India Election Intelligence

VoteYourWay is a React and FastAPI application for exploring Indian political-party manifestoes, promise-completion analysis, ML predictions, clustering, association rules, and personalised rankings.

The public demo intentionally serves a versioned, precomputed snapshot. That keeps it fast and dependable for visitors while retaining the full PDF/NLP/ML pipeline for local analysis and controlled persistent-backend runs.

## Deployment architecture

```text
Vercel (React / Vite) ── HTTPS ──> Render (FastAPI read-only API)
                                      └── committed PDFs, JSON outputs, and models
```

This is preferable to deploying the pipeline as a Vercel Python function: PDF extraction, NLTK data downloads, Groq calls/retries, scraping, model retraining, thread-based work, and disk writes can exceed serverless execution and persistence guarantees. The backend is deliberately configured with `PIPELINE_ENABLED=false` in the public deployment. The existing pipeline is not removed; it can be run locally or on a separately controlled persistent service.

## Runtime data that must be committed

The public API reads the following versioned artifacts. They must be included in Git before deploying from GitHub:

- `data/manifestoes/*.pdf` — six BJP/INC manifestoes (2009, 2014, 2019)
- `data/processed/predictions.json`, `party_scores.json`, `clustering_results.json`, and `apriori_results.json`
- `models/*.pkl` — classifier and prediction model artifacts, needed for an enabled pipeline
- `data/completion/*.json` and the other processed JSON outputs — reproducibility/supporting pipeline inputs

They are no longer ignored by `.gitignore`. Verify this once before committing:

```bash
git add data models
git status
```

Do not commit API keys or private source data. `backend/.env` remains ignored.

## Deploy the public demo

### 1. Deploy the API to Render

Create a Render Blueprint deployment from this repository; `render.yaml` installs `backend/requirements.txt` and starts `backend.main:app` from the repository root.

Set these Render environment variables:

```text
PIPELINE_ENABLED=false
FRONTEND_URLS=https://YOUR-VERCEL-PROJECT.vercel.app
```

After deployment, open `https://YOUR-RENDER-SERVICE/api/health`. It should return `"status": "ok"` and `"demo_data_available": true`.

### 2. Deploy the frontend to Vercel

Import the same repository in Vercel and set **Root Directory** to `frontend`.

Set this build-time environment variable (no trailing slash):

```text
VITE_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com
```

Vercel runs `npm run build`. `frontend/vercel.json` makes React Router deep links resolve to the SPA entry point. Add the final Vercel URL (and any custom domain) to Render's `FRONTEND_URLS`, comma-separated, then redeploy Render.

## Local development

Requirements: Python 3.11, Node.js 18+, and the tracked `data/` artifacts.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
npm --prefix frontend ci
uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```bash
npm --prefix frontend run dev
```

The Vite development proxy routes `/api` to `http://localhost:8000`; no frontend environment variable is needed locally. `./start.sh` and `start.bat` automate the same startup flow.

## Running the full pipeline

Copy `backend/.env.example` to `backend/.env`, then set a non-default `ADMIN_USERNAME` and `ADMIN_PASSWORD`. Set `PIPELINE_ENABLED=true` only on your local machine or a persistent worker/service you control. `GROQ_API_KEY` is optional; without it, deterministic rule-based extraction and completion analysis are used.

The pipeline writes PDFs-derived and model artifacts under `data/` and `models/`, may call Groq, and can scrape external sites only if requested. Review the generated files before committing a new demo snapshot. Do not expose pipeline credentials in a public frontend: browser Basic Auth is only a local/admin convenience, not a visitor-facing feature.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Lightweight deployment health check |
| GET | `/api/status` | Demo artifact availability and local pipeline status |
| GET | `/api/manifestoes` | Available manifesto PDFs |
| GET | `/api/promises` | Filterable precomputed promises |
| GET | `/api/scores` | Party scores |
| POST | `/api/scores/custom` | In-memory personalised scoring |
| GET | `/api/recommendation` | Recommendation from saved scores |
| GET | `/api/clustering` | Saved clustering output |
| GET | `/api/apriori` | Saved association-rule output |
| GET | `/api/analytics/overview` | Dashboard aggregate data |
| POST | `/api/pipeline/run` | Disabled in public deployment; authenticated when explicitly enabled |

## Tech stack

React, Vite, Tailwind CSS, FastAPI, pdfplumber, PyMuPDF, NLTK, scikit-learn, mlxtend, Groq (optional), requests, and BeautifulSoup.
