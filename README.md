# VoteYourWay — India Election Intelligence

A full-stack ML application to analyse Indian political party manifestoes, track promise completion, and make data-driven voting recommendations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python + FastAPI |
| Frontend | React + Vite + Tailwind CSS + Recharts |
| PDF Parsing | pdfplumber + PyMuPDF |
| NLP | NLTK (tokenize, lemmatize, stopwords) |
| ML | scikit-learn (Decision Tree, Logistic Regression, K-Means) |
| Association Mining | mlxtend (Apriori) |
| LLM (limited) | Groq API (promise extraction + completion matching only) |
| Web Scraping | requests + BeautifulSoup |

## Quick Start

### Linux / macOS
```bash
chmod +x start.sh
./start.sh
```

### Windows
```cmd
start.bat
```

Then open **http://localhost:3000** and click **Run Pipeline**.

---

## Manual Setup

### 1. Backend

```bash
cd backend
python3 -m venv ../venv
source ../venv/bin/activate          # Windows: ..\venv\Scripts\activate
pip install -r requirements.txt
```

#### Configure Groq API (optional but recommended)
```bash
# Edit backend/.env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at https://console.groq.com

### 2. Generate Sample PDFs
```bash
cd ..  # project root
python generate_sample_manifestoes.py
```
This creates 6 sample manifesto PDFs (BJP & INC × 2009/2014/2019) in `/data/manifestoes/`.

**To use your own PDFs:** Copy them to `/data/manifestoes/` with format `partycode_year.pdf`:
```
bjp_2024.pdf
inc_2024.pdf
aap_2024.pdf
```

### 3. Start Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

---

## How It Works

### ML Pipeline (triggered by "Run Pipeline" button)

1. **PDF Ingestion** — pdfplumber/PyMuPDF extracts clean text
2. **Preprocessing** — NLTK tokenisation, stopword removal, lemmatisation
3. **Promise Extraction** — Rule-based keywords + Groq LLM refinement
4. **Classification** — Decision Tree (ID3) classifies into 10 categories
5. **Completion Analysis** — Web scraping + Groq semantic matching
6. **K-Means Clustering** — Clusters parties by TF-IDF + category features
7. **Apriori Mining** — Discovers category co-occurrence rules
8. **Logistic Regression** — Predicts completion probability (0–1)
9. **Scoring** — Weighted formula produces final party score

### Scoring Formula
```
Score = (completion_rate × 0.40) +
        (predicted_completion_strength × 0.20) +
        (category_coverage × 0.15) +
        (promise_density × 0.10) +
        (consistency_score × 0.15)
```

### Prediction Interpretation
- **> 70%** → 🟢 Likely
- **40–70%** → 🟡 Uncertain
- **< 40%** → 🔴 Unlikely

---

## Project Structure

```
manifesto_app/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── ingestion.py            # PDF loading
│   ├── preprocessing.py        # NLP pipeline
│   ├── promise_extraction.py   # Hybrid extraction
│   ├── feature_engineering.py  # TF-IDF + features
│   ├── classification.py       # Decision Tree
│   ├── clustering.py           # K-Means
│   ├── apriori.py              # Association rules
│   ├── scraper.py              # Web scraper
│   ├── completion_analysis.py  # Status analysis
│   ├── prediction.py           # Logistic Regression
│   ├── scoring.py              # Party scoring
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── pages/
│       │   ├── Dashboard.jsx
│       │   ├── ManifestoViewer.jsx
│       │   ├── PromiseExplorer.jsx
│       │   ├── CompletionPanel.jsx
│       │   ├── ComparisonDashboard.jsx
│       │   └── Recommendation.jsx
│       └── utils/api.js
├── data/
│   ├── manifestoes/    ← Put PDF files here
│   ├── completion/     ← Scraped data
│   └── processed/      ← ML outputs
├── models/             ← Saved ML models
├── generate_sample_manifestoes.py
├── start.sh
└── start.bat
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/status | Pipeline status |
| POST | /api/pipeline/run | Start ML pipeline |
| GET | /api/manifestoes | List manifestoes |
| GET | /api/promises | Get promises (filterable) |
| GET | /api/scores | Party scores |
| POST | /api/scores/custom | Custom weighted scores |
| GET | /api/recommendation | Party recommendation |
| GET | /api/clustering | K-Means results |
| GET | /api/apriori | Association rules |
| GET | /api/analytics/overview | Dashboard data |

---

## Notes

- **LLM Usage**: Groq is used ONLY for promise extraction and completion semantic matching. All ML predictions use scikit-learn models.
- **Data persists**: Results cached in `/data/processed/`. Use "Re-run" to refresh.
- **Web scraping**: Enabled via `use_scraper: true` in pipeline request. Rate-limited by default.
- **Dark mode**: Toggle via sidebar button.
