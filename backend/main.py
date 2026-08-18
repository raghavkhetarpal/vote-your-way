"""
main.py - FastAPI backend server for Manifesto Analyzer
"""
import os
import json
import base64
import hmac
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─── Auth Setup ───
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
# A public deployment must never expose the long-running write-heavy pipeline
# merely because an environment variable was omitted.
PIPELINE_ENABLED = os.getenv("PIPELINE_ENABLED", "false").lower() == "true"

def verify_auth(authorization: Optional[str] = Header(None)) -> bool:
    """Verify basic auth credentials"""
    if not authorization or not ADMIN_USERNAME or not ADMIN_PASSWORD:
        return False
    try:
        scheme, credentials = authorization.split()
        if scheme.lower() != "basic":
            return False
        decoded = base64.b64decode(credentials).decode("utf-8")
        username, password = decoded.split(":", 1)
        return hmac.compare_digest(username, ADMIN_USERNAME) and hmac.compare_digest(password, ADMIN_PASSWORD)
    except (ValueError, UnicodeDecodeError):
        return False

app = FastAPI(
    title="Manifesto Analyzer API",
    description="Political party manifesto analysis using ML & data mining",
    version="1.0.0"
)

# ─── CORS Setup ───
origins = {
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
}
# FRONTEND_URL is retained for backwards compatibility; FRONTEND_URLS supports
# preview and custom domains without widening CORS to every origin.
for configured_origins in (os.getenv("FRONTEND_URL"), os.getenv("FRONTEND_URLS")):
    if configured_origins:
        origins.update(origin.strip().rstrip("/") for origin in configured_origins.split(",") if origin.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache
_cache: Dict[str, Any] = {
    "manifestoes": None,
    "promises": None,
    "scores": None,
    "clustering": None,
    "apriori": None,
    "pipeline_status": "idle",
    "pipeline_progress": 0,
    "pipeline_message": ""
}

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --- Request/Response Models ---

class CategoryWeights(BaseModel):
    Economy: float = 1.0
    Education: float = 1.0
    Healthcare: float = 1.0
    Infrastructure: float = 1.0
    Agriculture: float = 1.0
    Women: float = 1.0
    Youth: float = 1.0
    Environment: float = 1.0
    Defence: float = 1.0
    Others: float = 1.0


class ScoreRequest(BaseModel):
    category_weights: Optional[CategoryWeights] = None
    priority_category: Optional[str] = None


class PipelineRequest(BaseModel):
    force_rerun: bool = False
    use_scraper: bool = False


# --- Helper Functions ---

def update_status(status: str, progress: int, message: str):
    _cache["pipeline_status"] = status
    _cache["pipeline_progress"] = progress
    _cache["pipeline_message"] = message
    print(f"[{progress}%] {message}")


def run_pipeline(force_rerun: bool = False, use_scraper: bool = False):
    print("🔥 PIPELINE PROCESS STARTED")
    from .ingestion import load_all_manifestoes
    from .preprocessing import preprocess_all_manifestoes
    from .promise_extraction import extract_all_promises
    from .classification import classify_promises
    from .clustering import cluster_parties
    from .apriori import run_apriori
    from .completion_analysis import analyze_all_completions
    from .prediction import predict_completion_probabilities
    from .scoring import score_all_parties

    try:
        update_status("running", 1, "Starting pipeline...")
        
        update_status("running", 3, "Loading manifesto PDFs...")
        manifestoes = load_all_manifestoes()

        if not manifestoes:
            update_status("error", 0, "No manifesto PDFs found.")
            return

        update_status("running", 8, f"Found {len(manifestoes)} PDFs, extracting text...")
        _cache["manifestoes"] = [
            {k: v for k, v in m.items() if k != 'raw_text'}
            for m in manifestoes
        ]

        update_status("running", 12, "Preprocessing text...")
        processed = preprocess_all_manifestoes(manifestoes)

        update_status("running", 20, "Extracting promises from documents...")
        party_promises = extract_all_promises(processed)
        promises = [p for plist in party_promises.values() for p in plist]

        if not promises:
            update_status("error", 0, "No promises extracted.")
            return

        update_status("running", 35, f"Extracted {len(promises)} promises, classifying...")
        classified = classify_promises(promises, retrain=force_rerun)

        update_status("running", 50, "Analyzing promise completion status...")
        analyzed = analyze_all_completions(classified, use_scraper=use_scraper)

        update_status("running", 65, "Clustering parties by policy...")
        _cache["clustering"] = cluster_parties(analyzed)

        update_status("running", 75, "Mining association rules...")
        _cache["apriori"] = run_apriori(analyzed)

        update_status("running", 85, "Predicting future completion rates...")
        predicted = predict_completion_probabilities(analyzed, retrain=force_rerun)
        # 💾 Save predictions to disk
        with open(PROCESSED_DIR / "predictions.json", "w") as f:
            json.dump(predicted, f, indent=2)

        print("💾 Saved predictions.json")

        update_status("running", 95, "Scoring parties...")
        scores = score_all_parties(predicted)
        _cache["scores"]   = scores
        _cache["promises"] = predicted
        print(f"✅ Pipeline done — {len(predicted)} promises, {len(scores)} parties scored")

        update_status("completed", 100, f"Done — {len(promises)} promises analysed across {len(scores)} parties")

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_status("error", 0, f"Pipeline error: {str(e)}")


# --- API Endpoints ---

@app.get("/")
def root():
    return {"message": "Manifesto Analyzer API", "status": "running"}


@app.get("/api/status")
def get_status():
    predictions_path = PROCESSED_DIR / "predictions.json"

    has_data = predictions_path.exists()

    return {
        "pipeline_status": _cache.get("pipeline_status", "idle"),
        "progress": _cache.get("pipeline_progress", 0),
        "pipeline_message": _cache.get("pipeline_message", ""),
        "has_data": has_data,
        "promise_count": len(_cache.get("promises") or []),
    }


@app.get("/api/health")
def health_check():
    """Lightweight deployment health check that never loads models or PDFs."""
    return {"status": "ok", "demo_data_available": (PROCESSED_DIR / "predictions.json").exists()}


@app.post("/api/pipeline/run")
def trigger_pipeline(request: PipelineRequest, authorization: Optional[str] = Header(None)):
    if not PIPELINE_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="The public demo serves precomputed results. Run the pipeline locally or enable it on a dedicated persistent backend.",
        )

    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: configure valid pipeline credentials")
    
    if _cache["pipeline_status"] == "running":
        return {"message": "Pipeline already running"}

    _cache["pipeline_status"] = "running"

    # This route is deliberately for persistent processes only. Serverless
    # functions can be frozen as soon as the response is returned.
    import threading
    thread = threading.Thread(target=run_pipeline, args=(request.force_rerun, request.use_scraper), daemon=True)
    thread.start()

    return {"message": "Pipeline started"}

@app.get("/api/manifestoes")
def get_manifestoes():
    """Get list of loaded manifestoes."""
    if _cache["manifestoes"] is None:
        # The public demo ships a lightweight metadata cache. Prefer it over
        # re-parsing all PDFs on a cold worker; fall back to PDFs for local
        # pipeline development if the cache has not been generated yet.
        cache_path = PROCESSED_DIR / "manifestoes_cache.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as file:
                _cache["manifestoes"] = json.load(file)
        else:
            from .ingestion import load_all_manifestoes
            manifestoes = load_all_manifestoes()
            _cache["manifestoes"] = [{k: v for k, v in m.items() if k != 'raw_text'} for m in manifestoes]
    
    return {"manifestoes": _cache["manifestoes"] or [], "count": len(_cache["manifestoes"] or [])}


@app.get("/api/manifestoes/{party}/{year}/text")
def get_manifesto_text(party: str, year: str):
    """Get raw text of a specific manifesto."""
    from .ingestion import MANIFESTOES_DIR, load_manifesto
    
    for pdf_file in MANIFESTOES_DIR.glob("*.pdf"):
        meta_part = pdf_file.stem.lower()
        if party.lower() in meta_part and year in meta_part:
            data = load_manifesto(str(pdf_file))
            if data:
                return {
                    "party": data["party"],
                    "year": data["year"],
                    "text": data["raw_text"][:5000],  # First 5000 chars
                    "word_count": data["word_count"]
                }
    
    raise HTTPException(status_code=404, detail="Manifesto not found")


@app.get("/api/promises")
def get_promises(
    party: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(20),
    offset: int = Query(0),
):
    data = _cache.get("promises")
    if not data:
        path = PROCESSED_DIR / "predictions.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            _cache["promises"] = data
    if not data:
        return {"promises": [], "total": 0}
    filtered = data
    if party:
        filtered = [p for p in filtered if p.get("party","").lower() == party.lower()]
    if category:
        filtered = [p for p in filtered if p.get("category","").lower() == category.lower()]
    if status:
        filtered = [p for p in filtered if p.get("completion_status","").lower() == status.lower()]
    total = len(filtered)
    return {"promises": filtered[offset:offset+limit], "total": total}


@app.get("/api/scores")
def get_scores():
    data = _cache.get("scores")
    if not data:
        path = PROCESSED_DIR / "party_scores.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            _cache["scores"] = data
    return {"scores": data or []}


@app.post("/api/scores/custom")
def get_custom_scores(request: ScoreRequest):
    """Get scores with custom user weights."""
    promises = _cache.get("promises") or []
    
    if not promises:
        predictions_path = PROCESSED_DIR / "predictions.json"
        if predictions_path.exists():
            with open(predictions_path, 'r') as f:
                promises = json.load(f)
    
    if not promises:
        raise HTTPException(status_code=400, detail="No promise data available. Run pipeline first.")
    
    from .scoring import score_all_parties, generate_recommendation
    
    cat_weights = request.category_weights.dict() if request.category_weights else None
    print(f"📊 Custom scoring: weights={cat_weights}, priority={request.priority_category}")
    scores = score_all_parties(promises, category_weights=cat_weights)
    recommendation = generate_recommendation(scores, request.priority_category)
    
    return {"scores": scores, "recommendation": recommendation}


@app.get("/api/recommendation")
def get_recommendation(priority_category: Optional[str] = Query(None)):
    """Get party recommendation."""
    scores = _cache.get("scores") or []
    if not scores:
        from .scoring import load_party_scores
        scores = load_party_scores()
    if not scores:
        return {"best_overall": None, "overall_rank": [], "rationale": {}}
    from .scoring import generate_recommendation
    return generate_recommendation(scores, priority_category)


@app.get("/api/clustering")
def get_clustering():
    """Get clustering results."""
    clustering = _cache.get("clustering")
    
    if clustering is None:
        clustering_path = PROCESSED_DIR / "clustering_results.json"
        if clustering_path.exists():
            with open(clustering_path, 'r') as f:
                clustering = json.load(f)
            _cache["clustering"] = clustering
    
    return clustering or {"clusters": [], "party_clusters": {}, "pca_coords": []}


@app.get("/api/apriori")
def get_apriori():
    """Get association rules."""
    apriori = _cache.get("apriori")
    
    if apriori is None:
        apriori_path = PROCESSED_DIR / "apriori_results.json"
        if apriori_path.exists():
            with open(apriori_path, 'r') as f:
                apriori = json.load(f)
            _cache["apriori"] = apriori
    
    return apriori or {"frequent_itemsets": [], "rules": []}


@app.get("/api/analytics/overview")
def get_analytics_overview():
    try:
        # -------------------------------
        # LOAD DATA
        # -------------------------------
        promises = _cache.get("promises")

        if not promises:
            path = PROCESSED_DIR / "predictions.json"
            if path.exists():
                with open(path, "r") as f:
                    promises = json.load(f)

        if not promises:
            return {"error": "No data. Run pipeline first."}

        # -------------------------------
        # CLEAN DATA
        # -------------------------------
        normalized = []
        for p in promises:
            prob = p.get("completion_probability")
            normalized.append({
                "party": p.get("party", "Unknown"),
                "category": p.get("category", "Others"),
                "status": p.get("completion_status", "Unknown"),
                "prob": prob if prob is not None else None
            })

        total = len(normalized)

        # -------------------------------
        # CATEGORY DISTRIBUTION
        # -------------------------------
        category_distribution = {}
        for p in normalized:
            cat = p["category"]
            category_distribution[cat] = category_distribution.get(cat, 0) + 1

        # -------------------------------
        # COMPLETION BREAKDOWN (FIXED)
        # -------------------------------
        status_counts = {}
        for p in normalized:
            s = p["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        # -------------------------------
        # PREDICTION SUMMARY (FIXED)
        # -------------------------------
        probs = [p["prob"] for p in normalized if p["prob"] is not None]

        avg_prob = sum(probs) / len(probs) if probs else 0

        prediction_summary = {
            "average_probability": round(avg_prob, 3),
            "high_confidence": sum(1 for p in probs if p > 0.7)
        }

        return {
            "total_promises": total,
            "parties": list(set(p["party"] for p in normalized)),
            "category_distribution": category_distribution,
            "completion_stats": status_counts,
            "prediction_summary": prediction_summary,
            "categories": list(category_distribution.keys())
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
@app.get("/api/analytics/category/{category}")
def get_category_analysis(category: str):
    """Get detailed analysis for a specific category."""
    promises = _cache.get("promises") or []
    if not promises:
        predictions_path = PROCESSED_DIR / "predictions.json"
        if predictions_path.exists():
            with open(predictions_path, 'r') as f:
                promises = json.load(f)
    
    cat_promises = [p for p in promises if p.get("category", "").lower() == category.lower()]
    
    if not cat_promises:
        raise HTTPException(status_code=404, detail=f"No promises found for category: {category}")
    
    # Stats
    party_stats = {}
    for p in cat_promises:
        party = p.get("party", "Unknown")
        if party not in party_stats:
            party_stats[party] = {"total": 0, "completed": 0, "avg_probability": 0, "probs": []}
        party_stats[party]["total"] += 1
        if p.get("completion_status") == "Completed":
            party_stats[party]["completed"] += 1
        if p.get("completion_probability") is not None:
            party_stats[party]["probs"].append(p["completion_probability"])
    
    for party in party_stats:
        probs = party_stats[party]["probs"]
        party_stats[party]["avg_probability"] = round(sum(probs) / len(probs), 3) if probs else 0
        party_stats[party]["completion_rate"] = round(
            party_stats[party]["completed"] / max(party_stats[party]["total"], 1), 3
        )
        del party_stats[party]["probs"]
    
    return {
        "category": category,
        "total_promises": len(cat_promises),
        "party_stats": party_stats,
        "sample_promises": cat_promises[:10]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
