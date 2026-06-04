from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import re
from pathlib import Path
from typing import List, Dict, Tuple
from preprocessing import preprocess_sentence

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load embedding model (global for efficiency)
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')


KEYWORD_GROUPS = {
    "budget": ["budget", "crore", "lakh", "billion", "million", "fund", "allocat", "invest", "expenditure", "fiscal"],
    "policy": ["policy", "act", "law", "legislation", "reform", "regulation", "bill", "amendment", "scheme"],
    "infrastructure": ["road", "highway", "bridge", "railway", "airport", "port", "metro", "smart city", "construction"],
    "social": ["poor", "farmer", "women", "child", "youth", "elderly", "tribal", "minority", "dalit", "backward"],
    "digital": ["digital", "internet", "technology", "ai", "data", "online", "cyber", "software", "startup"],
    "employment": ["job", "employ", "skill", "training", "apprentice", "wage", "salary", "labour", "work"],
    "security": ["security", "defence", "army", "police", "border", "terrorism", "safety", "crime"],
    "environment": ["environment", "climate", "green", "renewable", "solar", "energy", "pollution", "forest", "water"],
}


def compute_keyword_features(text: str) -> Dict[str, int]:
    text_lower = text.lower()
    features = {}
    for group, keywords in KEYWORD_GROUPS.items():
        features[f"has_{group}"] = int(any(kw in text_lower for kw in keywords))
    return features


def build_tfidf_vectorizer(texts: List[str], max_features: int = 500):
    preprocessed = [preprocess_sentence(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(preprocessed)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    return vectorizer, tfidf_matrix, preprocessed


def engineer_features(promises: List[Dict], vectorizer=None, fit=True):
    if not promises:
        return np.array([]), [], None

    texts = [p["promise"] for p in promises]

    # TF-IDF
    if fit or vectorizer is None:
        vectorizer, tfidf_matrix, _ = build_tfidf_vectorizer(texts)
    else:
        preprocessed = [preprocess_sentence(t) for t in texts]
        tfidf_matrix = vectorizer.transform(preprocessed)

    tfidf_dense = tfidf_matrix.toarray()

    # 🔥 NEW: Embeddings
    embeddings = EMBEDDER.encode(texts)

    # Handcrafted features
    handcrafted = []
    for p in promises:
        text = p["promise"]
        kw_features = compute_keyword_features(text)

        row = [
            len(text.split()) / 50.0,
            int(bool(re.search(r'\d+', text))),
            int(bool(re.search(r'\d+\s*%', text))),
            int(bool(re.search(r'\b(20\d\d)\b', text))),
            1 if p.get("specificity") == "high" else 0,
        ] + list(kw_features.values())

        handcrafted.append(row)

    handcrafted_arr = np.array(handcrafted)

    # 🔥 FINAL COMBINATION
    combined = np.hstack([tfidf_dense, embeddings, handcrafted_arr])

    return combined, [], vectorizer