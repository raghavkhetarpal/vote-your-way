import json
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import preprocess_sentence

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Economy", "Education", "Healthcare", "Infrastructure", "Agriculture",
               "Women", "Youth", "Environment", "Defence", "Others"]


CATEGORY_KEYWORDS = {
    "Economy": ["economy", "gdp", "growth", "tax", "revenue", "inflation", "trade", "export",
                 "import", "fiscal", "monetary", "budget", "finance", "msme", "startup",
                 "investment", "business", "industry", "manufacture", "employment", "job"],
    "Education": ["education", "school", "college", "university", "student", "teacher",
                  "curriculum", "literacy", "scholarship", "skill", "training", "vocational",
                  "learning", "classroom", "research", "iit", "iim", "knowledge"],
    "Healthcare": ["health", "hospital", "doctor", "medicine", "medical", "clinic",
                   "vaccination", "disease", "treatment", "nurse", "insurance", "ayushman",
                   "nutrition", "maternal", "child health", "mental health", "pharmacy"],
    "Infrastructure": ["road", "highway", "bridge", "railway", "rail", "airport", "port",
                       "metro", "construction", "housing", "electricity", "power", "water",
                       "sanitation", "telecom", "broadband", "smart city", "urban"],
    "Agriculture": ["farmer", "agriculture", "crop", "irrigation", "soil", "harvest",
                    "msp", "rural", "village", "kisan", "fertilizer", "seed", "drought",
                    "flood", "animal husbandry", "dairy", "fishery"],
    "Women": ["women", "girl", "female", "gender", "empowerment", "maternity",
              "self help group", "beti", "mahila", "sexual harassment", "dowry", "widow"],
    "Youth": ["youth", "young", "generation", "startup", "entrepreneur", "internship",
              "sports", "recreation", "digital native", "innovation", "future"],
    "Environment": ["environment", "climate", "green", "renewable", "solar", "wind",
                    "pollution", "carbon", "forest", "biodiversity", "ecosystem",
                    "clean energy", "electric vehicle", "waste management"],
    "Defence": ["defence", "defense", "military", "army", "navy", "air force",
                "security", "border", "terrorism", "police", "national security",
                "soldier", "veteran", "weapon"],
}


def keyword_classify(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    if scores:
        return max(scores, key=scores.get)
    return "Others"


def build_training_data(promises: List[Dict]) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for p in promises:
        if p.get("source") == "groq" and p.get("category") in CATEGORIES:
            category = p["category"]
        else:
            category = keyword_classify(p["promise"])
        texts.append(p["promise"])
        labels.append(category)
    return texts, labels


def train_classifier(promises: List[Dict]):
    texts, labels = build_training_data(promises)

    if len(texts) < 5:
        print("Not enough data. Using keyword classification.")
        return None, None, None

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    preprocessed = [preprocess_sentence(t) for t in texts]
    X = vectorizer.fit_transform(preprocessed).toarray()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    if len(set(y)) > 1 and len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  RandomForest accuracy: {acc:.2f}")
    else:
        clf.fit(X, y)

    joblib.dump(clf, MODELS_DIR / "classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "classifier_vectorizer.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    return clf, vectorizer, le


def load_classifier():
    paths = [
        MODELS_DIR / "classifier.pkl",
        MODELS_DIR / "classifier_vectorizer.pkl",
        MODELS_DIR / "label_encoder.pkl"
    ]

    if all(p.exists() for p in paths):
        return (
            joblib.load(paths[0]),
            joblib.load(paths[1]),
            joblib.load(paths[2])
        )
    return None, None, None


def classify_promises(promises: List[Dict], retrain: bool = False):
    if not promises:
        return []

    clf, vectorizer, le = load_classifier()

    if clf is None or retrain:
        print("Training RandomForest classifier...")
        clf, vectorizer, le = train_classifier(promises)

    classified = []

    for p in promises:
        p_copy = dict(p)

        if clf is not None:
            preprocessed = preprocess_sentence(p["promise"])
            X = vectorizer.transform([preprocessed]).toarray()

            pred_idx = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]

            p_copy["category"] = le.inverse_transform([pred_idx])[0]
            p_copy["category_confidence"] = float(max(proba))
        else:
            p_copy["category"] = keyword_classify(p["promise"])
            p_copy["category_confidence"] = 0.6

        classified.append(p_copy)

    return classified
