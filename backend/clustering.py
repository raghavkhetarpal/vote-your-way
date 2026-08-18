import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import preprocess_sentence

MODELS_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Economy", "Education", "Healthcare", "Infrastructure", "Agriculture",
               "Women", "Youth", "Environment", "Defence", "Others"]


def build_party_feature_vectors(promises: List[Dict]) -> Tuple[Dict, np.ndarray, List[str]]:
    party_promises = {}
    for p in promises:
        party = p.get("party", "Unknown")
        party_promises.setdefault(party, []).append(p)

    party_labels = list(party_promises.keys())

    party_texts = [
        " ".join(preprocess_sentence(p["promise"]) for p in party_promises[party])
        for party in party_labels
    ]

    vec = TfidfVectorizer(max_features=100, ngram_range=(1, 2), sublinear_tf=True)

    if len(party_texts) >= 2:
        tfidf_matrix = vec.fit_transform(party_texts).toarray()
    else:
        tfidf_matrix = np.zeros((len(party_labels), 100))

    cat_features = []
    for party in party_labels:
        party_cats = [p.get("category", "Others") for p in party_promises[party]]
        cat_dist = [party_cats.count(cat) / max(len(party_cats), 1) for cat in CATEGORIES]
        cat_features.append(cat_dist)

    cat_features = np.array(cat_features)

    count_features = np.array([[len(party_promises[p])] for p in party_labels], dtype=float)

    completion_features = []
    for party in party_labels:
        p_list = party_promises[party]
        statuses = [p.get("completion_status", "Not Started") for p in p_list]
        total = max(len(statuses), 1)

        completion_features.append([
            statuses.count("Completed") / total,
            statuses.count("In Progress") / total,
            statuses.count("Failed") / total
        ])

    completion_features = np.array(completion_features)

    combined = np.hstack([tfidf_matrix, cat_features, count_features, completion_features])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined) if combined.shape[0] > 1 else combined

    return party_promises, combined_scaled, party_labels


def cluster_parties(promises: List[Dict], n_clusters: int = None) -> Dict:
    if not promises:
        return {"clusters": [], "party_clusters": {}, "pca_coords": []}

    party_promises, feature_matrix, party_labels = build_party_feature_vectors(promises)

    n_parties = len(party_labels)

    if n_parties < 2:
        return {
            "clusters": [{"id": 0, "parties": party_labels, "size": n_parties}],
            "party_clusters": {p: 0 for p in party_labels},
            "pca_coords": [[0.0, 0.0] for _ in party_labels],
            "party_labels": party_labels
        }

    if n_clusters is None:
        n_clusters = min(max(2, n_parties // 2), 5)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(feature_matrix)

    # PCA Visualization
    if feature_matrix.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(feature_matrix)
    else:
        coords_2d = np.zeros((n_parties, 2))

    clusters = {}
    for i, (party, cid) in enumerate(zip(party_labels, cluster_labels)):
        cid = int(cid)
        clusters.setdefault(cid, {"id": cid, "parties": [], "size": 0})
        clusters[cid]["parties"].append(party)
        clusters[cid]["size"] += 1

    result = {
        "clusters": list(clusters.values()),
        "party_clusters": {party: int(cid) for party, cid in zip(party_labels, cluster_labels)},
        "pca_coords": [
            {
                "party": party,
                "x": float(coords_2d[i][0]),
                "y": float(coords_2d[i][1]),
                "cluster": int(cluster_labels[i])
            }
            for i, party in enumerate(party_labels)
        ],
        "party_labels": party_labels,
        "n_clusters": n_clusters
    }

    output_path = PROCESSED_DIR / "clustering_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def cluster_promises_by_similarity(promises: List[Dict], n_clusters: int = 10):
    if len(promises) < n_clusters:
        n_clusters = max(2, len(promises) // 2)

    texts = [preprocess_sentence(p["promise"]) for p in promises]

    vec = TfidfVectorizer(max_features=200)
    X = vec.fit_transform(texts).toarray()

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    result = []
    for p, label in zip(promises, labels):
        p_copy = dict(p)
        p_copy["promise_cluster"] = int(label)
        result.append(p_copy)

    return result
