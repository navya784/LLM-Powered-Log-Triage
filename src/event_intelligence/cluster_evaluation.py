"""
Clustering evaluation (Phase 2: "Semantic clustering" must be *evaluated*,
not just implemented).

Internal metrics need no labels and are always computed:
  silhouette (cosine), Davies-Bouldin, cluster-size distribution, noise %.
External metrics need ground truth and are computed only when the dataset
provides it:
  ARI, NMI (noise points, label -1, are treated as one extra group).

Internal metrics ignore noise points. A metric that is undefined for the
given clustering (e.g. silhouette with a single cluster) is reported as
None with the reason in `notes`, never as 0.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def cluster_size_distribution(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {"n_clusters": 0, "n_noise": 0, "noise_fraction": 0.0, "sizes": []}
    clustered = labels[labels != -1]
    _, counts = np.unique(clustered, return_counts=True)
    sizes = sorted((int(c) for c in counts), reverse=True)
    n_noise = int((labels == -1).sum())
    return {
        "n_clusters": len(sizes),
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(labels), 6),
        "sizes": sizes,
        "min_size": sizes[-1] if sizes else 0,
        "max_size": sizes[0] if sizes else 0,
        "median_size": float(np.median(sizes)) if sizes else 0.0,
        "singleton_clusters": sum(1 for s in sizes if s == 1),
    }


def internal_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    embeddings, labels = np.asarray(embeddings), np.asarray(labels)
    if len(embeddings) != len(labels):
        raise ValueError(f"embeddings ({len(embeddings)}) and labels ({len(labels)}) differ in length.")
    mask = labels != -1
    x, y = embeddings[mask], labels[mask]
    n_labels = len(np.unique(y))
    out: dict = {"silhouette_cosine": None, "davies_bouldin": None, "notes": []}
    if n_labels < 2:
        out["notes"].append("fewer than 2 clusters (excluding noise): internal metrics undefined")
        return out
    if n_labels > len(y) - 1:
        out["notes"].append("every point is its own cluster: internal metrics undefined")
        return out
    out["silhouette_cosine"] = round(float(silhouette_score(x, y, metric="cosine")), 6)
    out["davies_bouldin"] = round(float(davies_bouldin_score(x, y)), 6)
    return out


def external_metrics(labels: np.ndarray, truth: Sequence) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    labels = np.asarray(labels)
    if len(labels) != len(truth):
        raise ValueError(f"labels ({len(labels)}) and ground truth ({len(truth)}) differ in length.")
    if len(labels) == 0:
        return {"ari": None, "nmi": None, "n_ground_truth_classes": 0}
    return {
        "ari": round(float(adjusted_rand_score(truth, labels)), 6),
        "nmi": round(float(normalized_mutual_info_score(truth, labels)), 6),
        "n_ground_truth_classes": len(set(truth)),
        "noise_treated_as": "one extra group",
    }


def evaluate_clustering(
    embeddings: np.ndarray, labels: np.ndarray, ground_truth: Optional[Sequence] = None
) -> dict:
    result = {
        "size_distribution": cluster_size_distribution(labels),
        "internal": internal_metrics(embeddings, labels),
        "external": None,
    }
    if ground_truth is not None:
        result["external"] = external_metrics(labels, ground_truth)
    return result
