"""
Semantic clustering (Phase 2: "Semantic clustering").

Groups embedded log events into semantic groups that Phase 3 (temporal +
dependency graph) builds nodes from, and that the evidence builder (Phase 4)
packages for the LLM instead of raw individual log lines.

Three algorithms:
  - agglomerative : cosine-distance threshold, no fixed k (default; sklearn only)
  - kmeans        : fixed k, fast, useful for ablations
  - hdbscan       : density-based, flags noise as -1

Strictness (nothing changes silently):
  * an unknown algorithm raises ValueError;
  * a missing optional dependency raises ImportError, unless the caller
    passes allow_fallback=True (demo/CI), in which case the change is
    recorded in `ClusterResult.params["fallback_from"]`;
  * `len(events) == len(embeddings) == len(labels)` is enforced; a mismatch
    raises instead of silently dropping events (no bare zip()).

`run_clustering` returns the labels AND the exact parameters used (algorithm,
threshold / k / min_cluster_size, seed, number of clusters, noise), so an
experiment's configuration is always recorded with its result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src import config
from src.preprocessing.schema import LogEvent


@dataclass
class ClusterResult:
    labels: np.ndarray
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithms. Each takes (embeddings, params) and returns (labels, extra_params)
# ---------------------------------------------------------------------------

def _cluster_hdbscan(embeddings: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    if len(embeddings) < 2:
        return np.full(len(embeddings), -1, dtype=int), {"implementation": None}
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"],
            metric="euclidean",  # embeddings are L2-normalized, so euclidean ~ cosine
        )
        impl = "hdbscan"
    except ImportError:
        from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3

        clusterer = HDBSCAN(
            min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"],
            metric="euclidean",
        )
        impl = "sklearn.cluster.HDBSCAN"
    return np.asarray(clusterer.fit_predict(embeddings), dtype=int), {"implementation": impl}


def _cluster_kmeans(embeddings: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import KMeans

    k = min(p["n_clusters"], max(1, len(embeddings)))
    model = KMeans(n_clusters=k, random_state=p["seed"], n_init="auto")
    return np.asarray(model.fit_predict(embeddings), dtype=int), {"n_clusters_requested": k}


def _cluster_agglomerative(embeddings: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int), {}
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=p["distance_threshold"],
        metric="cosine", linkage="average",
    )
    return np.asarray(model.fit_predict(embeddings), dtype=int), {}


_ALGORITHMS = {
    "hdbscan": _cluster_hdbscan,
    "kmeans": _cluster_kmeans,
    "agglomerative": _cluster_agglomerative,
}


def run_clustering(
    embeddings: np.ndarray,
    algorithm: Optional[str] = None,
    *,
    distance_threshold: Optional[float] = None,
    n_clusters: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    seed: Optional[int] = None,
    allow_fallback: bool = False,
) -> ClusterResult:
    algorithm = algorithm or config.CLUSTERING_ALGORITHM
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Unknown clustering algorithm '{algorithm}'. Choose one of {sorted(_ALGORITHMS)}."
        )
    embeddings = np.asarray(embeddings)
    if len(embeddings) == 0:
        return ClusterResult(np.array([], dtype=int), {"algorithm": algorithm, "n_events": 0,
                                                       "n_clusters": 0, "n_noise": 0,
                                                       "seed": seed if seed is not None else config.RANDOM_SEED})
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D (n_events, dim) array, got shape {embeddings.shape}.")

    params = {
        "distance_threshold": distance_threshold if distance_threshold is not None else config.AGGLOMERATIVE_DISTANCE_THRESHOLD,
        "n_clusters": n_clusters if n_clusters is not None else config.KMEANS_N_CLUSTERS,
        "min_cluster_size": min_cluster_size if min_cluster_size is not None else config.HDBSCAN_MIN_CLUSTER_SIZE,
        "min_samples": min_samples if min_samples is not None else config.HDBSCAN_MIN_SAMPLES,
        "seed": seed if seed is not None else config.RANDOM_SEED,
    }

    used = algorithm
    fallback_from = None
    try:
        labels, extra = _ALGORITHMS[algorithm](embeddings, params)
    except ImportError:
        if not allow_fallback:
            raise
        print(f"[clustering] WARNING: '{algorithm}' unavailable, using agglomerative "
              f"(recorded as fallback_from).")
        fallback_from, used = algorithm, "agglomerative"
        labels, extra = _cluster_agglomerative(embeddings, params)

    if len(labels) != len(embeddings):
        raise RuntimeError(
            f"Clustering returned {len(labels)} labels for {len(embeddings)} embeddings."
        )

    # Record only the parameters that the algorithm actually used.
    used_keys = {
        "agglomerative": ["distance_threshold"],
        "kmeans": [],   # k is recorded as n_clusters_requested by the algorithm itself
        "hdbscan": ["min_cluster_size", "min_samples"],
    }[used]
    out = {"algorithm": used, "seed": params["seed"], **{k: params[k] for k in used_keys}, **extra}
    if used == "agglomerative":
        out["metric"], out["linkage"] = "cosine", "average"  # fixed in this implementation
    if fallback_from:
        out["fallback_from"] = fallback_from
    non_noise = labels[labels != -1]
    out["n_events"] = int(len(labels))
    out["n_clusters"] = int(len(np.unique(non_noise)))
    out["n_noise"] = int((labels == -1).sum())
    return ClusterResult(labels=labels, params=out)


def cluster_embeddings(embeddings: np.ndarray, algorithm: Optional[str] = None, **kwargs) -> np.ndarray:
    """Convenience wrapper returning only the labels (see `run_clustering`)."""
    return run_clustering(embeddings, algorithm, **kwargs).labels


def assign_clusters(
    events: list[LogEvent], embeddings: np.ndarray, algorithm: Optional[str] = None, **kwargs
) -> list[LogEvent]:
    """Clusters `embeddings` and writes `semantic_cluster` onto each event.
    Raises if events and embeddings are not aligned one-to-one."""
    result = assign_clusters_with_params(events, embeddings, algorithm, **kwargs)
    return result[0]


def assign_clusters_with_params(
    events: list[LogEvent], embeddings: np.ndarray, algorithm: Optional[str] = None, **kwargs
) -> tuple[list[LogEvent], ClusterResult]:
    if len(events) != len(embeddings):
        raise ValueError(
            f"events and embeddings are not aligned: {len(events)} events vs "
            f"{len(embeddings)} embedding rows."
        )
    result = run_clustering(embeddings, algorithm, **kwargs)
    if not (len(events) == len(embeddings) == len(result.labels)):
        raise RuntimeError("events / embeddings / labels lengths differ after clustering.")
    for event, label in zip(events, result.labels):  # lengths verified above
        event.semantic_cluster = int(label)
    return events, result


def summarize_clusters(events: list[LogEvent]) -> list[dict]:
    """Builds the "Semantic Groups" hand-off artifact -- one entry per
    cluster with representative template, member count, services/hosts,
    severity mix and time span, ready for the temporal/dependency graph and
    the evidence builder to consume instead of raw log lines."""
    if any(e.semantic_cluster is None for e in events):
        raise ValueError("summarize_clusters: some events have no semantic_cluster; run assign_clusters first.")

    groups: dict[int, list[LogEvent]] = defaultdict(list)
    for e in events:
        groups[e.semantic_cluster].append(e)

    summaries = []
    for cluster_id, members in sorted(groups.items(), key=lambda kv: (kv[0] == -1, -len(kv[1]), kv[0])):
        templates: dict[str, int] = defaultdict(int)
        for m in members:
            templates[m.template if m.template is not None else m.message] += 1
        representative_template = max(templates.items(), key=lambda kv: kv[1])[0]

        timestamps = sorted(m.timestamp_iso for m in members if m.timestamp_iso)
        severities: dict[str, int] = defaultdict(int)
        for m in members:
            severities[m.severity] += 1

        summaries.append({
            "cluster_id": cluster_id,
            "is_noise": cluster_id == -1,
            "size": len(members),
            "representative_template": representative_template,
            "distinct_templates": len(templates),
            "services": sorted({m.service for m in members if m.service}),
            "hosts": sorted({m.host for m in members if m.host}),
            "event_types": sorted({m.event_type for m in members if m.event_type}),
            "severity_distribution": dict(severities),
            "time_span": {
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
            },
            "event_ids": [m.event_id for m in members],
        })
    return summaries
