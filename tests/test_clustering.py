import numpy as np
import pytest

from src import config
from src.event_intelligence import clustering
from src.event_intelligence.clustering import (
    assign_clusters, cluster_embeddings, run_clustering, summarize_clusters,
)
from src.preprocessing.schema import LogEvent


def _blobs(n_per=20, centers=((0, 0, 1), (0, 1, 0), (1, 0, 0)), noise=0.02, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    pts, truth = [], []
    for i, c in enumerate(centers):
        pts.append(np.array(c, dtype="float32") + rng.normal(scale=noise, size=(n_per, len(c))))
        truth += [i] * n_per
    x = np.vstack(pts).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True), np.array(truth)


def _events(n, **kw):
    return [LogEvent(event_id=f"e{i}", source_file="f.log", line_number=i, raw_message="r", message="m", **kw)
            for i in range(n)]


# --- basic behaviour -----------------------------------------------------------------

def test_clustering_runs_on_small_embedding_matrix():
    rng = np.random.default_rng(config.RANDOM_SEED)
    embeddings = rng.normal(size=(20, 8)).astype("float32")
    labels = cluster_embeddings(embeddings, algorithm="agglomerative")
    assert len(labels) == 20


@pytest.mark.parametrize("algorithm", ["agglomerative", "kmeans", "hdbscan"])
def test_empty_input(algorithm):
    labels = cluster_embeddings(np.zeros((0, 0), dtype="float32"), algorithm=algorithm)
    assert len(labels) == 0


@pytest.mark.parametrize("algorithm,expected", [("agglomerative", 0), ("kmeans", 0), ("hdbscan", -1)])
def test_single_event(algorithm, expected):
    labels = cluster_embeddings(np.array([[1.0, 0.0, 0.0]], dtype="float32"), algorithm=algorithm)
    assert list(labels) == [expected]      # hdbscan cannot form a cluster of 1 -> noise


def test_multiple_similar_events_form_one_cluster():
    x, _ = _blobs(n_per=15, centers=((0, 0, 1),), noise=0.01)
    assert len(set(cluster_embeddings(x, algorithm="agglomerative"))) == 1


@pytest.mark.parametrize("algorithm,kwargs", [
    ("agglomerative", {"distance_threshold": 0.2}),
    ("kmeans", {"n_clusters": 3}),
    ("hdbscan", {"min_cluster_size": 5, "min_samples": 3}),
])
def test_different_semantic_groups_are_separated(algorithm, kwargs):
    from sklearn.metrics import adjusted_rand_score

    x, truth = _blobs()
    labels = cluster_embeddings(x, algorithm=algorithm, **kwargs)
    assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)


def test_hdbscan_flags_outliers_as_noise():
    x, _ = _blobs(n_per=25, noise=0.01)
    outliers = np.array([[0.7, 0.7, 0.0], [-0.6, 0.6, 0.5], [0.0, -0.9, 0.4]], dtype="float32")
    outliers /= np.linalg.norm(outliers, axis=1, keepdims=True)
    labels = cluster_embeddings(np.vstack([x, outliers]), algorithm="hdbscan", min_cluster_size=10, min_samples=5)
    assert (labels == -1).any()
    assert set(labels[:75]) - {-1} == {0, 1, 2}         # the three real groups survive


def test_clustering_is_reproducible_for_a_seed():
    x, _ = _blobs(noise=0.3)
    a = cluster_embeddings(x, algorithm="kmeans", n_clusters=4, seed=5)
    b = cluster_embeddings(x, algorithm="kmeans", n_clusters=4, seed=5)
    assert np.array_equal(a, b)


# --- strictness -----------------------------------------------------------------------

def test_invalid_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown clustering algorithm"):
        cluster_embeddings(np.eye(3, dtype="float32"), algorithm="spectral")


def test_non_2d_embeddings_raise():
    with pytest.raises(ValueError, match="2-D"):
        cluster_embeddings(np.arange(6, dtype="float32"), algorithm="kmeans")


def test_missing_optional_dependency_is_an_error_unless_fallback_allowed(monkeypatch):
    def needs_missing_pkg(embeddings, p):
        raise ImportError("no module named fancy")

    monkeypatch.setitem(clustering._ALGORITHMS, "hdbscan", needs_missing_pkg)
    x, _ = _blobs()
    with pytest.raises(ImportError):
        run_clustering(x, "hdbscan")
    result = run_clustering(x, "hdbscan", allow_fallback=True)
    assert result.params["algorithm"] == "agglomerative"
    assert result.params["fallback_from"] == "hdbscan"


# --- alignment: events / embeddings / labels --------------------------------------------

def test_events_and_embeddings_must_be_aligned():
    x, _ = _blobs(n_per=5)                    # 15 rows
    with pytest.raises(ValueError, match="not aligned"):
        assign_clusters(_events(14), x)       # one event fewer -> would have been silently truncated by zip
    with pytest.raises(ValueError, match="not aligned"):
        assign_clusters(_events(16), x)


def test_label_count_mismatch_from_an_algorithm_is_an_error(monkeypatch):
    monkeypatch.setitem(clustering._ALGORITHMS, "kmeans", lambda e, p: (np.zeros(len(e) - 1, dtype=int), {}))
    x, _ = _blobs(n_per=5)
    with pytest.raises(RuntimeError, match="labels"):
        run_clustering(x, "kmeans")


def test_assign_clusters_writes_a_label_on_every_event():
    x, _ = _blobs(n_per=5)
    events = assign_clusters(_events(15), x, algorithm="kmeans", n_clusters=3)
    assert len(events) == len(x) == 15
    assert all(e.semantic_cluster is not None for e in events)
    assert len({e.semantic_cluster for e in events}) == 3


# --- recorded parameters -----------------------------------------------------------------

def test_result_records_the_configuration_used():
    x, _ = _blobs()
    r = run_clustering(x, "agglomerative", distance_threshold=0.2, seed=11)
    assert r.params["algorithm"] == "agglomerative" and r.params["distance_threshold"] == 0.2
    assert r.params["seed"] == 11 and r.params["n_clusters"] == 3 and r.params["n_noise"] == 0
    assert r.params["n_events"] == 60

    k = run_clustering(x, "kmeans", n_clusters=3)
    assert k.params["n_clusters_requested"] == 3 and k.params["n_clusters"] == 3

    h = run_clustering(x, "hdbscan", min_cluster_size=7, min_samples=2)
    assert h.params["min_cluster_size"] == 7 and h.params["min_samples"] == 2
    assert h.params["implementation"] in {"hdbscan", "sklearn.cluster.HDBSCAN"}


# --- cluster summary ---------------------------------------------------------------------

def test_cluster_summary_contents():
    events = [
        LogEvent(event_id="a", source_file="f", line_number=1, raw_message="r", message="m", template="T1",
                 service="auth", host="h1", event_type="auth_event", severity="INFO",
                 timestamp_iso="2026-09-20T08:00:05+00:00", semantic_cluster=0),
        LogEvent(event_id="b", source_file="f", line_number=2, raw_message="r", message="m", template="T1",
                 service="pay", host="h2", event_type="auth_event", severity="ERROR",
                 timestamp_iso="2026-09-20T08:00:01+00:00", semantic_cluster=0),
        LogEvent(event_id="c", source_file="f", line_number=3, raw_message="r", message="m", template="T2",
                 severity="WARN", semantic_cluster=1),
        LogEvent(event_id="d", source_file="f", line_number=4, raw_message="r", message="m", template="T3",
                 severity="INFO", semantic_cluster=-1),
    ]
    groups = summarize_clusters(events)
    by_id = {g["cluster_id"]: g for g in groups}
    assert groups[-1]["cluster_id"] == -1 and groups[-1]["is_noise"]        # noise sorted last
    g0 = by_id[0]
    assert g0["event_ids"] == ["a", "b"] and g0["size"] == 2
    assert g0["representative_template"] == "T1"
    assert g0["services"] == ["auth", "pay"] and g0["hosts"] == ["h1", "h2"]
    assert g0["event_types"] == ["auth_event"]
    assert g0["severity_distribution"] == {"INFO": 1, "ERROR": 1}
    assert g0["time_span"] == {"start": "2026-09-20T08:00:01+00:00", "end": "2026-09-20T08:00:05+00:00"}
    assert by_id[1]["time_span"] == {"start": None, "end": None}
    assert sorted(i for g in groups for i in g["event_ids"]) == ["a", "b", "c", "d"]   # nobody lost


def test_summary_requires_clustered_events():
    with pytest.raises(ValueError, match="assign_clusters"):
        summarize_clusters(_events(2))
