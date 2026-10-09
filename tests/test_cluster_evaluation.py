import numpy as np
import pytest

from src.event_intelligence.cluster_evaluation import (
    cluster_size_distribution, evaluate_clustering, external_metrics, internal_metrics,
)


def _two_blobs():
    a = np.tile([1.0, 0.0], (10, 1)) + np.linspace(0, 0.02, 10)[:, None] * [0, 1]
    b = np.tile([0.0, 1.0], (10, 1)) + np.linspace(0, 0.02, 10)[:, None] * [1, 0]
    x = np.vstack([a, b]).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True), np.array([0] * 10 + [1] * 10)


def test_size_distribution_and_noise():
    d = cluster_size_distribution(np.array([0, 0, 0, 1, 1, -1]))
    assert d["sizes"] == [3, 2] and d["n_clusters"] == 2
    assert d["n_noise"] == 1 and d["noise_fraction"] == pytest.approx(1 / 6, abs=1e-5)
    assert cluster_size_distribution(np.array([]))["n_clusters"] == 0


def test_internal_metrics_on_well_separated_clusters():
    x, y = _two_blobs()
    m = internal_metrics(x, y)
    assert m["silhouette_cosine"] > 0.9 and m["davies_bouldin"] < 0.2


def test_internal_metrics_are_none_not_zero_when_undefined():
    x, _ = _two_blobs()
    m = internal_metrics(x, np.zeros(20, dtype=int))
    assert m["silhouette_cosine"] is None and m["davies_bouldin"] is None and m["notes"]
    m = internal_metrics(x, np.arange(20))
    assert m["silhouette_cosine"] is None


def test_internal_metrics_ignore_noise_points():
    x, y = _two_blobs()
    y = y.copy()
    y[0] = -1
    assert internal_metrics(x, y)["silhouette_cosine"] > 0.9


def test_external_metrics_perfect_and_imperfect():
    truth = ["a"] * 5 + ["b"] * 5
    perfect = external_metrics(np.array([3] * 5 + [7] * 5), truth)       # label ids need not match
    assert perfect["ari"] == 1.0 and perfect["nmi"] == 1.0
    assert external_metrics(np.zeros(10, dtype=int), truth)["ari"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        external_metrics(np.array([0, 1]), ["a"])
    with pytest.raises(ValueError):
        internal_metrics(np.zeros((3, 2)), np.array([0, 1]))


def test_evaluate_clustering_bundles_everything():
    x, y = _two_blobs()
    without = evaluate_clustering(x, y)
    assert without["external"] is None
    with_truth = evaluate_clustering(x, y, ground_truth=list(y))
    assert with_truth["external"]["ari"] == 1.0
    assert with_truth["size_distribution"]["n_clusters"] == 2


def test_purity_homogeneity_completeness():
    from src.event_intelligence.cluster_evaluation import purity

    truth = ["a"] * 5 + ["b"] * 5
    perfect = external_metrics(np.array([0] * 5 + [1] * 5), truth)
    assert perfect["purity"] == 1.0 and perfect["homogeneity"] == 1.0 and perfect["completeness"] == 1.0
    assert purity(np.zeros(10, dtype=int), truth) == 0.5          # one cluster holding two classes
    over_split = external_metrics(np.arange(10), truth)            # every event its own cluster
    assert over_split["purity"] == 1.0 and over_split["homogeneity"] == 1.0 and over_split["completeness"] < 1.0


def test_mixed_clusters_lists_where_grouping_disagrees_with_labels():
    from src.event_intelligence.cluster_evaluation import mixed_clusters

    labels = np.array([0, 0, 0, 0, 1, 1, 2])
    truth = ["a", "a", "a", "b", "c", "c", "d"]
    mixed = mixed_clusters(labels, truth)
    assert mixed == [{"cluster_id": 0, "size": 4, "truth_classes": {"a": 3, "b": 1}}]


def test_evaluation_states_its_scope():
    x, y = _two_blobs()
    note = evaluate_clustering(x, y)["scope_note"]
    assert "not a measure of root-cause" in note
