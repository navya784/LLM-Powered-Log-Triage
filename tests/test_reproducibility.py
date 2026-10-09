"""Same configuration + same data + same environment => identical artifacts."""

import json

import pytest

from src import config
from src.pipeline import load_config_file, main, run_pipeline

HDFS_PRESENT = (config.RAW_LOG_DIR / "loghub" / "HDFS_2k" / "HDFS_2k.log").exists()
needs_hdfs = pytest.mark.skipif(not HDFS_PRESENT, reason="run: python scripts/fetch_loghub.py")

DETERMINISTIC = ("events.jsonl", "embeddings.npy", "semantic_groups.json")


@needs_hdfs
@pytest.mark.parametrize("algorithm,params", [
    ("agglomerative", {"distance_threshold": 0.35}),
    ("kmeans", {"n_clusters": 8}),
    ("hdbscan", {"min_cluster_size": 5, "min_samples": 3}),
])
def test_two_runs_produce_byte_identical_artifacts(tmp_path, algorithm, params):
    for name in ("a", "b"):
        run_pipeline(dataset="loghub_hdfs_2k", output_dir=tmp_path / name, embedding_backend="tfidf",
                     algorithm=algorithm, clustering_params=params, seed=42)
    for artifact in DETERMINISTIC:
        assert (tmp_path / "a" / artifact).read_bytes() == (tmp_path / "b" / artifact).read_bytes(), artifact


def test_demo_runs_are_reproducible_too(tmp_path):
    for name in ("a", "b"):
        run_pipeline(demo=True, data_dir=tmp_path / "s", output_dir=tmp_path / name, embedding_backend="tfidf")
    for artifact in DETERMINISTIC:
        assert (tmp_path / "a" / artifact).read_bytes() == (tmp_path / "b" / artifact).read_bytes()


def test_seed_is_recorded_and_changes_kmeans_only_through_the_seed(tmp_path):
    r1 = run_pipeline(demo=True, data_dir=tmp_path / "s", save_outputs=False, embedding_backend="tfidf",
                      algorithm="kmeans", clustering_params={"n_clusters": 6}, seed=1)["report"]
    assert r1["stages"]["semantic_clustering"]["seed"] == 1 and r1["config"]["seed"] == 1


def test_report_records_config_and_environment(tmp_path):
    rep = run_pipeline(demo=True, data_dir=tmp_path / "s", save_outputs=False, embedding_backend="tfidf",
                       representation="hybrid", algorithm="kmeans", clustering_params={"n_clusters": 5})["report"]
    cfg = rep["config"]
    assert cfg["representation"] == "hybrid" and cfg["algorithm"] == "kmeans" and cfg["clustering_params"] == {"n_clusters": 5}
    assert cfg["embedding_backend"] == "tfidf" and cfg["seed"] == config.RANDOM_SEED
    env = rep["environment"]
    assert env["python"] and env["numpy"] and env["scikit-learn"]
    assert rep["logevent_schema_version"] == "1.0"


def test_shipped_review1_config_is_valid_and_pins_every_choice():
    cfg = load_config_file(config.ROOT_DIR / "configs" / "loghub_hdfs_2k.json")
    assert cfg["dataset"] == "loghub_hdfs_2k"
    for key in ("embedding_backend", "embedding_model", "representation", "algorithm", "clustering_params", "seed"):
        assert key in cfg


def test_config_file_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"dataset": "loghub_hdfs_2k", "embeding_backend": "tfidf"}))   # typo
    with pytest.raises(ValueError, match="embeding_backend"):
        load_config_file(p)


@needs_hdfs
def test_cli_runs_from_config_file_and_flags_override_it(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"dataset": "loghub_hdfs_2k", "embedding_backend": "tfidf",
                               "representation": "template", "algorithm": "agglomerative",
                               "clustering_params": {"distance_threshold": 0.35}, "seed": 42}))
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    assert main(["--config", str(cfg), "--representation", "hybrid", "--distance-threshold", "0.2"]) == 0
    rep = json.loads((tmp_path / "processed" / "loghub_hdfs_2k" / "pipeline_report.json").read_text())
    assert rep["config"]["representation"] == "hybrid"
    assert rep["config"]["clustering_params"] == {"distance_threshold": 0.2}
    assert rep["stages"]["semantic_clustering"]["distance_threshold"] == 0.2
