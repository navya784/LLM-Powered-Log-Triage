"""
End-to-end checks for the Phase 1 + Phase 2 pipeline.

DEMO tests use generated fixture logs (connectivity only). RESEARCH tests use
the registered LogHub dataset. Embeddings use the TF-IDF backend explicitly so
the tests do not need to download a model; strictness about the real backend is
tested separately.
"""

import json

import pytest

from src import config
from src.event_intelligence import embeddings as emb
from src.event_intelligence.embeddings import EmbeddingBackendUnavailable
from src.pipeline import RunMode, main, run_pipeline
from src.preprocessing.datasets import DatasetMissingError

HDFS_PRESENT = (config.RAW_LOG_DIR / "loghub" / "HDFS_2k" / "HDFS_2k.log").exists()
needs_hdfs = pytest.mark.skipif(not HDFS_PRESENT, reason="run: python scripts/fetch_loghub.py")


# --- mode handling: no silent synthetic fallback ---------------------------------------------

def test_no_dataset_and_no_demo_reports_missing_dataset(tmp_path):
    with pytest.raises(DatasetMissingError, match="No dataset supplied"):
        run_pipeline(output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_research_run_never_generates_data_when_files_are_missing(tmp_path):
    with pytest.raises(DatasetMissingError):
        run_pipeline(dataset="loghub_hdfs_2k", data_dir=tmp_path / "empty", output_dir=tmp_path / "out",
                     embedding_backend="tfidf")
    assert not (tmp_path / "empty").exists()          # nothing was created
    assert not (tmp_path / "out").exists()


def test_demo_and_dataset_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        run_pipeline(dataset="loghub_hdfs_2k", demo=True)


def test_synthetic_dataset_cannot_be_used_as_research():
    with pytest.raises(ValueError, match="synthetic"):
        run_pipeline(dataset="demo_synthetic")


def test_cli_exit_codes(capsys):
    assert main([]) == 2
    assert "DATASET MISSING" in capsys.readouterr().err
    assert main(["--dataset", "demo_synthetic"]) == 2


# --- DEMO run -----------------------------------------------------------------------------------

def test_demo_run_is_labelled_and_writes_the_handoff_artifacts(tmp_path):
    out = tmp_path / "out"
    result = run_pipeline(demo=True, data_dir=tmp_path / "sample", output_dir=out, embedding_backend="tfidf")
    report = result["report"]
    assert report["mode"] == RunMode.DEMO.value and report["is_synthetic"] is True
    assert "not a research result" in report["warning"]
    assert (tmp_path / "sample").is_dir()             # fixtures were generated, explicitly, in DEMO mode

    events = result["events"]
    assert len(events) == 5 * 60
    assert result["embeddings"].shape[0] == len(events)
    assert all(e.semantic_cluster is not None and e.normalized_message is not None for e in events)
    assert all(e.dataset_id == "demo_synthetic" for e in events)

    for name in config.ARTIFACT_NAMES.values():
        assert (out / name).exists(), name
    groups = json.loads((out / config.ARTIFACT_NAMES["semantic_groups"]).read_text())
    assert groups["run"]["mode"] == "DEMO" and groups["run"]["is_synthetic"] is True
    assert groups["run"]["embedding"]["embedding_backend"] == "tfidf-svd"

    # the gateway access log gets its service from dataset config, and says so
    access = [e for e in events if e.source_file == "gateway_access.log"]
    assert access and all(e.service == "gateway" and e.metadata["service_source"] == "dataset_config" for e in access)
    assert all(e.process is None for e in access)


def test_parse_stats_are_in_the_report(tmp_path):
    report = run_pipeline(demo=True, data_dir=tmp_path / "s", output_dir=tmp_path / "o",
                          embedding_backend="tfidf")["report"]
    p = report["stages"]["parsing"]
    assert p["lines_read"] == 300 and p["parsed"] + p["fallback_plain_text"] + p["failed"] == 300
    assert set(p["formats"]) <= {"bracket", "syslog", "json_log", "apache_access", "key_value"}
    assert p["parse_failure_rate"] == 0.0


def test_demo_uses_existing_fixture_files_without_regenerating(tmp_path):
    from src.preprocessing.synthetic_generator import generate_sample_dataset

    generate_sample_dataset(out_dir=tmp_path / "s", n_per_format=7, seed=3)
    result = run_pipeline(demo=True, data_dir=tmp_path / "s", output_dir=tmp_path / "o", embedding_backend="tfidf",
                          save_outputs=False)
    assert len(result["events"]) == 35 and result["output_dir"] is None


# --- RESEARCH run ---------------------------------------------------------------------------------

def _no_real_embedding_model(monkeypatch):
    def unavailable(texts, model_name):
        raise EmbeddingBackendUnavailable("simulated: no model")
    monkeypatch.setattr(emb, "_embed_with_sentence_transformers", unavailable)


@needs_hdfs
def test_research_run_fails_instead_of_switching_embedding_backend(tmp_path, monkeypatch):
    _no_real_embedding_model(monkeypatch)
    with pytest.raises(EmbeddingBackendUnavailable):
        run_pipeline(dataset="loghub_hdfs_2k", output_dir=tmp_path / "o")        # default backend = sentence-transformers
    assert main(["--dataset", "loghub_hdfs_2k"]) == 3


def test_demo_run_may_fall_back_but_records_it(tmp_path, monkeypatch):
    _no_real_embedding_model(monkeypatch)
    report = run_pipeline(demo=True, data_dir=tmp_path / "s", output_dir=tmp_path / "o")["report"]
    assert report["stages"]["embeddings"]["fallback_from"] == "sentence-transformers"


@needs_hdfs
def test_research_run_on_loghub_hdfs_2k(tmp_path):
    out = tmp_path / "processed"
    result = run_pipeline(dataset="loghub_hdfs_2k", output_dir=out, embedding_backend="tfidf")
    report = result["report"]
    assert report["mode"] == "RESEARCH" and report["is_synthetic"] is False and "warning" not in report
    assert report["stages"]["parsing"]["lines_read"] == 2000
    assert report["stages"]["parsing"]["formats"] == {"hdfs": 2000}

    emb_info = report["stages"]["embeddings"]
    assert emb_info["representation"] == "template" and emb_info["embedding_backend"] == "tfidf-svd"
    assert emb_info["embedding_dimension"] == result["embeddings"].shape[1]

    clu = report["stages"]["semantic_clustering"]
    assert clu["algorithm"] == "agglomerative" and clu["distance_threshold"] == config.AGGLOMERATIVE_DISTANCE_THRESHOLD
    assert clu["seed"] == config.RANDOM_SEED and clu["n_clusters"] > 0
    ev = clu["evaluation"]
    assert ev["ground_truth"] == {"available": True, "coverage": 1.0}
    assert ev["external"]["ari"] is not None and ev["internal"]["silhouette_cosine"] is not None

    events = result["events"]
    assert {e.raw_message for e in events} and all(e.raw_message == e.raw_message.rstrip("\r\n") for e in events)
    assert len({e.event_id for e in events}) == 2000
    jsonl = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    assert len(jsonl) == 2000 and jsonl[0]["dataset_id"] == "loghub_hdfs_2k"


@needs_hdfs
def test_representation_switch_changes_what_is_embedded(tmp_path):
    runs = {}
    for rep in ("message", "template", "hybrid"):
        r = run_pipeline(dataset="loghub_hdfs_2k", output_dir=tmp_path / rep, embedding_backend="tfidf",
                         representation=rep, save_outputs=False)
        runs[rep] = r["report"]["stages"]["embeddings"]
        assert runs[rep]["representation"] == rep
    assert runs["message"]["n_unique_texts"] > runs["template"]["n_unique_texts"]


@needs_hdfs
def test_parquet_artifact_is_readable(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    run_pipeline(dataset="loghub_hdfs_2k", output_dir=tmp_path, embedding_backend="tfidf")
    table = pq.read_table(tmp_path / "events.parquet")
    assert table.num_rows == 2000
    assert {"event_id", "dataset_id", "raw_message", "normalized_message", "request_id", "trace_id",
            "exception", "metadata", "parser_name", "parse_confidence"} <= set(table.column_names)
