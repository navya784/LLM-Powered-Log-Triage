import pytest

from src import config
from src.preprocessing import ingestion
from src.preprocessing.datasets import (
    DATASETS, DEMO_SYNTHETIC, LOGHUB_HDFS_2K, DatasetMissingError, DatasetSpec, get_dataset,
)
from src.preprocessing.ingestion import ParseStats, find_dataset_files, ingest_dataset, ingest_files, iter_raw_lines


def _spec(tmp_path, **kw):
    return DatasetSpec(dataset_id="t", name="t", directory=tmp_path, file_patterns=("*.log",), **kw)


def test_ingestion_module_does_not_generate_data():
    assert not hasattr(ingestion, "generate_sample_dataset")
    assert not hasattr(ingestion, "load_dataset")


def test_missing_directory_raises_instead_of_generating(tmp_path):
    spec = DatasetSpec(dataset_id="t", name="t", directory=tmp_path / "nope", fetch_hint="python fetch.py")
    with pytest.raises(DatasetMissingError, match="fetch.py"):
        find_dataset_files(spec)


def test_empty_directory_raises_and_writes_nothing(tmp_path):
    with pytest.raises(DatasetMissingError):
        find_dataset_files(_spec(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_unknown_dataset_name_is_reported():
    with pytest.raises(DatasetMissingError, match="Unknown dataset"):
        get_dataset("does_not_exist")


def test_registry_separates_research_from_synthetic():
    assert LOGHUB_HDFS_2K.is_synthetic is False
    assert DEMO_SYNTHETIC.is_synthetic is True
    assert "demo_synthetic" in DATASETS


def test_iter_raw_lines_keeps_line_numbers_and_skips_blanks(tmp_path):
    p = tmp_path / "a.log"
    p.write_bytes(b"first\r\n\r\nthird\n   \nfifth")
    assert list(iter_raw_lines(p)) == [(1, "first"), (3, "third"), (5, "fifth")]


def test_parse_stats_partition_every_line(tmp_path):
    (tmp_path / "mixed.log").write_text(
        "2026-09-20 10:30:00 [ERROR] api: boom\n"
        "\n"
        '{"level": "INFO", "msg": "ok", "service": "s"}\n'
        "totally unstructured text\n"
        "more unstructured text\n"
    )
    events, stats = ingest_files([tmp_path / "mixed.log"], tmp_path, _spec(tmp_path))
    d = stats.to_dict()
    assert d["lines_read"] == 4 == len(events)
    assert d["parsed"] == 2 and d["fallback_plain_text"] == 2 and d["failed"] == 0
    assert d["parsed"] + d["fallback_plain_text"] + d["failed"] == d["lines_read"]
    assert d["parse_failure_rate"] == 0.0 and d["fallback_rate"] == 0.5
    assert d["blank_lines_skipped"] == 1
    assert d["formats"] == {"bracket": 1, "json_log": 1}
    assert d["per_file"]["mixed.log"]["lines_read"] == 4


def test_failed_lines_are_counted_and_kept(tmp_path, monkeypatch):
    from src.preprocessing import parsers

    def boom(line):
        if "explode" in line:
            raise ValueError("bad")
        return None

    monkeypatch.setattr(parsers, "PARSERS", [("boom", boom)])
    (tmp_path / "a.log").write_text("fine line\nexplode here\n")
    events, stats = ingest_files([tmp_path / "a.log"], tmp_path)
    assert (stats.failed, stats.fallback_plain_text, stats.parsed) == (1, 1, 0)
    assert stats.parse_failure_rate == 0.5
    assert len(events) == 2


def test_empty_stats_do_not_divide_by_zero():
    assert ParseStats().to_dict()["parse_failure_rate"] == 0.0


def test_event_ids_are_dataset_aware(tmp_path):
    (tmp_path / "a.log").write_text("same line\n")
    e1, _ = ingest_files([tmp_path / "a.log"], tmp_path, dataset_id="ds1")
    e2, _ = ingest_files([tmp_path / "a.log"], tmp_path, dataset_id="ds2")
    assert e1[0].event_id != e2[0].event_id


def test_service_mapping_comes_from_dataset_config(tmp_path):
    (tmp_path / "gateway_access.log").write_text('10.0.0.1 - - [20/Sep/2026:08:01:03 +0000] "GET /x HTTP/1.1" 200 5\n')
    (tmp_path / "other_access.log").write_text('10.0.0.1 - - [20/Sep/2026:08:01:03 +0000] "GET /x HTTP/1.1" 200 5\n')
    spec = _spec(tmp_path, service_map={"gateway_access.log": "gateway"})
    events, _ = ingest_files(sorted(tmp_path.glob("*.log")), tmp_path, spec)
    by_file = {e.source_file: e for e in events}
    assert by_file["gateway_access.log"].service == "gateway"
    assert by_file["gateway_access.log"].metadata["service_source"] == "dataset_config"
    assert by_file["other_access.log"].service is None       # no mapping -> not invented


@pytest.mark.skipif(not (config.RAW_LOG_DIR / "loghub" / "HDFS_2k" / "HDFS_2k.log").exists(),
                    reason="LogHub HDFS_2k not fetched (python scripts/fetch_loghub.py)")
def test_real_dataset_hdfs_2k_parses_completely():
    events, stats, files = ingest_dataset(LOGHUB_HDFS_2K)
    assert [f.name for f in files] == ["HDFS_2k.log"]
    assert stats.lines_read == 2000 == len(events)
    assert stats.parsed == 2000 and stats.failed == 0 and stats.fallback_plain_text == 0
    assert dict(stats.formats) == {"hdfs": 2000}
    assert all(e.dataset_id == "loghub_hdfs_2k" and e.source_file == "HDFS_2k.log" for e in events)
    assert len({e.event_id for e in events}) == 2000
    labels = LOGHUB_HDFS_2K.ground_truth_loader(LOGHUB_HDFS_2K.resolve_dir())
    assert len(labels) == 2000 and len(set(labels.values())) == 14
    assert all((e.source_file, e.line_number) in labels for e in events)
