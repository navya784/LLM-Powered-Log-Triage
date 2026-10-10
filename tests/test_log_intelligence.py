"""Tests for src/log_intelligence.py (schema, parsing, IDs, timestamps, clustering, pipeline).

HDFS_2k end-to-end tests are skipped unless the Loghub sample is placed at
data/loghub/HDFS_2k/{HDFS_2k.log,labels.csv} (see the PR description).
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import numpy as np

try:  # the evidence layer may not be on the base branch yet
    from src.evidence import EventStore
except ImportError:
    EventStore = None

requires_evidence = pytest.mark.skipif(EventStore is None, reason="src.evidence not available")
from src.log_intelligence import (
    NOISE_CLUSTER_ID, SEVERITIES, ClusteringConfig, EmbeddingConfig, EmbeddingError,
    IngestConfig, LogEvent, ParseError, PipelineConfig, SentenceTransformerEmbedder,
    build_parsers, cluster_embeddings, config_hash, detect_parser, evaluate_events,
    export_json_schema, extract_template, ingest_file, ingest_lines, load_config,
    load_labels_csv, main, make_embedder, make_event_id, make_template_id,
    nearest_to_centroid, normalize_severity, parse_epoch, parse_iso_like, purity,
    run_pipeline, score_partition, tokenize_for_embedding,
)

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------
# SCHEMA
# ----------------------------------------------------------------------


def _event(**overrides):
    base = dict(
        event_id=make_event_id("a.log", 1, "x"),
        message="hello",
        raw="x",
        parser="project",
    )
    base.update(overrides)
    return LogEvent(**base)


# ---------------- event IDs ----------------
def test_event_id_is_deterministic():
    assert make_event_id("a.log", 5, "line") == make_event_id("a.log", 5, "line")


def test_event_id_format():
    assert make_event_id("a.log", 1, "x").startswith("evt_")
    assert len(make_event_id("a.log", 1, "x")) == len("evt_") + 16


def test_event_id_ignores_trailing_line_terminators():
    assert make_event_id("a.log", 1, "x\r\n") == make_event_id("a.log", 1, "x")


def test_identical_lines_on_different_lines_get_different_ids():
    assert make_event_id("a.log", 1, "same") != make_event_id("a.log", 2, "same")


def test_event_id_depends_on_source_and_content():
    base = make_event_id("a.log", 1, "x")
    assert base != make_event_id("b.log", 1, "x")
    assert base != make_event_id("a.log", 1, "y")


@pytest.mark.parametrize("bad", [0, -1])
def test_event_id_rejects_non_positive_line_numbers(bad):
    with pytest.raises(ValueError):
        make_event_id("a.log", bad, "x")


@pytest.mark.parametrize("bad", ["", "evt_123", "EVT_" + "a" * 16, "evt_" + "g" * 16, " evt_" + "a" * 16])
def test_malformed_event_ids_rejected(bad):
    with pytest.raises(ValidationError):
        _event(event_id=bad)


def test_template_id_format_and_stability():
    tid = make_template_id("a <*> b")
    assert tid == make_template_id("a <*> b") and tid != make_template_id("a <*> c")
    with pytest.raises(ValidationError):
        _event(template="t", template_id="bad")


# ---------------- timestamps ----------------
def test_timestamp_requires_utc_offset():
    with pytest.raises(ValidationError):
        _event(timestamp_iso="2024-01-01T00:00:00")


@pytest.mark.parametrize("bad", ["yesterday", "2024-13-01T00:00:00+00:00", "2024-01-01"[:5], ""])
def test_invalid_timestamps_rejected(bad):
    with pytest.raises(ValidationError):
        _event(timestamp_iso=bad)


def test_valid_timestamp_and_none_accepted():
    assert _event(timestamp_iso="2024-01-01T00:00:00+00:00").timestamp_iso
    assert _event(timestamp_iso=None).timestamp_iso is None


def test_tz_assumed_requires_timestamp():
    with pytest.raises(ValidationError):
        _event(timestamp_iso=None, timestamp_tz_assumed=True)


# ---------------- other field validation / missing fields ----------------
def test_missing_optional_fields_are_none():
    ev = _event()
    assert ev.timestamp_iso is ev.severity is ev.service is ev.template is None
    assert ev.source_file is None and ev.line_number is None


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError):
        LogEvent(event_id=make_event_id("a", 1, "x"), raw="x", parser="p")  # no message


def test_severity_validation_and_normalization():
    assert _event(severity="WARN").severity == "WARN"
    with pytest.raises(ValidationError):
        _event(severity="WARNING")
    assert normalize_severity("warning") == "WARN"
    assert normalize_severity("Critical") == "FATAL"
    assert normalize_severity("nonsense") is None and normalize_severity(None) is None
    assert set(SEVERITIES) == {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}


def test_partial_requires_warning_and_blank_service_rejected():
    with pytest.raises(ValidationError):
        _event(parse_status="partial")
    assert _event(parse_status="partial", parse_warnings=("missing_service",))
    with pytest.raises(ValidationError):
        _event(service="  ")


def test_extra_fields_forbidden_and_model_frozen():
    with pytest.raises(ValidationError):
        _event(unknown_field=1)
    with pytest.raises(ValidationError):
        _event().message = "changed"


def test_line_number_must_be_positive_int():
    with pytest.raises(ValidationError):
        _event(line_number=0)
    with pytest.raises(ValidationError):
        _event(line_number="3")


def test_template_requires_template_id():
    with pytest.raises(ValidationError):
        _event(template="a <*>")


# ---------------- interface contract ----------------
def test_events_round_trip_through_json():
    ev = _event(timestamp_iso="2024-01-01T00:00:00+00:00", severity="ERROR", service="api")
    assert LogEvent.model_validate(json.loads(json.dumps(ev.to_record()))) == ev


def test_contract_fields_for_evidence_layer_present():
    props = export_json_schema()["properties"]
    for name in ("event_id", "timestamp_iso", "severity", "service", "message",
                 "template", "source_file", "line_number"):
        assert name in props


@requires_evidence
def test_evidence_event_store_accepts_log_events():
    events = [
        _event(event_id=make_event_id("a.log", i, f"m{i}"), message=f"m{i}",
               timestamp_iso="2024-01-01T00:00:00+00:00", severity="INFO",
               service="svc", template="m<*>", template_id=make_template_id("m<*>"),
               source_file="a.log", line_number=i)
        for i in (1, 2, 3)
    ]
    store = EventStore(events)
    assert len(store) == 3
    assert store.require(events[0].event_id).service == "svc"





# ----------------------------------------------------------------------
# INGEST
# ----------------------------------------------------------------------
def ingest(lines, **cfg):
    return ingest_lines(lines, "t.log", IngestConfig(**cfg))


# ---------------- formats ----------------
def test_project_format():
    r = ingest(["2026-09-20 10:30:00 [ERROR] database: Connection failed"], parser="project")
    ev = r.events[0]
    assert (ev.service, ev.severity, ev.message) == ("database", "ERROR", "Connection failed")
    assert ev.timestamp_iso == "2026-09-20T10:30:00+00:00" and ev.timestamp_tz_assumed
    assert ev.parse_status == "ok"


def test_hdfs_format_and_pid_attribute():
    r = ingest(["081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1 terminating"])
    ev = r.events[0]
    assert r.parser == "hdfs"
    assert ev.timestamp_iso == "2008-11-09T20:36:15+00:00"
    assert ev.attributes == {"pid": "148"} and ev.service == "dfs.DataNode$PacketResponder"
    assert ev.template == "PacketResponder <*> for block blk_<*> terminating"


def test_generic_iso_with_offset_is_converted_to_utc():
    r = ingest(["2024-03-01T12:00:00.250+05:30 WARN payments - slow upstream 900ms"], parser="generic_iso")
    ev = r.events[0]
    assert ev.timestamp_iso == "2024-03-01T06:30:00.250000+00:00" and not ev.timestamp_tz_assumed
    assert ev.severity == "WARN" and ev.service == "payments"


def test_generic_iso_zulu_and_comma_fraction():
    assert parse_iso_like("2024-01-01 00:00:00,123")[0] == "2024-01-01T00:00:00.123000+00:00"
    assert parse_iso_like("2024-01-01T00:00:00Z") == ("2024-01-01T00:00:00+00:00", False)


def test_syslog_flags_assumed_year_and_missing_severity():
    r = ingest(["Jun 14 15:16:01 host1 sshd[1234]: Failed password"], parser="syslog", default_year=2024)
    ev = r.events[0]
    assert ev.timestamp_iso == "2024-06-14T15:16:01+00:00"
    assert ev.attributes == {"host": "host1", "pid": "1234"}
    assert ev.parse_status == "partial"
    assert {"year_assumed", "missing_severity"} <= set(ev.parse_warnings)


def test_jsonl_with_aliases_and_extra_attributes():
    line = json.dumps({"ts": 1700000000, "lvl": "error", "app": "api", "msg": "boom", "req": 7, "nested": {"a": 1}})
    ev = ingest([line], parser="jsonl").events[0]
    assert ev.timestamp_iso == "2023-11-14T22:13:20+00:00" and not ev.timestamp_tz_assumed
    assert (ev.severity, ev.service, ev.message) == ("ERROR", "api", "boom")
    assert ev.attributes == {"req": "7"}  # nested objects are not flattened


# ---------------- timestamps ----------------
def test_epoch_seconds_and_milliseconds_agree():
    assert parse_epoch(1700000000)[0] == parse_epoch(1700000000000)[0]
    assert parse_epoch(True) is None


def test_invalid_calendar_timestamp_keeps_event_but_flags_it():
    ev = ingest(["2026-13-45 10:30:00 [ERROR] db: x"], parser="project").events[0]
    assert ev.timestamp_iso is None and ev.parse_status == "partial"
    assert "invalid_timestamp" in ev.parse_warnings


def test_invalid_hdfs_time_flagged():
    ev = ingest(["081109 256199 148 INFO dfs.X: hi"], parser="hdfs").events[0]
    assert ev.timestamp_iso is None and "invalid_timestamp" in ev.parse_warnings


def test_epoch_out_of_range_is_none():
    assert parse_epoch(1e30) is None


# ---------------- missing fields ----------------
def test_jsonl_missing_optional_fields_become_none_with_warnings():
    ev = ingest(['{"message": "only message"}'], parser="jsonl").events[0]
    assert ev.timestamp_iso is None and ev.severity is None and ev.service is None
    assert {"missing_timestamp", "missing_severity", "missing_service"} <= set(ev.parse_warnings)
    assert ev.parse_status == "partial"


def test_unknown_severity_flagged_not_guessed():
    ev = ingest(['{"message": "m", "level": "LOUD", "service": "s", "timestamp": "2024-01-01T00:00:00Z"}'],
                parser="jsonl").events[0]
    assert ev.severity is None and ev.parse_warnings == ("unknown_severity",)


def test_empty_message_is_partial():
    ev = ingest(["2026-09-20 10:30:00 [INFO] svc:"], parser="project").events[0]
    assert ev.message == "" and "empty_message" in ev.parse_warnings


# ---------------- failures ----------------
def test_unparseable_lines_become_failures_with_reasons():
    r = ingest(["garbage line", "2026-09-20 10:30:00 [INFO] svc: ok"], parser="project")
    assert len(r.events) == 1 and len(r.failures) == 1
    f = r.failures[0]
    assert (f.line_number, f.reason, f.raw) == (1, "no_match", "garbage line")


@pytest.mark.parametrize("line,reason", [
    ("{not json", "invalid_json"),
    ("plain text", "not_json_object"),
    ('{"a": ', "invalid_json"),
    ('{"level": "INFO"}', "missing_message"),
    ('{"message": 5}', "missing_message"),
    ("[1, 2]", "not_json_object"),
])
def test_jsonl_failure_reasons(line, reason):
    r = ingest([line], parser="jsonl")
    assert r.events == [] and r.failures[0].reason == reason


def test_blank_lines_skipped_and_counted():
    r = ingest(["", "   ", "2026-09-20 10:30:00 [INFO] svc: ok"], parser="project")
    assert r.blank_lines == 2 and len(r.events) == 1 and r.events[0].line_number == 3


def test_line_numbers_are_physical_and_accounting_balances():
    lines = ["2026-09-20 10:30:00 [INFO] svc: a", "", "junk", "2026-09-20 10:30:01 [INFO] svc: b"]
    r = ingest(lines, parser="project")
    assert [e.line_number for e in r.events] == [1, 4]
    assert r.accounting_ok()


def test_overlong_line_rejected():
    r = ingest(["2026-09-20 10:30:00 [INFO] svc: " + "x" * 1_000_001], parser="project")
    assert r.failures[0].reason == "line_too_long" and len(r.failures[0].raw) < 300


def test_unknown_parser_name_raises():
    with pytest.raises(ValueError):
        ingest(["x"], parser="nope")


def test_no_matching_parser_marks_every_line_failed():
    r = ingest(["alpha", "", "beta"])
    assert r.parser == "none" and r.events == [] and r.blank_lines == 1
    assert all(f.reason.startswith("no_parser_matched") for f in r.failures) and r.accounting_ok()


def test_empty_input():
    r = ingest([])
    assert r.events == [] and r.failures == [] and r.total_lines == 0


def test_detect_parser_empty_raises():
    with pytest.raises(ParseError):
        detect_parser([], build_parsers())


# ---------------- continuations ----------------
def test_stack_trace_lines_attach_to_previous_event():
    lines = [
        "2026-09-20 10:30:00 [ERROR] api: Unhandled exception",
        "    at com.example.Foo.bar(Foo.java:10)",
        "Caused by: java.io.IOException: reset",
        "2026-09-20 10:30:01 [INFO] api: next",
    ]
    r = ingest(lines, parser="project")
    assert len(r.events) == 2 and r.continuation_lines == 2 and r.failures == []
    first = r.events[0]
    assert first.message.splitlines()[0] == "Unhandled exception" and "Caused by" in first.message
    assert first.attributes["continuation_lines"] == "2"
    assert first.template == "Unhandled exception"  # template ignores the trace
    assert r.accounting_ok()


def test_leading_continuation_without_previous_event_is_failure():
    r = ingest(["    at orphan", "2026-09-20 10:30:01 [INFO] api: ok"], parser="project")
    assert len(r.failures) == 1 and r.failures[0].line_number == 1


def test_continuations_can_be_disabled():
    r = ingest(["2026-09-20 10:30:00 [ERROR] api: x", "    at foo"], parser="project", attach_continuations=False)
    assert len(r.failures) == 1


# ---------------- IDs through ingestion ----------------
LINES = ["2026-09-20 10:30:00 [INFO] svc: same", "2026-09-20 10:30:00 [INFO] svc: same"]


def test_ids_stable_across_runs_and_unique_for_duplicate_lines():
    a, b = ingest(LINES, parser="project"), ingest(LINES, parser="project")
    assert [e.event_id for e in a.events] == [e.event_id for e in b.events]
    assert a.events[0].event_id != a.events[1].event_id


def test_ids_do_not_depend_on_file_location(tmp_path):
    for d in ("x", "y"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "app.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    ids = [[e.event_id for e in ingest_file(tmp_path / d / "app.log").events] for d in ("x", "y")]
    assert ids[0] == ids[1]


# ---------------- file reading ----------------
def test_crlf_files_parse_cleanly(tmp_path):
    p = tmp_path / "w.log"
    p.write_bytes(b"2026-09-20 10:30:00 [INFO] svc: hi\r\n2026-09-20 10:30:01 [INFO] svc: there\r\n")
    r = ingest_file(p)
    assert [e.message for e in r.events] == ["hi", "there"] and "\r" not in r.events[0].raw


def test_invalid_utf8_is_replaced_and_flagged(tmp_path):
    p = tmp_path / "bad.log"
    p.write_bytes(b"2026-09-20 10:30:00 [INFO] svc: caf\xe9\n")
    ev = ingest_file(p).events[0]
    assert "\ufffd" in ev.message and "encoding_replaced" in ev.parse_warnings and ev.parse_status == "partial"


# ---------------- normalization ----------------
@pytest.mark.parametrize("message,template", [
    ("Received block blk_-123 of size 67108864 from /10.251.42.84", "Received block blk_<*> of size <*> from /<*>"),
    ("retry 3 after 1.5s", "retry <*> after <*>s"),
    ("user 550e8400-e29b-41d4-a716-446655440000 logged in", "user <*> logged in"),
    ("loaded /var/lib/app/data.db ok", "loaded <*> ok"),
    ("addr 0xDEADBEEF bad", "addr <*> bad"),
    ("delete blk_1 blk_2 blk_3", "delete blk_<*>"),
    ("  lots   of\tspace ", "lots of space"),
    ("", ""),
])
def test_template_extraction(message, template):
    assert extract_template(message)[0] == template


def test_template_parameters_in_order():
    assert extract_template("a 1 b 2.5 c")[1] == ("1", "2.5")


def test_same_statement_same_template_different_statement_different():
    t1 = extract_template("Connection to 10.0.0.1:80 failed after 3 retries")[0]
    t2 = extract_template("Connection to 10.9.9.9:443 failed after 12 retries")[0]
    t3 = extract_template("Disk usage high")[0]
    assert t1 == t2 and t1 != t3


# ----------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------
TEXTS = [
    "Connection to database timed out",
    "Connection to database refused",
    "Disk usage above threshold",
    "Disk usage critical threshold exceeded",
    "User login succeeded",
]


def embed(texts=TEXTS, **kw):
    return make_embedder(EmbeddingConfig(**kw)).embed(texts)


def test_embeddings_are_l2_normalised():
    norms = np.linalg.norm(embed(), axis=1)
    assert np.allclose(norms, 1.0)


def test_embeddings_are_deterministic():
    assert np.array_equal(embed(), embed())


def test_embedding_seed_is_recorded_in_description():
    assert make_embedder(EmbeddingConfig(seed=7)).describe()["seed"] == 7


def test_empty_text_list_and_empty_vocabulary_raise():
    with pytest.raises(EmbeddingError):
        embed([])
    with pytest.raises(EmbeddingError):
        embed(["<*> <*>", "123 456"])


def test_tokenizer_drops_placeholders_and_splits_camel_case():
    assert tokenize_for_embedding("addStoredBlock <*>").split() == ["add", "stored", "block"]


def test_tiny_corpus_skips_svd():
    assert embed(["alpha beta", "gamma delta"]).shape[0] == 2


def test_sentence_transformer_missing_dependency_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "sentence_transformers":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(EmbeddingError, match="not installed"):
        SentenceTransformerEmbedder(EmbeddingConfig(backend="sentence_transformer")).embed(["x"])


def test_similar_texts_cluster_together():
    ids = cluster_embeddings(embed(), [1] * 5, ClusteringConfig(distance_threshold=0.7)).cluster_ids
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2] and ids[4] not in (ids[0], ids[2])


def test_cluster_ids_are_canonical_largest_first():
    emb = embed()
    ids = cluster_embeddings(emb, [1, 1, 10, 10, 1], ClusteringConfig(distance_threshold=0.7)).cluster_ids
    assert ids[2] == ids[3] == "cluster_000"  # heaviest cluster gets the first id


def test_clustering_is_deterministic_for_all_algorithms():
    emb = embed()
    for cfg in (ClusteringConfig(), ClusteringConfig(algorithm="kmeans", n_clusters=3, seed=1),
                ClusteringConfig(algorithm="dbscan", eps=0.6, min_samples=1)):
        assert cluster_embeddings(emb, [1] * 5, cfg) == cluster_embeddings(emb, [1] * 5, cfg)


def test_kmeans_requires_k_and_caps_at_n():
    with pytest.raises(ValueError):
        ClusteringConfig(algorithm="kmeans")
    a = cluster_embeddings(embed(), [1] * 5, ClusteringConfig(algorithm="kmeans", n_clusters=50))
    assert a.n_clusters <= 5


def test_dbscan_labels_outliers_as_noise():
    emb = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = cluster_embeddings(emb, [1, 1, 1], ClusteringConfig(algorithm="dbscan", eps=0.1, min_samples=2)).cluster_ids
    assert ids[2] == NOISE_CLUSTER_ID and ids[0] == ids[1] != NOISE_CLUSTER_ID


def test_degenerate_inputs():
    assert cluster_embeddings(np.zeros((0, 0)), [], ClusteringConfig()).cluster_ids == ()
    assert cluster_embeddings(np.ones((1, 2)), [3], ClusteringConfig()).cluster_ids == ("cluster_000",)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ClusteringConfig(distance_threshold=0)
    with pytest.raises(ValueError):
        ClusteringConfig(bogus=1)


def test_nearest_to_centroid_prefers_heavy_member():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert nearest_to_centroid(emb, [1, 50], [0, 1]) == 1


# ---------------- evaluation ----------------
def test_perfect_and_degenerate_partitions():
    labels = ["a", "a", "b", "b"]
    perfect = score_partition(["x", "x", "y", "y"], labels)
    assert perfect["adjusted_rand_index"] == 1.0 and perfect["purity"] == 1.0
    single = score_partition(["z"] * 4, labels)
    assert single["homogeneity"] == 0.0 and single["completeness"] == 1.0


def test_purity_and_length_mismatch():
    assert purity(["x", "x", "x"], ["a", "a", "b"]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        score_partition(["x"], ["a", "b"])
    assert score_partition([], []) == {"n_events": 0}


def test_noise_counts_as_singletons_not_one_cluster():
    s = score_partition([NOISE_CLUSTER_ID] * 3, ["a", "a", "a"])
    assert s["n_clusters"] == 3 and s["completeness"] < 1.0


def test_evaluate_events_uses_only_labelled_events_and_reports_baselines():
    res = evaluate_events([1, 2, 3, None], ["c0", "c0", "c1", "c1"], ["t1", "t1", "t2", None],
                          {1: "A", 2: "A", 3: "B"})
    assert res["coverage"]["n_labelled"] == 3
    assert set(res["baselines"]) == {"template_identity", "single_cluster", "shuffled_assignments"}
    assert res["clustering"]["adjusted_rand_index"] == 1.0
    assert any("root cause" in c for c in res["caveats"])


def test_load_labels_csv_validates(tmp_path):
    ok = tmp_path / "ok.csv"
    ok.write_text("line_number,event_id\n1,A\n2,B\n", encoding="utf-8")
    assert load_labels_csv(ok) == {1: "A", 2: "B"}
    for name, body, msg in [
        ("dup.csv", "line_number,event_id\n1,A\n1,B\n", "duplicate"),
        ("badnum.csv", "line_number,event_id\nx,A\n", "bad line number"),
        ("miss.csv", "line,label\n1,A\n", "missing columns"),
    ]:
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            load_labels_csv(p)


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
HDFS = ROOT / "data" / "loghub" / "HDFS_2k"
SYNTH = (
    "2026-09-20 10:30:00 [ERROR] database: Connection to 10.0.0.1:5432 timed out after 30s\n"
    "2026-09-20 10:30:01 [ERROR] database: Connection to 10.0.0.2:5432 timed out after 45s\n"
    "2026-09-20 10:30:02 [WARN] api: Slow response 900ms for user 42\n"
    "this line is garbage\n"
    "2026-09-20 10:30:03 [WARN] api: Slow response 1200ms for user 77\n"
    "2026-09-20 10:30:04 [INFO] auth: User login succeeded\n"
)


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def synth_run(tmp_path):
    src = tmp_path / "app.log"
    src.write_text(SYNTH, encoding="utf-8")
    manifest = run_pipeline(src, tmp_path / "out", PipelineConfig())
    return tmp_path / "out", manifest


def test_all_artifacts_written_and_machine_readable(synth_run):
    out, manifest = synth_run
    for name in ("events.jsonl", "parse_failures.jsonl", "templates.json", "template_embeddings.npy",
                 "clusters.json", "event_clusters.jsonl", "run_manifest.json"):
        assert (out / name).exists(), name
    events = [LogEvent.model_validate(r) for r in read_jsonl(out / "events.jsonl")]
    assert len(events) == 5
    failures = read_jsonl(out / "parse_failures.jsonl")
    assert [f["line_number"] for f in failures] == [4]
    assert manifest["ingest"]["accounting_ok"] and manifest["ingest"]["n_failures"] == 1


def test_event_clusters_cover_every_event_and_templates_group(synth_run):
    out, _ = synth_run
    events = read_jsonl(out / "events.jsonl")
    assign = {r["event_id"]: r for r in read_jsonl(out / "event_clusters.jsonl")}
    assert set(assign) == {e["event_id"] for e in events}
    by_template = {}
    for e in events:
        by_template.setdefault(e["template_id"], set()).add(assign[e["event_id"]]["cluster_id"])
    assert all(len(c) == 1 for c in by_template.values())  # one template -> one cluster


@requires_evidence
def test_events_feed_the_evidence_event_store(synth_run):
    out, _ = synth_run
    store = EventStore(read_jsonl(out / "events.jsonl"))
    assert len(store) == 5


def test_reruns_are_byte_identical(tmp_path):
    src = tmp_path / "app.log"
    src.write_text(SYNTH, encoding="utf-8")
    run_pipeline(src, tmp_path / "a", PipelineConfig())
    run_pipeline(src, tmp_path / "b", PipelineConfig())
    for f in sorted(p.name for p in (tmp_path / "a").iterdir()):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f


def test_manifest_records_config_hash_and_dataset_hash(synth_run):
    _, manifest = synth_run
    assert manifest["config_sha256"] == config_hash(PipelineConfig())
    assert len(manifest["dataset"]["sha256"]) == 64 and manifest["artifacts"]["events.jsonl"]


def test_config_hash_changes_with_config():
    other = PipelineConfig(text_field="message")
    assert config_hash(other) != config_hash(PipelineConfig())


def test_empty_input_produces_valid_empty_artifacts(tmp_path):
    src = tmp_path / "empty.log"
    src.write_text("", encoding="utf-8")
    manifest = run_pipeline(src, tmp_path / "o", PipelineConfig())
    assert manifest["ingest"]["n_events"] == 0
    clusters = json.loads((tmp_path / "o" / "clusters.json").read_text(encoding="utf-8"))
    assert clusters["skipped"] == "no_events" and clusters["clusters"] == []


def test_all_unparseable_input_does_not_crash(tmp_path):
    src = tmp_path / "junk.log"
    src.write_text("nothing\nuseful\n", encoding="utf-8")
    manifest = run_pipeline(src, tmp_path / "o", PipelineConfig())
    assert manifest["ingest"]["n_failures"] == 2 and manifest["ingest"]["n_events"] == 0


def test_unembeddable_text_degrades_to_single_cluster(tmp_path):
    src = tmp_path / "nums.log"
    src.write_text("2026-09-20 10:30:00 [INFO] svc: 123\n2026-09-20 10:30:01 [INFO] svc: 456\n", encoding="utf-8")
    run_pipeline(src, tmp_path / "o", PipelineConfig())
    clusters = json.loads((tmp_path / "o" / "clusters.json").read_text(encoding="utf-8"))
    assert clusters["skipped"].startswith("embedding_failed") and clusters["n_clusters"] == 1


def test_message_text_field_supported(tmp_path):
    src = tmp_path / "app.log"
    src.write_text(SYNTH, encoding="utf-8")
    m = run_pipeline(src, tmp_path / "o", PipelineConfig(text_field="message"))
    assert m["n_clusters"] >= 1


def test_yaml_config_loading(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("clustering:\n  distance_threshold: 0.3\ntext_field: message\n", encoding="utf-8")
    c = load_config(str(cfg))
    assert c.clustering.distance_threshold == 0.3 and c.text_field == "message"
    bad = tmp_path / "bad.yaml"
    bad.write_text("clustering:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_cli_run_and_schema(tmp_path, capsys):
    src = tmp_path / "app.log"
    src.write_text(SYNTH, encoding="utf-8")
    assert main(["run", "--input", str(src), "--out", str(tmp_path / "o")]) == 0
    assert "events=5" in capsys.readouterr().out
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "LogEvent"


# ---------------- the Review-1 demonstration ----------------
@pytest.mark.skipif(not (HDFS / "HDFS_2k.log").exists(), reason="HDFS_2k dataset not present")
def test_hdfs_2k_raw_lines_to_events_to_semantic_groups(tmp_path):
    cfg = PipelineConfig()  # defaults == the documented HDFS_2k configuration
    manifest = run_pipeline(HDFS / "HDFS_2k.log", tmp_path, cfg, labels_path=HDFS / "labels.csv")

    ing = manifest["ingest"]
    assert (ing["parser"], ing["total_lines"], ing["n_events"], ing["n_failures"]) == ("hdfs", 2000, 2000, 0)
    events = read_jsonl(tmp_path / "events.jsonl")
    assert len({e["event_id"] for e in events}) == 2000
    assert all(e["timestamp_iso"] and e["severity"] and e["service"] for e in events)
    if EventStore is not None:
        assert len(EventStore(events)) == 2000

    ev = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert ev["coverage"]["n_labelled"] == 2000
    assert ev["clustering"]["adjusted_rand_index"] > 0.9
    assert ev["baselines"]["shuffled_assignments"]["adjusted_rand_index"] < 0.05
    # Not a root-cause claim: the evaluation must carry its own caveats.
    assert any("root cause" in c for c in ev["caveats"])


