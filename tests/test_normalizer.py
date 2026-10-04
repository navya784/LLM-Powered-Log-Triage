from src.preprocessing.normalizer import (
    extract_entities, normalize_event, normalize_severity, normalize_timestamp, strip_embedded_prefix,
)
from src.preprocessing.parsers import parse_line


def test_normalize_severity_aliases():
    assert normalize_severity("WARNING") == "WARN"
    assert normalize_severity("crit") == "CRITICAL"
    assert normalize_severity("FATAL") == "CRITICAL"
    assert normalize_severity(None) == "UNKNOWN"
    assert normalize_severity("bogus") == "UNKNOWN"


def test_normalize_timestamp_syslog_style():
    iso = normalize_timestamp("Sep 20 08:01:03", reference_year=2026)
    assert iso is not None and iso.startswith("2026-09-20")


def test_normalize_timestamp_apache_style():
    iso = normalize_timestamp("20/Sep/2026:08:01:03 +0000")
    assert iso is not None and iso.startswith("2026-09-20")


def test_normalize_timestamp_hdfs_compact_style():
    assert normalize_timestamp("081109 203615") == "2008-11-09T20:36:15+00:00"
    assert normalize_timestamp("081399 203615") is None      # impossible month


def test_strip_embedded_prefix_collapses_duplicate_events():
    a = strip_embedded_prefix("auth-service[4213]: health check passed")
    assert a == "health check passed"


def test_extract_entities():
    entities = extract_entities("failed to connect to auth-service at 10.0.0.21: timeout after 342ms")
    assert entities["ips"] == ["10.0.0.21"]
    assert entities["latency_ms"] == 342


def test_normalization_preserves_raw_data_and_fills_normalized_fields():
    raw = 'ts=2026-09-20T08:01:03 level=WARNING service=Order_Service host=10.0.0.13 msg="auth-service[4213]: pool at 91%"'
    e = parse_line("f.log", 7, raw, dataset_id="d")
    before = (e.raw_message, e.message, e.timestamp_raw, e.source_file, e.line_number)
    normalize_event(e)
    assert (e.raw_message, e.message, e.timestamp_raw, e.source_file, e.line_number) == before
    assert e.normalized_message == "pool at 91%"          # prefix stripped here only
    assert e.message == "auth-service[4213]: pool at 91%"  # parsed message untouched
    assert e.severity == "WARN" and e.severity_raw == "WARNING"
    assert e.service == "order-service"
    assert e.timestamp_iso.startswith("2026-09-20")


def test_yearless_timestamps_are_flagged_as_assumed():
    e = parse_line("f.log", 1, "Sep 20 08:01:03 10.0.0.11 INFO auth-service[1]: ok")
    normalize_event(e)
    assert e.metadata.get("timestamp_year_assumed") is True
    e2 = parse_line("f.jsonl", 1, '{"level": "INFO", "msg": "x", "timestamp": "2026-09-20T08:00:00Z"}')
    normalize_event(e2)
    assert "timestamp_year_assumed" not in e2.metadata
