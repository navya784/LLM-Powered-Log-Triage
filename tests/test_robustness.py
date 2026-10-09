"""
Review 1: parsing failures, missing fields, timestamps and event ids.

Principle under test: bad input never stops a run and never loses a line, and
a field that is not in the line is None rather than invented.
"""

import json

import pytest

from src.preprocessing import parsers
from src.preprocessing.ingestion import ingest_files
from src.preprocessing.normalizer import normalize_event, normalize_events, normalize_timestamp
from src.preprocessing.parsers import make_event_id, parse_line
from src.preprocessing.schema import validate_event


def _parsed(line, **kw):
    e = parse_line("f.log", 1, line, dataset_id="d", **kw)
    normalize_events([e])
    return e


# --- parsing failures ---------------------------------------------------------------

@pytest.mark.parametrize("line", [
    '{"level": "ERROR", "msg": "cut off',          # truncated JSON
    "[1, 2, 3]",                                    # valid JSON, not an object
    "2026-13-45 99:99:99 [ERROR] api: impossible date",
    "Sep 20 08:01:03 10.0.0.11 ERROR auth: syslog line without [pid]",
    '10.0.0.1 - - [20/Sep/2026:08:01:03 +0000] "GET / HTTP/1.1"',   # access line without status
    "081109 203615 148 LOUD dfs.X: unknown severity word",
    "level=INFO msg=only-two-pairs",
    "",
    "   ",
    "abc\x00def",
    "x" * 20000,
])
def test_malformed_lines_fall_back_to_plain_text_and_keep_the_raw_line(line):
    e = _parsed(line)
    assert e.metadata["parse_status"] == "fallback_plain_text"
    assert e.log_format == "plain_text" and e.parse_confidence == 0.0
    assert e.raw_message == line and e.message == line
    assert e.severity == "UNKNOWN" and e.timestamp_iso is None
    assert validate_event(e, final=False) == []


def test_every_line_is_kept_even_when_a_parser_raises(tmp_path, monkeypatch):
    def explode_on_marker(line):
        if "BOOM" in line:
            raise RuntimeError("parser bug")
        return None

    monkeypatch.setattr(parsers, "PARSERS", [("flaky", explode_on_marker)])
    (tmp_path / "a.log").write_text("ok line\nBOOM line\nanother ok line\n")
    events, stats = ingest_files([tmp_path / "a.log"], tmp_path)
    assert [e.line_number for e in events] == [1, 2, 3]
    assert (stats.parsed, stats.fallback_plain_text, stats.failed) == (0, 2, 1)
    assert events[1].metadata["parse_error"].startswith("RuntimeError")
    assert events[1].raw_message == "BOOM line"


def test_json_with_non_string_fields_does_not_crash_the_pipeline():
    line = '{"level": "INFO", "msg": 123, "ts": 1758355200123, "host": 7, "service": "a", "pid": 42}'
    e = _parsed(line)
    assert e.log_format == "json_log"
    assert e.message == "123" and e.host == "7" and e.pid == "42"
    assert e.timestamp_iso == "2025-09-20T08:00:00.123000+00:00"
    assert validate_event(e, final=False) == []


def test_normalization_failure_is_recorded_not_fatal(monkeypatch):
    from src.preprocessing import normalizer

    e = parse_line("f.log", 1, "2026-09-20 10:30:00 [INFO] api: fine")
    ok = parse_line("f.log", 2, "2026-09-20 10:30:01 [INFO] api: also fine")

    def broken(msg):
        if "fine" in msg and "also" not in msg:
            raise ValueError("regex blew up")
        return msg

    monkeypatch.setattr(normalizer, "strip_embedded_prefix", broken)
    normalize_events([e, ok])
    assert "ValueError" in e.metadata["normalize_error"]
    assert e.normalized_message == e.message          # falls back to the parsed message
    assert "normalize_error" not in ok.metadata


# --- missing fields -------------------------------------------------------------------

def test_json_with_only_a_message():
    e = _parsed('{"msg": "just text"}')
    assert e.log_format == "json_log" and e.message == "just text"
    assert e.severity == "UNKNOWN" and e.severity_raw is None
    assert e.timestamp_raw is None and e.timestamp_iso is None
    assert e.service is None and e.host is None and e.request_id is None and e.trace_id is None
    assert 0 < e.parse_confidence < 1
    assert "timestamp_unparseable" not in e.metadata        # absent, not invalid


def test_json_with_no_known_fields_keeps_the_object_as_the_message():
    e = _parsed('{"foo": 1}')
    assert e.message == '{"foo": 1}' and e.severity == "UNKNOWN"
    assert e.metadata["extra_fields"] == {"foo": 1}


def test_null_and_empty_values_count_as_missing():
    e = _parsed('{"level": null, "msg": "x", "service": "", "host": null, "request_id": ""}')
    assert e.severity == "UNKNOWN" and e.service is None and e.host is None and e.request_id is None


def test_empty_dataset_hint_does_not_create_a_service():
    e = _parsed('{"level": "INFO", "msg": "x"}', service_hint=None)
    assert e.service is None and "service_source" not in e.metadata


def test_missing_fields_never_become_guesses():
    e = _parsed('10.0.0.1 - - [20/Sep/2026:08:01:03 +0000] "GET /x HTTP/1.1" 200 5')
    assert (e.service, e.process, e.pid, e.request_id, e.trace_id, e.exception) == (None,) * 6


# --- timestamps -----------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2026-09-20T08:00:00Z", "2026-09-20T08:00:00+00:00"),
    ("2026-09-20T08:00:00+05:30", "2026-09-20T02:30:00+00:00"),         # offset converted to UTC
    ("2026-09-20 08:00:00", "2026-09-20T08:00:00+00:00"),               # no timezone -> UTC
    ("20/Sep/2026:08:01:03 +0530", "2026-09-20T02:31:03+00:00"),
    ("081109 203615", "2008-11-09T20:36:15+00:00"),                      # HDFS compact form
    ("1758355200", "2025-09-20T08:00:00+00:00"),                         # epoch seconds
    ("1758355200123", "2025-09-20T08:00:00.123000+00:00"),               # epoch milliseconds
    ("  2026-09-20 08:00:00  ", "2026-09-20T08:00:00+00:00"),            # surrounding whitespace
])
def test_timestamps_are_normalized_to_utc(raw, expected):
    assert normalize_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["garbage", "", None, "2026-02-30 00:00:00", "081399 203615", "99999999999999999999"])
def test_unusable_timestamps_become_none_not_a_wrong_date(raw):
    assert normalize_timestamp(raw) is None


def test_unparseable_and_missing_timestamps_are_told_apart():
    bad = _parsed("081399 203615 148 INFO dfs.X: month 13")
    assert bad.timestamp_raw == "081399 203615" and bad.timestamp_iso is None
    assert bad.metadata["timestamp_unparseable"] is True
    missing = _parsed("plain text")
    assert missing.timestamp_raw is None and "timestamp_unparseable" not in missing.metadata


def test_year_less_timestamp_is_flagged_as_assumed():
    e = _parsed("Sep 20 08:01:03 10.0.0.11 INFO auth-service[1]: ok")
    assert e.timestamp_iso.startswith("2026-09-20") and e.metadata["timestamp_year_assumed"] is True


def test_timestamp_order_is_preserved_by_normalization():
    lines = [f"2026-09-20 10:30:{s:02d} [INFO] api: tick" for s in range(0, 50, 7)]
    iso = [_parsed(l).timestamp_iso for l in lines]
    assert iso == sorted(iso) and len(set(iso)) == len(iso)


# --- event ids ------------------------------------------------------------------------

def test_event_id_does_not_depend_on_parsing_or_normalization():
    line = "2026-09-20 10:30:00 [ERROR] api: boom"
    raw = parse_line("f.log", 3, line, dataset_id="d")
    before = raw.event_id
    normalize_event(raw)
    assert raw.event_id == before == make_event_id("d", "f.log", 3, line)


def test_event_id_is_stable_across_ingestion_runs_and_line_endings(tmp_path):
    (tmp_path / "unix.log").write_bytes(b"first\nsecond\n")
    (tmp_path / "windows.log").write_bytes(b"first\r\nsecond\r\n")
    a, _ = ingest_files([tmp_path / "unix.log"], tmp_path, dataset_id="d")
    a2, _ = ingest_files([tmp_path / "unix.log"], tmp_path, dataset_id="d")
    assert [e.event_id for e in a] == [e.event_id for e in a2]
    w, _ = ingest_files([tmp_path / "windows.log"], tmp_path, dataset_id="d")
    assert [e.raw_message for e in w] == ["first", "second"]       # no stray \r
    assert [e.raw_message for e in w] == [e.raw_message for e in a]


def test_identical_lines_on_different_line_numbers_get_different_ids(tmp_path):
    (tmp_path / "a.log").write_text("same\nsame\nsame\n")
    events, _ = ingest_files([tmp_path / "a.log"], tmp_path, dataset_id="d")
    assert len({e.event_id for e in events}) == 3


def test_event_id_is_a_fixed_width_hex_string():
    eid = make_event_id("d", "f.log", 1, "x")
    assert len(eid) == 16 and int(eid, 16) >= 0
    assert json.loads(json.dumps(eid)) == eid
