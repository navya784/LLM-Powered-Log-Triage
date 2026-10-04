import pytest

from src.preprocessing import parsers
from src.preprocessing.parsers import make_event_id, parse_line


# --- one test per claimed format ------------------------------------------------

def test_bracket_parser():
    line = "2026-09-20 10:30:00 [ERROR] api: Database connection timeout"
    e = parse_line("test.log", 1, line)
    assert e.log_format == "bracket" and e.parser_name == "bracket"
    assert e.severity == "ERROR"
    assert e.service == "api"
    assert e.host is None and e.process is None and e.pid is None
    assert "timeout" in e.message.lower()
    assert e.timestamp_raw is not None


def test_syslog_parser():
    line = "Sep 20 08:01:03 10.0.0.11 ERROR payment-service[4213]: failed to connect to auth-service at 10.0.0.21: timeout after 342ms"
    e = parse_line("test.log", 1, line)
    assert e.log_format == "syslog" and e.parser_name == "syslog"
    assert e.host == "10.0.0.11"
    assert e.service == "payment-service"
    assert e.pid == "4213"
    assert e.severity == "ERROR"
    assert e.timestamp_raw == "Sep 20 08:01:03"
    assert e.message.startswith("failed to connect")
    # the syslog TAG is a program name, so process is legitimately filled here
    assert e.process == "payment-service"
    assert e.metadata["service_source"] == "syslog_tag"


def test_json_parser():
    line = ('{"level": "ERROR", "msg": "payment declined for order ORD-1234", "service": "payment-service", '
            '"timestamp": "2026-09-20T08:01:03Z", "host": "10.0.0.12", "request_id": "req-9", '
            '"trace_id": "abc123", "exception": "PaymentDeclined", "region": "eu"}')
    e = parse_line("test.jsonl", 1, line)
    assert e.log_format == "json_log" and e.parser_name == "json"
    assert e.severity == "ERROR"
    assert e.service == "payment-service"
    assert e.host == "10.0.0.12"
    assert e.request_id == "req-9" and e.trace_id == "abc123" and e.exception == "PaymentDeclined"
    assert e.metadata["extra_fields"] == {"region": "eu"}
    assert e.process is None and e.pid is None       # not in the line -> not invented


def test_apache_parser():
    line = '10.0.0.11 - - [20/Sep/2026:08:01:03 +0000] "GET /api/v1/orders HTTP/1.1" 502 1234'
    e = parse_line("access.log", 1, line)
    assert e.log_format == "apache_access" and e.parser_name == "apache"
    assert e.host == "10.0.0.11"
    assert e.severity == "ERROR"
    assert e.metadata["http_status"] == 502
    assert "502" in e.message
    # a raw access line carries no service / process / pid
    assert e.service is None and e.process is None and e.pid is None


def test_key_value_parser():
    line = 'ts=2026-09-20T08:01:03 level=WARN service=order-service host=10.0.0.13 request_id=r-1 msg="connection pool at 91% capacity"'
    e = parse_line("infra.log", 1, line)
    assert e.log_format == "key_value" and e.parser_name == "key_value"
    assert e.severity == "WARN"
    assert e.service == "order-service"
    assert e.host == "10.0.0.13"
    assert e.request_id == "r-1"
    assert e.message == "connection pool at 91% capacity"


def test_hdfs_parser():
    line = "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating"
    e = parse_line("HDFS_2k.log", 1, line, dataset_id="loghub_hdfs_2k")
    assert e.log_format == "hdfs" and e.parser_name == "hdfs"
    assert e.severity == "INFO"
    assert e.timestamp_raw == "081109 203615"
    assert e.message == "PacketResponder 1 for block blk_38865049064139660 terminating"
    # java thread id / logger class are NOT an OS pid / process
    assert e.pid is None and e.process is None
    assert e.metadata["thread_id"] == "148"
    assert e.metadata["component"] == "dfs.DataNode$PacketResponder"


def test_plain_text_fallback():
    line = "something completely unstructured happened"
    e = parse_line("misc.log", 3, line)
    assert e.log_format == "plain_text" and e.parser_name == "plain_text"
    assert e.metadata["parse_status"] == "fallback_plain_text"
    assert e.severity == "UNKNOWN"
    assert e.message == line and e.raw_message == line
    assert e.parse_confidence == 0.0
    assert e.service is None and e.timestamp_raw is None


# --- severity normalization ----------------------------------------------------------

@pytest.mark.parametrize("token,expected", [
    ("WARNING", "WARN"), ("CRIT", "CRITICAL"), ("FATAL", "CRITICAL"),
    ("ERR", "ERROR"), ("NOTICE", "INFO"), ("warning", "WARN"),
])
def test_severity_alias_normalization(token, expected):
    # through the syslog parser...
    line = f"Sep 20 08:01:03 10.0.0.11 {token} auth-service[1]: something happened"
    e = parse_line("t.log", 1, line)
    assert e.severity == expected
    assert e.severity_raw.upper() == token.upper()      # original token kept
    # ...through the JSON parser...
    e = parse_line("t.jsonl", 1, f'{{"level": "{token}", "msg": "x", "service": "s"}}')
    assert e.severity == expected
    # ...and through the bracket parser
    e = parse_line("t.log", 1, f"2026-09-20 10:30:00 [{token.upper()}] api: hello")
    assert e.severity == expected


def test_unknown_severity_stays_unknown():
    e = parse_line("t.jsonl", 1, '{"level": "LOUD", "msg": "x"}')
    assert e.severity == "UNKNOWN" and e.severity_raw == "LOUD"


# --- event ids ------------------------------------------------------------------------

def test_event_id_determinism():
    a = parse_line("f.log", 5, "hello world", dataset_id="ds1")
    b = parse_line("f.log", 5, "hello world", dataset_id="ds1")
    assert a.event_id == b.event_id
    assert make_event_id("ds1", "f.log", 5, "hello world") == a.event_id


def test_event_id_uniqueness():
    base = make_event_id("ds1", "f.log", 5, "hello")
    variants = {
        make_event_id("ds2", "f.log", 5, "hello"),      # different dataset, same file name
        make_event_id("ds1", "g.log", 5, "hello"),      # different file
        make_event_id("ds1", "f.log", 6, "hello"),      # different line
        make_event_id("ds1", "f.log", 5, "hello!"),     # different content
        make_event_id(None, "f.log", 5, "hello"),       # no dataset
    }
    assert base not in variants and len(variants) == 5
    # field boundaries are unambiguous ("a","bc" vs "ab","c")
    assert make_event_id("a", "bc", 1, "x") != make_event_id("ab", "c", 1, "x")
    # many distinct lines -> many distinct ids
    ids = {parse_line("f.log", i, f"line {i}", dataset_id="d").event_id for i in range(1, 501)}
    assert len(ids) == 500


# --- service / process are never invented -------------------------------------------

def test_no_hardcoded_gateway_service_for_access_logs():
    line = '10.0.0.11 - - [20/Sep/2026:08:01:03 +0000] "GET /x HTTP/1.1" 200 10'
    assert parse_line("any.log", 1, line).service is None


def test_service_hint_comes_from_dataset_config_and_is_recorded():
    line = '10.0.0.11 - - [20/Sep/2026:08:01:03 +0000] "GET /x HTTP/1.1" 200 10'
    e = parse_line("gw.log", 1, line, service_hint="edge-proxy")
    assert e.service == "edge-proxy"
    assert e.metadata["service_source"] == "dataset_config"
    assert e.process is None


def test_service_hint_never_overrides_service_in_the_line():
    e = parse_line("f.jsonl", 1, '{"level": "INFO", "msg": "x", "service": "auth-service"}', service_hint="other")
    assert e.service == "auth-service"
    assert e.metadata["service_source"] == "log_line"


def test_process_is_not_copied_from_service():
    for line in [
        "2026-09-20 10:30:00 [ERROR] api: boom",
        '{"level": "INFO", "msg": "x", "service": "auth-service"}',
        'ts=2026-09-20T08:01:03 level=INFO service=a host=h msg="m"',
    ]:
        e = parse_line("f.log", 1, line)
        assert e.service is not None
        assert e.process is None and e.pid is None


# --- raw preservation / provenance -------------------------------------------------------

def test_raw_message_and_location_are_preserved():
    line = "2026-09-20 10:30:00 [ERROR] api: Database connection timeout"
    e = parse_line("logs/app.log", 42, line, dataset_id="ds")
    assert e.raw_message == line
    assert e.source_file == "logs/app.log" and e.line_number == 42 and e.dataset_id == "ds"


def test_parse_confidence_reflects_extracted_fields():
    full = parse_line("f.jsonl", 1, '{"level": "INFO", "msg": "x", "timestamp": "2026-09-20T08:00:00Z"}')
    partial = parse_line("f.jsonl", 1, '{"msg": "x"}')
    assert full.parse_confidence == 1.0
    assert 0.0 < partial.parse_confidence < 1.0


def test_parser_that_raises_is_marked_failed_not_dropped(monkeypatch):
    def boom(line):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(parsers, "PARSERS", [("boom", boom)])
    e = parse_line("f.log", 1, "whatever")
    assert e.metadata["parse_status"] == "failed"
    assert "kaboom" in e.metadata["parse_error"]
    assert e.raw_message == "whatever" and e.log_format == "plain_text"
