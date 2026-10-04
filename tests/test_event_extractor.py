import pytest

from src.event_intelligence.event_extractor import classify_event_type, extract_event, mine_template
from src.preprocessing.normalizer import normalize_event
from src.preprocessing.parsers import parse_line


def test_template_mining_collapses_variable_tokens():
    t1 = mine_template("failed to connect to payment-service at 10.0.0.12: timeout after 342ms")
    t2 = mine_template("failed to connect to payment-service at 10.0.0.21: timeout after 981ms")
    assert t1 == t2
    assert "<IP>" in t1
    assert "<NUM>" in t1


def test_event_type_classification():
    assert classify_event_type("failed to connect to db: timeout") == "connection_failure"
    assert classify_event_type("user 1234 logged in") == "auth_event"
    assert classify_event_type("out of memory, restarting worker") == "resource_exhaustion"
    assert classify_event_type("database query failed: deadlock detected on table orders") == "db_error"
    assert classify_event_type("payment declined for order ORD-1: insufficient funds") == "payment_failure"
    assert classify_event_type("GET /api/v1/orders HTTP/1.1 -> 502 (1234 bytes)") == "http_error"


# Negative tests: a keyword alone must not decide the label.
@pytest.mark.parametrize("message", [
    "database connection established",
    "connected to database successfully",
    "database backup completed",
    "processed 404 records",
    "request completed in 500ms status=200",
    "timeout configured to 30s",
    "health check passed for database",
])
def test_event_type_does_not_overreach_on_keywords(message):
    assert classify_event_type(message) not in {"db_error", "connection_failure", "http_error"}


def test_database_connection_established_is_not_db_error():
    assert classify_event_type("database connection established") != "db_error"


def test_unmatched_message_is_other():
    assert classify_event_type("PacketResponder 1 for block blk_1 terminating") == "other"


def test_extraction_does_not_override_message_or_raw():
    e = parse_line("f.log", 1, "2026-09-20 10:30:00 [INFO] api: database connection established to 10.0.0.5")
    normalize_event(e)
    before = (e.raw_message, e.message, e.normalized_message)
    extract_event(e)
    assert (e.raw_message, e.message, e.normalized_message) == before
    assert e.template == "database connection established to <IP>"
    assert e.event_type == "other"


def test_extraction_reads_normalized_message():
    e = parse_line("f.log", 1, 'ts=2026-09-20T08:00:00 level=INFO service=a host=h msg="auth-service[42]: health check passed"')
    normalize_event(e)
    extract_event(e)
    assert e.template == "health check passed"       # not "auth-service[<NUM>]: health check passed"
