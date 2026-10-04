"""
Multi-format log parsing (Phase 1: "Improve parser" -> "Multi-format ingestion").

Extends the existing single-format parser in `log_preprocessor.py` (the
"[LEVEL] service: message" bracket format) rather than replacing it. This
module adds format *detection* on top, so any of several real-world log
shapes route to the right parser and land in the same `LogEvent` schema.

Add a new source format by writing a `_try_parse_X(line)` function and
registering it in `PARSERS` as `(name, function)` (order = priority; first
match wins).

Design rules (see docs/interface_contract.md):
  * A parser only returns what the line actually contains. It never invents
    a service, process or pid. Where a line carries no service (web access
    logs, HDFS ...) the *dataset configuration* supplies one -- via
    `service_hint` in `parse_line` -- and that provenance is recorded in
    `metadata["service_source"]`.
  * `service`, `process` and `pid` are different concepts. `process` is only
    set by a format whose line has a process/program name (syslog TAG).
  * Severity aliases (WARNING, CRIT, FATAL ...) are normalized BEFORE being
    validated against the canonical enum, so real severity information is
    not lost to UNKNOWN. The original token is kept in `severity_raw`.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Optional

from src import config
from src.preprocessing.log_preprocessor import extract_log_fields
from src.preprocessing.normalizer import normalize_severity
from src.preprocessing.schema import LogEvent, LogFormat, ParseStatus, Severity


_SEVERITY_TOKENS = "|".join(sorted(config.SEVERITY_ALIASES, key=len, reverse=True))

# Field-name variants seen in structured logs. Only used to *read* fields the
# line actually carries.
_REQUEST_ID_KEYS = ("request_id", "requestid", "req_id", "x-request-id")
_TRACE_ID_KEYS = ("trace_id", "traceid", "trace")
_EXCEPTION_KEYS = ("exception", "exc_info", "stack_trace", "stacktrace")
_PROCESS_KEYS = ("process", "process_name", "proc", "program")
_PID_KEYS = ("pid", "process_id")


def _first(d: dict, keys: tuple) -> Optional[str]:
    """First non-empty value among `keys` (case-insensitive), as a string."""
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = lowered.get(k)
        if v not in (None, ""):
            return str(v)
    return None


# ---------------------------------------------------------------------------
# Format 0: the existing bracket format, reused as-is
# "YYYY-MM-DD HH:MM:SS [LEVEL] service: message"
# ---------------------------------------------------------------------------

def _try_parse_bracket(line: str) -> Optional[dict]:
    fields = extract_log_fields(line)
    if fields is None:
        return None
    ts = fields.get("timestamp")
    return {
        "ts": ts.isoformat() if ts is not None else None,
        "host": None,
        "severity": (fields.get("level") or "").upper(),
        "service": fields.get("service"),
        "message": fields.get("message", line),
        "log_format": LogFormat.BRACKET.value,
    }


# ---------------------------------------------------------------------------
# Format 1: syslog-style
# "Sep 20 08:01:03 host SEVERITY tag[pid]: message"
# The syslog TAG is by definition the program (process) name, so it fills
# `process`; this project also uses it as the service name (recorded in
# metadata["service_source"]).
# ---------------------------------------------------------------------------

_SYSLOG_RE = re.compile(
    r"^(?P<ts>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    rf"(?P<severity>(?i:{_SEVERITY_TOKENS}))\s+"
    r"(?P<tag>[\w\-.]+)\[(?P<pid>\d+)\]:\s*"
    r"(?P<message>.*)$"
)


def _try_parse_syslog(line: str) -> Optional[dict]:
    m = _SYSLOG_RE.match(line)
    if not m:
        return None
    d = m.groupdict()
    tag = d.pop("tag")
    d["service"] = tag
    d["process"] = tag
    d["service_source"] = "syslog_tag"
    d["log_format"] = LogFormat.SYSLOG.value
    return d


# ---------------------------------------------------------------------------
# Format 2: Apache/nginx-style access log
# An access log line has NO service, process or pid. Service attribution, if
# wanted, comes from dataset configuration -- not from this parser.
# ---------------------------------------------------------------------------

_APACHE_RE = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<bytes>\S+)'
)


def _try_parse_apache(line: str) -> Optional[dict]:
    m = _APACHE_RE.match(line)
    if not m:
        return None
    d = m.groupdict()
    status = int(d["status"])
    d["severity"] = Severity.ERROR.value if status >= 500 else (
        Severity.WARN.value if status >= 400 else Severity.INFO.value
    )
    d["message"] = f'{d["request"]} -> {d["status"]} ({d["bytes"]} bytes)'
    d["log_format"] = LogFormat.APACHE_ACCESS.value
    d["metadata"] = {"http_status": status, "severity_source": "derived_from_http_status"}
    return d


# ---------------------------------------------------------------------------
# Format 3: JSON lines
# ---------------------------------------------------------------------------

_JSON_CONSUMED = {
    "timestamp", "time", "ts", "host", "hostname", "level", "severity",
    "service", "app", "msg", "message",
} | set(_REQUEST_ID_KEYS) | set(_TRACE_ID_KEYS) | set(_EXCEPTION_KEYS) | set(_PROCESS_KEYS) | set(_PID_KEYS)


def _try_parse_json(line: str) -> Optional[dict]:
    stripped = line.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    extra = {k: v for k, v in obj.items() if str(k).lower() not in _JSON_CONSUMED}
    return {
        "ts": obj.get("timestamp") or obj.get("time") or obj.get("ts"),
        "host": obj.get("host") or obj.get("hostname"),
        "severity": str(obj.get("level") or obj.get("severity") or "").upper(),
        "service": obj.get("service") or obj.get("app"),
        "process": _first(obj, _PROCESS_KEYS),
        "pid": _first(obj, _PID_KEYS),
        "request_id": _first(obj, _REQUEST_ID_KEYS),
        "trace_id": _first(obj, _TRACE_ID_KEYS),
        "exception": _first(obj, _EXCEPTION_KEYS),
        "message": obj.get("msg") or obj.get("message") or json.dumps(obj),
        "log_format": LogFormat.JSON_LOG.value,
        "metadata": {"extra_fields": extra} if extra else {},
    }


# ---------------------------------------------------------------------------
# Format 4: key=value
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
_KV_CONSUMED = {"ts", "timestamp", "host", "ip", "level", "service", "msg"} \
    | set(_REQUEST_ID_KEYS) | set(_TRACE_ID_KEYS) | set(_EXCEPTION_KEYS) | set(_PROCESS_KEYS) | set(_PID_KEYS)


def _try_parse_kv(line: str) -> Optional[dict]:
    pairs = dict(_KV_RE.findall(line))
    if len(pairs) < 3 or "level" not in {k.lower() for k in pairs}:
        return None
    norm = {k.lower(): v.strip('"') for k, v in pairs.items()}
    extra = {k: v for k, v in norm.items() if k not in _KV_CONSUMED}
    return {
        "ts": norm.get("ts") or norm.get("timestamp"),
        "host": norm.get("host") or norm.get("ip"),
        "severity": norm.get("level", "").upper(),
        "service": norm.get("service"),
        "process": _first(norm, _PROCESS_KEYS),
        "pid": _first(norm, _PID_KEYS),
        "request_id": _first(norm, _REQUEST_ID_KEYS),
        "trace_id": _first(norm, _TRACE_ID_KEYS),
        "exception": _first(norm, _EXCEPTION_KEYS),
        "message": norm.get("msg", line),
        "log_format": LogFormat.KEY_VALUE.value,
        "metadata": {"extra_fields": extra} if extra else {},
    }


# ---------------------------------------------------------------------------
# Format 5: LogHub HDFS
# "081109 203615 148 INFO dfs.DataNode$PacketResponder: message"
#   date(yymmdd) time(HHMMSS) thread-id LEVEL java-logger: message
# The number is a Java thread id and the "component" is a logger class name:
# neither is an OS pid / process, so they go to metadata, not pid / process.
# The line has no service; the dataset configuration provides one.
# ---------------------------------------------------------------------------

_HDFS_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<thread_id>\d+)\s+"
    rf"(?P<severity>(?i:{_SEVERITY_TOKENS}))\s+"
    r"(?P<component>[\w.$]+):\s*(?P<message>.*)$"
)


def _try_parse_hdfs(line: str) -> Optional[dict]:
    m = _HDFS_RE.match(line)
    if not m:
        return None
    d = m.groupdict()
    return {
        "ts": f'{d["date"]} {d["time"]}',
        "host": None,
        "severity": d["severity"].upper(),
        "message": d["message"],
        "log_format": LogFormat.HDFS.value,
        "metadata": {"component": d["component"], "thread_id": d["thread_id"]},
    }


def _fallback_plain_text(line: str) -> dict:
    return {
        "ts": None, "host": None,
        "severity": Severity.UNKNOWN.value,
        "service": None, "message": line,
        "log_format": LogFormat.PLAIN_TEXT.value,
    }


# Order matters: bracket first (existing, known-good parser), then most
# specific to least specific. Each entry is (parser_name, function).
PARSERS: list[tuple[str, Callable[[str], Optional[dict]]]] = [
    ("bracket", _try_parse_bracket),
    ("json", _try_parse_json),
    ("syslog", _try_parse_syslog),
    ("hdfs", _try_parse_hdfs),
    ("apache", _try_parse_apache),
    ("key_value", _try_parse_kv),
]


def _confidence(parsed: dict) -> float:
    """Structural completeness: share of the core fields (timestamp,
    severity, message) the parser extracted. A heuristic for filtering and
    diagnostics -- NOT a probability. Plain-text fallback is 0.0."""
    if parsed["parse_status"] != ParseStatus.PARSED.value:
        return 0.0
    has_ts = bool(parsed.get("ts"))
    has_sev = normalize_severity(parsed.get("severity")) != Severity.UNKNOWN.value
    has_msg = bool(parsed.get("message"))
    return round((has_ts + has_sev + has_msg) / 3, 3)


def detect_and_parse(line: str) -> dict:
    """Routes a line to the first parser that matches. Always returns a dict
    with `parser_name` and `parse_status`; a parser that raises does not
    crash ingestion -- the line is kept as plain text with status FAILED."""
    for name, parser in PARSERS:
        try:
            result = parser(line)
        except Exception as exc:  # noqa: BLE001 - one bad line must not stop ingestion
            result = _fallback_plain_text(line)
            result.update(
                parser_name=name,
                parse_status=ParseStatus.FAILED.value,
                metadata={"parse_error": f"{type(exc).__name__}: {exc}"},
            )
            return result
        if result is not None:
            result["parser_name"] = name
            result["parse_status"] = ParseStatus.PARSED.value
            return result
    result = _fallback_plain_text(line)
    result.update(parser_name="plain_text", parse_status=ParseStatus.FALLBACK_PLAIN_TEXT.value)
    return result


def make_event_id(dataset_id: Optional[str], source_file: str, line_number: int, raw: str) -> str:
    """Deterministic, dataset-aware id. Two files with the same name in
    different datasets get different ids; the same input always gets the
    same id. Fields are joined with an unambiguous separator (\\x1f)."""
    key = "\x1f".join([dataset_id or "", source_file, str(line_number), raw])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def parse_line(
    source_file: str,
    line_number: int,
    raw_line: str,
    *,
    dataset_id: Optional[str] = None,
    service_hint: Optional[str] = None,
) -> LogEvent:
    """Parses one raw line into a `LogEvent`.

    `service_hint` is the service that the *dataset configuration* attributes
    to this source. It is used only when the line itself carries no service,
    and that is recorded in `metadata["service_source"]`.
    """
    parsed = detect_and_parse(raw_line)

    severity_raw = parsed.get("severity") or None
    severity = normalize_severity(severity_raw)   # alias -> canonical BEFORE validation
    if severity not in {s.value for s in Severity}:
        severity = Severity.UNKNOWN.value

    metadata = dict(parsed.get("metadata") or {})
    metadata["parse_status"] = parsed["parse_status"]

    service = parsed.get("service") or None
    if service:
        metadata.setdefault("service_source", parsed.get("service_source", "log_line"))
    elif service_hint:
        service = service_hint
        metadata["service_source"] = "dataset_config"

    return LogEvent(
        event_id=make_event_id(dataset_id, source_file, line_number, raw_line),
        dataset_id=dataset_id,
        source_file=source_file,
        line_number=line_number,
        raw_message=raw_line,
        message=parsed.get("message", raw_line),
        timestamp_raw=parsed.get("ts"),
        host=parsed.get("host"),
        service=service,
        process=parsed.get("process"),
        pid=parsed.get("pid"),
        severity=severity,
        severity_raw=severity_raw,
        log_format=parsed.get("log_format", LogFormat.UNKNOWN.value),
        request_id=parsed.get("request_id"),
        trace_id=parsed.get("trace_id"),
        exception=parsed.get("exception"),
        parser_name=parsed["parser_name"],
        parse_confidence=_confidence(parsed),
        metadata=metadata,
    )
