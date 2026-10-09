"""
Normalization (Phase 1, extends "Log normalization" already listed as
implemented in the README) — brings every parser's output into the
canonical schema: ISO-8601 UTC timestamps, a 5-level severity scale,
cleaned host/service names, and extracted entities (IPs, status codes,
latency, error codes) that later phases (evidence builder, LLM integration)
can read directly instead of re-parsing message strings.

Raw vs normalized: normalization NEVER overwrites the original data.
`raw_message` (the exact line), `message` (as parsed), `timestamp_raw` and
`severity_raw` are left untouched; normalized values go into
`normalized_message`, `timestamp_iso` and `severity`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from dateutil import parser as dateutil_parser

from src import config
from src.preprocessing.schema import LogEvent, Severity


_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_STATUS_RE = re.compile(r"\b([1-5]\d{2})\b(?=\s*(?:\(|bytes|$))")
_MS_RE = re.compile(r"\b(\d+)\s*ms\b", re.IGNORECASE)
_ERRCODE_RE = re.compile(r"\b(ERR|ERROR)[_\-]?(\d{2,5})\b", re.IGNORECASE)

# Strips a leading "service[pid]:" prefix that some formats embed inside the
# message body itself (e.g. key=value logs) so identical events collapse to
# the same message/template regardless of which parser produced them.
_EMBEDDED_PREFIX_RE = re.compile(r"^[\w\-.]+\[\d+\]:\s*")


# Timestamps that carry no year ("Sep 20 08:01:03"): the year is assumed.
_YEARLESS_RE = re.compile(r"^[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}$")
# Unix epoch: 10 digits = seconds, 13 digits = milliseconds (optionally with a fraction).
_EPOCH_RE = re.compile(r"^(\d{10}|\d{13})(\.\d+)?$")
# LogHub HDFS compact form: "yymmdd HHMMSS" (e.g. "081109 203615").
_COMPACT_YMD_HMS_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})\s+(\d{2})(\d{2})(\d{2})$")


def timestamp_year_is_assumed(ts_raw: Optional[str]) -> bool:
    return bool(ts_raw and _YEARLESS_RE.match(ts_raw.strip()))


def normalize_timestamp(ts_raw: Optional[str], reference_year: int = 2026) -> Optional[str]:
    """Parses a variety of timestamp formats into ISO-8601 UTC. Syslog
    timestamps ('Sep 20 08:01:03') have no year, so a reference year is
    injected (normalize_event flags this in metadata) — swap this for file
    mtime or a known collection window in production. Timestamps without a
    timezone are assumed to be UTC."""
    if not ts_raw:
        return None

    ts_raw = ts_raw.strip()
    compact = _COMPACT_YMD_HMS_RE.match(ts_raw)
    if compact:
        yy, mo, dd, hh, mi, ss = (int(g) for g in compact.groups())
        try:
            return datetime(2000 + yy, mo, dd, hh, mi, ss, tzinfo=timezone.utc).isoformat()
        except ValueError:
            return None

    epoch = _EPOCH_RE.match(ts_raw)
    if epoch:
        value = float(ts_raw)
        if len(epoch.group(1)) == 13:
            value /= 1000.0
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    candidate = ts_raw
    m = re.match(r"^(\d{1,2}/\w{3}/\d{4}):(\d{2}:\d{2}:\d{2}\s*[+-]\d{4})$", ts_raw)
    if m:
        candidate = f"{m.group(1)} {m.group(2)}"

    try:
        dt = dateutil_parser.parse(candidate, default=datetime(reference_year, 1, 1))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError, TypeError):
        return None


def normalize_severity(sev_raw: Optional[str]) -> str:
    if not sev_raw:
        return Severity.UNKNOWN.value
    return config.SEVERITY_ALIASES.get(sev_raw.strip().upper(), Severity.UNKNOWN.value)


def normalize_host(host_raw: Optional[str]) -> Optional[str]:
    return host_raw.strip().strip(",") if host_raw else None


def normalize_service(service_raw: Optional[str]) -> Optional[str]:
    return service_raw.strip().lower().replace("_", "-") if service_raw else None


def strip_embedded_prefix(message: str) -> str:
    return _EMBEDDED_PREFIX_RE.sub("", message)


def extract_entities(message: str) -> dict:
    entities: dict = {}
    ips = _IP_RE.findall(message)
    if ips:
        entities["ips"] = sorted(set(ips))
    status_match = _STATUS_RE.search(message)
    if status_match:
        entities["status_code"] = int(status_match.group(1))
    ms_match = _MS_RE.search(message)
    if ms_match:
        entities["latency_ms"] = int(ms_match.group(1))
    err_match = _ERRCODE_RE.search(message)
    if err_match:
        entities["error_code"] = err_match.group(0).upper()
    return entities


def normalize_event(event: LogEvent) -> LogEvent:
    """Fills the normalized fields. `raw_message`, `message`, `timestamp_raw`
    and `severity_raw` are never modified."""
    event.timestamp_iso = normalize_timestamp(event.timestamp_raw)
    if event.timestamp_raw and event.timestamp_iso is None:
        # a timestamp was present but could not be interpreted (vs. absent)
        event.metadata["timestamp_unparseable"] = True
    if timestamp_year_is_assumed(event.timestamp_raw):
        event.metadata["timestamp_year_assumed"] = True
    if event.severity_raw is None and event.severity not in (None, Severity.UNKNOWN.value):
        event.severity_raw = event.severity
    event.severity = normalize_severity(event.severity)
    event.host = normalize_host(event.host)
    event.service = normalize_service(event.service)
    event.normalized_message = strip_embedded_prefix(event.message)
    event.entities = extract_entities(event.normalized_message)
    return event


def normalize_events(events: list[LogEvent]) -> list[LogEvent]:
    """Normalizes every event. One malformed event must not abort the run:
    if normalization raises, the event is kept, its `normalized_message`
    falls back to the parsed message, and the error is recorded in
    `metadata["normalize_error"]` (counted in the pipeline report)."""
    for e in events:
        try:
            normalize_event(e)
        except Exception as exc:  # noqa: BLE001
            e.metadata["normalize_error"] = f"{type(exc).__name__}: {exc}"
            if e.normalized_message is None:
                e.normalized_message = e.message if isinstance(e.message, str) else str(e.message)
            e.severity = e.severity if e.severity else Severity.UNKNOWN.value
    return events
