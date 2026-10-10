"""Log intelligence (Member 1): raw logs -> structured events -> semantic groups.

Single-file implementation. Pipeline::

    raw log file -> parse -> normalize/template -> LogEvent
                 -> embed unique texts -> cluster -> artifacts (+ optional evaluation)

Sections: SCHEMA (canonical ``LogEvent`` contract), PARSERS, NORMALIZE, INGEST,
EMBEDDINGS, CLUSTERING, EVALUATION, PIPELINE, CLI.

Usage::

    python -m src.log_intelligence run --input app.log --out artifacts/run1 \
        [--labels labels.csv] [--config config.yaml] [--sweep]
    python -m src.log_intelligence schema [--out log_event.schema.json]

Not a root-cause claim: good semantic clustering does not show that the system can
identify root causes. See ``run_pipeline`` for the artifacts written.
"""

from __future__ import annotations

__version__ = "0.1.0"

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Literal, Mapping, Protocol, Sequence

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

# ============================================================================
# SCHEMA
# ============================================================================
SCHEMA_VERSION: Final[int] = 1

SEVERITIES: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARN", "ERROR", "FATAL")

#: Raw level strings seen in common log formats -> canonical severity.
_SEVERITY_ALIASES: Final[dict[str, str]] = {
    "TRACE": "DEBUG",
    "FINE": "DEBUG",
    "FINER": "DEBUG",
    "FINEST": "DEBUG",
    "DEBUG": "DEBUG",
    "DBG": "DEBUG",
    "INFO": "INFO",
    "INFORMATION": "INFO",
    "NOTICE": "INFO",
    "WARN": "WARN",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "ERR": "ERROR",
    "SEVERE": "ERROR",
    "FATAL": "FATAL",
    "CRITICAL": "FATAL",
    "CRIT": "FATAL",
    "ALERT": "FATAL",
    "EMERG": "FATAL",
    "EMERGENCY": "FATAL",
}

ParseStatus = Literal["ok", "partial"]


def normalize_severity(raw: str | None) -> str | None:
    """Map a raw level string onto :data:`SEVERITIES`; ``None`` if unknown."""
    if raw is None:
        return None
    return _SEVERITY_ALIASES.get(raw.strip().upper())


def make_event_id(source_name: str, line_number: int, raw_line: str) -> str:
    """Deterministic, collision-resistant event ID.

    The hash covers the source name, 1-based line number and the raw line text
    (with trailing line terminators removed) so that identical lines at
    different positions get different IDs, and re-ingesting the same file always
    reproduces the same IDs.
    """
    if line_number < 1:
        raise ValueError("line_number must be >= 1")
    payload = f"{source_name}\x1f{line_number}\x1f{raw_line.rstrip(chr(13) + chr(10))}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"evt_{digest}"


def make_template_id(template: str) -> str:
    """Stable ID for a normalized message template."""
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
    return f"tpl_{digest}"


_EVENT_ID_RE = re.compile(r"^evt_[0-9a-f]{16}$")
_TEMPLATE_ID_RE = re.compile(r"^tpl_[0-9a-f]{12}$")


class LogEvent(BaseModel):
    """One structured log event extracted from one raw log record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)

    # ---- fields consumed by src.evidence.EventStore (do not rename) ----
    event_id: str
    timestamp_iso: str | None = None
    severity: str | None = None
    service: str | None = None
    message: str
    template: str | None = None
    source_file: str | None = None
    line_number: int | None = Field(default=None, ge=1, strict=True)

    # ---- additional canonical fields ----
    template_id: str | None = None
    timestamp_tz_assumed: bool = False
    parameters: tuple[str, ...] = Field(default_factory=tuple)
    attributes: dict[str, str] = Field(default_factory=dict)
    raw: str
    parser: str
    parse_status: ParseStatus = "ok"
    parse_warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("event_id")
    @classmethod
    def _check_event_id(cls, value: str) -> str:
        if not _EVENT_ID_RE.match(value):
            raise ValueError("event_id must match evt_<16 hex chars>")
        return value

    @field_validator("template_id")
    @classmethod
    def _check_template_id(cls, value: str | None) -> str | None:
        if value is not None and not _TEMPLATE_ID_RE.match(value):
            raise ValueError("template_id must match tpl_<12 hex chars>")
        return value

    @field_validator("timestamp_iso")
    @classmethod
    def _check_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("timestamp_iso must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp_iso must include a UTC offset")
        return value

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, value: str | None) -> str | None:
        if value is not None and value not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES} or None")
        return value

    @field_validator("service")
    @classmethod
    def _check_service(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("service must be None or a non-blank string")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "LogEvent":
        if self.timestamp_iso is None and self.timestamp_tz_assumed:
            raise ValueError("timestamp_tz_assumed requires a timestamp_iso")
        if self.template is not None and self.template_id is None:
            raise ValueError("template_id is required when template is set")
        if self.parse_status == "partial" and not self.parse_warnings:
            raise ValueError("partial events must carry at least one parse warning")
        return self

    # ------------------------------------------------------------------
    def to_record(self) -> dict[str, Any]:
        """JSON-ready dict with a stable key order (used for JSONL artifacts)."""
        return self.model_dump(mode="json")


class ParseFailure(BaseModel):
    """A raw line that could not be turned into an event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file: str
    line_number: int = Field(ge=1, strict=True)
    raw: str
    reason: str
    parser: str


def utc_isoformat(moment: datetime) -> str:
    """Render ``moment`` as an ISO-8601 string in UTC (``+00:00``)."""
    if moment.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return moment.astimezone(timezone.utc).isoformat()


def export_json_schema() -> dict[str, Any]:
    """JSON Schema for the machine-readable interface contract."""
    schema = LogEvent.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "LogEvent"
    return schema

# ============================================================================
# PARSERS
# ============================================================================
_LEVEL_WORDS = (
    "TRACE|FINEST|FINER|FINE|DEBUG|DBG|INFORMATION|INFO|NOTICE|WARNING|WARN|"
    "ERROR|ERR|SEVERE|FATAL|CRITICAL|CRIT|ALERT|EMERGENCY|EMERG"
)


class ParseError(Exception):
    """A line could not be parsed; ``reason`` is a short stable code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ParsedRecord:
    """Parser output before normalization / ID assignment."""

    message: str
    timestamp_iso: str | None = None
    timestamp_tz_assumed: bool = False
    severity: str | None = None
    service: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class LineParser(Protocol):
    name: str

    def parse(self, line: str) -> ParsedRecord: ...


# ----------------------------------------------------------------------------
# Timestamp helpers
# ----------------------------------------------------------------------------

def parse_iso_like(text: str) -> tuple[str, bool] | None:
    """Parse ISO-8601-ish text. Returns ``(utc_iso, tz_was_assumed)`` or ``None``.

    Accepts ``T`` or space separators, ``Z`` / ``+HH:MM`` / ``+HHMM`` offsets and
    comma decimal fractions (log4j style). Invalid calendar values return
    ``None`` rather than raising.
    """
    candidate = text.strip().replace(",", ".")
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    assumed = parsed.tzinfo is None or parsed.utcoffset() is None
    if assumed:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return utc_isoformat(parsed), assumed


def parse_with_format(text: str, fmt: str) -> tuple[str, bool] | None:
    """Parse a timezone-less timestamp with ``strptime``; UTC is assumed."""
    try:
        parsed = datetime.strptime(text, fmt)
    except ValueError:
        return None
    return utc_isoformat(parsed.replace(tzinfo=timezone.utc)), True


def parse_epoch(value: float) -> tuple[str, bool] | None:
    """Parse epoch seconds or milliseconds (auto-detected by magnitude)."""
    if isinstance(value, bool):
        return None
    seconds = float(value)
    if seconds > 1e11:  # milliseconds
        seconds /= 1000.0
    try:
        moment = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    except (OverflowError, ValueError):
        return None
    return utc_isoformat(moment), False


def _finalize(
    *,
    message: str,
    ts: tuple[str, bool] | None,
    ts_present: bool,
    level_raw: str | None,
    service: str | None,
    attributes: Mapping[str, str] | None = None,
    extra_warnings: Sequence[str] = (),
) -> ParsedRecord:
    warnings = list(extra_warnings)
    if ts is None:
        warnings.append("invalid_timestamp" if ts_present else "missing_timestamp")
    severity = normalize_severity(level_raw)
    if severity is None:
        warnings.append("unknown_severity" if level_raw else "missing_severity")
    if not service:
        service = None
        warnings.append("missing_service")
    if not message.strip():
        warnings.append("empty_message")
    return ParsedRecord(
        message=message,
        timestamp_iso=ts[0] if ts else None,
        timestamp_tz_assumed=bool(ts and ts[1]),
        severity=severity,
        service=service,
        attributes=dict(sorted((attributes or {}).items())),
        warnings=tuple(warnings),
    )


# ----------------------------------------------------------------------------
# Regex-based parsers
# ----------------------------------------------------------------------------

class _RegexParser:
    name = "regex"
    _pattern: re.Pattern[str]

    def _match(self, line: str) -> re.Match[str]:
        match = self._pattern.match(line.strip())
        if match is None:
            raise ParseError("no_match")
        return match


class ProjectFormatParser(_RegexParser):
    """``YYYY-MM-DD HH:MM:SS [LEVEL] service: message`` (existing project format)."""

    name = "project"
    _pattern = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
        r"\[(?P<level>\w+)\]\s+(?P<service>[\w.$-]+):\s*(?P<message>.*)$"
    )

    def parse(self, line: str) -> ParsedRecord:
        m = self._match(line)
        return _finalize(
            message=m["message"],
            ts=parse_iso_like(m["ts"]),
            ts_present=True,
            level_raw=m["level"],
            service=m["service"],
        )


class GenericIsoParser(_RegexParser):
    """``<ISO timestamp> LEVEL [service[:| -]] message`` (log4j / logback / app logs)."""

    name = "generic_iso"
    _pattern = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
        r"(?:Z|[+-]\d{2}:?\d{2})?)\s+"
        rf"\[?(?P<level>{_LEVEL_WORDS})\]?\s+"
        r"(?:\[?(?P<service>[\w.$/-]+)\]?\s*[:\-]\s+)?(?P<message>.*)$",
        re.IGNORECASE,
    )

    def parse(self, line: str) -> ParsedRecord:
        m = self._match(line)
        return _finalize(
            message=m["message"],
            ts=parse_iso_like(m["ts"]),
            ts_present=True,
            level_raw=m["level"],
            service=m["service"],
        )


class HdfsParser(_RegexParser):
    """Loghub HDFS: ``yymmdd HHMMSS pid LEVEL component: message``."""

    name = "hdfs"
    _pattern = re.compile(
        r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>[A-Za-z]+)\s+"
        r"(?P<service>[\w.$-]+):\s*(?P<message>.*)$"
    )

    def parse(self, line: str) -> ParsedRecord:
        m = self._match(line)
        ts = parse_with_format(f"{m['date']} {m['time']}", "%y%m%d %H%M%S")
        return _finalize(
            message=m["message"],
            ts=ts,
            ts_present=True,
            level_raw=m["level"],
            service=m["service"],
            attributes={"pid": m["pid"]},
        )


class SyslogParser(_RegexParser):
    """BSD syslog: ``Mon DD HH:MM:SS host service[pid]: message`` (no year in source)."""

    name = "syslog"
    _pattern = re.compile(
        r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+(?P<service>[\w./-]+?)(?:\[(?P<pid>\d+)\])?:\s*(?P<message>.*)$"
    )

    def __init__(self, default_year: int = 1970):
        self.default_year = default_year

    def parse(self, line: str) -> ParsedRecord:
        m = self._match(line)
        stamp = f"{self.default_year} {m['mon']} {m['day']} {m['time']}"
        ts = parse_with_format(stamp, "%Y %b %d %H:%M:%S")
        attrs = {"host": m["host"]}
        if m["pid"]:
            attrs["pid"] = m["pid"]
        return _finalize(
            message=m["message"],
            ts=ts,
            ts_present=True,
            level_raw=None,
            service=m["service"],
            attributes=attrs,
            extra_warnings=("year_assumed",),
        )


# ----------------------------------------------------------------------------
# JSON lines
# ----------------------------------------------------------------------------

_TS_KEYS = ("timestamp", "@timestamp", "time", "ts", "datetime")
_LEVEL_KEYS = ("level", "severity", "lvl", "loglevel", "log_level")
_SERVICE_KEYS = ("service", "component", "logger", "app", "name")
_MESSAGE_KEYS = ("message", "msg", "log", "text")


def _first_key(obj: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in obj:
            return key
    return None


class JsonLinesParser:
    """One JSON object per line with common field-name aliases."""

    name = "jsonl"

    def parse(self, line: str) -> ParsedRecord:
        text = line.strip()
        if not text.startswith("{"):
            raise ParseError("not_json_object")
        try:
            obj = json.loads(text)
        except ValueError as exc:
            raise ParseError("invalid_json") from exc
        if not isinstance(obj, dict):
            raise ParseError("not_json_object")

        msg_key = _first_key(obj, _MESSAGE_KEYS)
        if msg_key is None or not isinstance(obj[msg_key], str):
            raise ParseError("missing_message")

        ts_key = _first_key(obj, _TS_KEYS)
        ts: tuple[str, bool] | None = None
        if ts_key is not None:
            raw_ts = obj[ts_key]
            if isinstance(raw_ts, str):
                ts = parse_iso_like(raw_ts)
            elif isinstance(raw_ts, (int, float)) and not isinstance(raw_ts, bool):
                ts = parse_epoch(raw_ts)

        level_key = _first_key(obj, _LEVEL_KEYS)
        level_raw = obj[level_key] if level_key and isinstance(obj[level_key], str) else None
        service_key = _first_key(obj, _SERVICE_KEYS)
        service = obj[service_key] if service_key and isinstance(obj[service_key], str) else None

        consumed = {k for k in (msg_key, ts_key, level_key, service_key) if k}
        attributes = {
            str(k): str(v)
            for k, v in obj.items()
            if k not in consumed and isinstance(v, (str, int, float, bool))
        }
        return _finalize(
            message=obj[msg_key],
            ts=ts,
            ts_present=ts_key is not None,
            level_raw=level_raw,
            service=service,
            attributes=attributes,
        )


# ----------------------------------------------------------------------------
# Registry / auto-detection
# ----------------------------------------------------------------------------

def build_parsers(default_year: int = 1970) -> dict[str, LineParser]:
    """Return all parsers keyed by name, in auto-detection priority order."""
    parsers: list[LineParser] = [
        ProjectFormatParser(),
        HdfsParser(),
        JsonLinesParser(),
        GenericIsoParser(),
        SyslogParser(default_year=default_year),
    ]
    return {p.name: p for p in parsers}


def detect_parser(
    lines: Sequence[str],
    parsers: Mapping[str, LineParser],
    *,
    sample_size: int = 200,
    min_success_rate: float = 0.6,
) -> str:
    """Pick the parser that succeeds on the largest share of non-blank sample lines.

    Ties are broken by registry order, which keeps detection deterministic.
    Raises :class:`ParseError` if no parser reaches ``min_success_rate``.
    """
    sample = [ln for ln in lines if ln.strip()][:sample_size]
    if not sample:
        raise ParseError("empty_input")
    best_name, best_rate = "", -1.0
    for name, parser in parsers.items():
        ok = 0
        for line in sample:
            try:
                parser.parse(line)
                ok += 1
            except ParseError:
                pass
        rate = ok / len(sample)
        if rate > best_rate:
            best_name, best_rate = name, rate
    if best_rate < min_success_rate:
        raise ParseError(f"no_parser_matched(best_rate={best_rate:.2f})")
    return best_name

# ============================================================================
# NORMALIZE
# ============================================================================
PLACEHOLDER = "<*>"

_WS = re.compile(r"\s+")
# "blk_<*> blk_<*> blk_<*>" (variable-length lists) -> "blk_<*>" so list length does
# not create a new template per event.
_REPEATED = re.compile(r"(\S*<\*>)(?:\s+\1)+")

# (name, regex, replacement builder). Alternatives earlier in the list win ties
# at the same start position.
_RULES: list[tuple[str, str, Callable[[re.Match[str]], str]]] = [
    (
        "uuid",
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        lambda m: PLACEHOLDER,
    ),
    (
        "path",
        r"(?<![\w<>*.])/(?:[\w.@=+-]+/)+[\w.@=+-]*",
        lambda m: PLACEHOLDER,
    ),
    (
        "ipv4",
        r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?(?!\w)(?!\.\d)",
        lambda m: PLACEHOLDER,
    ),
    ("block", r"\bblk_-?\d+", lambda m: "blk_" + PLACEHOLDER),
    ("hex0x", r"\b0[xX][0-9a-fA-F]+\b", lambda m: PLACEHOLDER),
    (
        "longhex",
        r"\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}\b",
        lambda m: PLACEHOLDER,
    ),
    (
        "prefixed_id",
        r"\b(?P<pfx>[A-Za-z][A-Za-z]*[_-])\d+(?:[_-]\d+)*\b",
        lambda m: m.group("pfx") + PLACEHOLDER,
    ),
    (
        "number",
        r"(?<![A-Za-z_\d.<*-])-?\d+(?:\.\d+)?(?=(?i:ms|us|ns|s|m|h|d|kb|mb|gb|tb|b)?(?![A-Za-z_\d]))",
        lambda m: PLACEHOLDER,
    ),
]

_COMBINED = re.compile(
    "|".join(f"(?P<r{i}>{pattern})" for i, (_, pattern, _) in enumerate(_RULES))
)


def collapse_whitespace(text: str) -> str:
    """Trim and collapse all whitespace runs (including newlines) to one space."""
    return _WS.sub(" ", text).strip()


def extract_template(message: str) -> tuple[str, tuple[str, ...]]:
    """Return ``(template, parameters)`` for ``message``.

    ``parameters`` are the masked substrings, in order of appearance, so the
    original (whitespace-collapsed) message can be reconstructed from the
    template. An empty message yields an empty template.
    """
    text = collapse_whitespace(message)
    params: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        for i, (_, _, builder) in enumerate(_RULES):
            if match.group(f"r{i}") is not None:
                params.append(match.group(0))
                return builder(match)
        raise AssertionError("unreachable: combined pattern matched no rule")

    template = _REPEATED.sub(r"\1", _COMBINED.sub(_sub, text))
    return template, tuple(params)

# ============================================================================
# INGEST
# ============================================================================
_CONTINUATION = re.compile(r"^(?:\s+\S|Caused by:|\.\.\. \d+ more|Traceback|at\s)")

MAX_LINE_LENGTH = 1_000_000


class IngestConfig(BaseModel):
    """Ingestion settings (part of the reproducible pipeline configuration)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parser: str = "auto"
    default_year: int = Field(default=1970, ge=1970, le=2100)
    detect_sample_size: int = Field(default=200, ge=1)
    detect_min_success_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    attach_continuations: bool = True
    encoding: str = "utf-8"
    encoding_errors: str = "replace"


@dataclass
class IngestResult:
    source_name: str
    parser: str
    events: list[LogEvent] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    continuation_lines: int = 0

    def accounting_ok(self) -> bool:
        return self.total_lines == (
            self.blank_lines + len(self.failures) + len(self.events) + self.continuation_lines
        )


@dataclass
class _Pending:
    line_number: int
    raw: str
    record: ParsedRecord
    continuations: list[str] = field(default_factory=list)


def read_lines(path: str | Path, config: IngestConfig = IngestConfig()) -> list[str]:
    """Read physical lines with universal-newline handling (``\\r\\n`` safe)."""
    with open(path, encoding=config.encoding, errors=config.encoding_errors, newline=None) as fh:
        return [line.rstrip("\n") for line in fh]


def _build_event(source_name: str, parser_name: str, pending: _Pending) -> LogEvent:
    rec = pending.record
    first_message = rec.message
    message = first_message
    if pending.continuations:
        message = "\n".join([first_message, *(c.strip() for c in pending.continuations)])
    template, params = extract_template(first_message)
    raw = "\n".join([pending.raw, *pending.continuations])

    warnings = list(rec.warnings)
    if "\ufffd" in raw:
        warnings.append("encoding_replaced")
    attributes = dict(rec.attributes)
    if pending.continuations:
        attributes["continuation_lines"] = str(len(pending.continuations))

    return LogEvent(
        event_id=make_event_id(source_name, pending.line_number, pending.raw),
        timestamp_iso=rec.timestamp_iso,
        timestamp_tz_assumed=rec.timestamp_tz_assumed,
        severity=rec.severity,
        service=rec.service,
        message=message,
        template=template,
        template_id=make_template_id(template),
        source_file=source_name,
        line_number=pending.line_number,
        parameters=params,
        attributes=dict(sorted(attributes.items())),
        raw=raw,
        parser=parser_name,
        parse_status="partial" if warnings else "ok",
        parse_warnings=tuple(warnings),
    )


def ingest_lines(
    lines: Iterable[str],
    source_name: str,
    config: IngestConfig = IngestConfig(),
    parsers: dict[str, LineParser] | None = None,
) -> IngestResult:
    """Parse already-split physical lines (no trailing newline) into events."""
    all_lines = list(lines)
    registry = parsers or build_parsers(default_year=config.default_year)

    if config.parser == "auto":
        try:
            parser_name = detect_parser(
                all_lines,
                registry,
                sample_size=config.detect_sample_size,
                min_success_rate=config.detect_min_success_rate,
            )
        except ParseError as exc:
            result = IngestResult(source_name=source_name, parser="none")
            for number, line in enumerate(all_lines, start=1):
                result.total_lines += 1
                if not line.strip():
                    result.blank_lines += 1
                else:
                    result.failures.append(
                        ParseFailure(
                            source_file=source_name,
                            line_number=number,
                            raw=line,
                            reason=exc.reason,
                            parser="none",
                        )
                    )
            return result
    else:
        if config.parser not in registry:
            raise ValueError(f"Unknown parser {config.parser!r}; choose from {sorted(registry)}")
        parser_name = config.parser
    parser = registry[parser_name]

    result = IngestResult(source_name=source_name, parser=parser_name)
    pendings: list[_Pending] = []

    for number, line in enumerate(all_lines, start=1):
        result.total_lines += 1
        if not line.strip():
            result.blank_lines += 1
            continue
        if len(line) > MAX_LINE_LENGTH:
            result.failures.append(
                ParseFailure(
                    source_file=source_name,
                    line_number=number,
                    raw=line[:200] + "...[truncated]",
                    reason="line_too_long",
                    parser=parser_name,
                )
            )
            continue
        try:
            record = parser.parse(line)
        except ParseError as exc:
            if config.attach_continuations and pendings and _CONTINUATION.match(line):
                pendings[-1].continuations.append(line)
                result.continuation_lines += 1
            else:
                result.failures.append(
                    ParseFailure(
                        source_file=source_name,
                        line_number=number,
                        raw=line,
                        reason=exc.reason,
                        parser=parser_name,
                    )
                )
            continue
        pendings.append(_Pending(line_number=number, raw=line, record=record))

    result.events = [_build_event(source_name, parser_name, p) for p in pendings]
    return result


def ingest_file(
    path: str | Path,
    config: IngestConfig = IngestConfig(),
    source_name: str | None = None,
) -> IngestResult:
    """Ingest a log file. ``source_name`` defaults to the file's basename so event
    IDs do not depend on where the file lives on disk."""
    p = Path(path)
    return ingest_lines(read_lines(p, config), source_name or p.name, config)


def unique_templates(events: Sequence[LogEvent]) -> list[tuple[str, str, int]]:
    """``(template_id, template, count)`` in order of first appearance."""
    counts: dict[str, list] = {}
    for ev in events:
        if ev.template is None or ev.template_id is None:
            continue
        entry = counts.setdefault(ev.template_id, [ev.template, 0])
        entry[1] += 1
    return [(tid, tpl, n) for tid, (tpl, n) in counts.items()]

# ============================================================================
# EMBEDDINGS
# ============================================================================
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


class EmbeddingError(ValueError):
    """Raised when texts cannot be embedded (e.g. nothing but placeholders)."""


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["tfidf_svd", "sentence_transformer"] = "tfidf_svd"
    # tfidf_svd
    dim: int = Field(default=32, ge=2)
    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=2, ge=1)
    min_df: int = Field(default=1, ge=1)
    # sentence_transformer
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str | None = None
    batch_size: int = Field(default=64, ge=1)
    device: str = "cpu"
    # shared
    seed: int = 0


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, d)`` float array with L2-normalised rows."""
        ...

    def describe(self) -> dict[str, Any]: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # zero vectors stay zero instead of becoming NaN
    return matrix / norms


def tokenize_for_embedding(text: str) -> str:
    """Lower-case, split camelCase, drop placeholders so only wording remains."""
    text = text.replace("<*>", " ")
    text = _CAMEL.sub(" ", text)
    return text.lower()


class TfidfSvdEmbedder:
    name = "tfidf_svd"

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if len(texts) == 0:
            raise EmbeddingError("no texts to embed")
        vectorizer = TfidfVectorizer(
            preprocessor=tokenize_for_embedding,
            token_pattern=r"[a-z]{2,}",
            ngram_range=(self.cfg.ngram_min, self.cfg.ngram_max),
            min_df=self.cfg.min_df,
            sublinear_tf=True,
            dtype=np.float64,
        )
        try:
            tfidf = vectorizer.fit_transform(list(texts))
        except ValueError as exc:  # "empty vocabulary"
            raise EmbeddingError(f"cannot build vocabulary: {exc}") from exc

        n_samples, n_features = tfidf.shape
        max_components = min(self.cfg.dim, n_features - 1, n_samples - 1)
        if max_components >= 2:
            svd = TruncatedSVD(
                n_components=max_components, algorithm="arpack", random_state=self.cfg.seed
            )
            dense = svd.fit_transform(tfidf)
            # SVD component signs are arbitrary; fix them so output is stable.
            for j in range(dense.shape[1]):
                col = dense[:, j]
                if col[np.argmax(np.abs(col))] < 0:
                    dense[:, j] = -col
        else:
            dense = tfidf.toarray()
        return _l2_normalize(np.asarray(dense, dtype=np.float64))

    def describe(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "backend": self.name,
            "dim": c.dim,
            "ngram_range": [c.ngram_min, c.ngram_max],
            "min_df": c.min_df,
            "seed": c.seed,
        }


class SentenceTransformerEmbedder:
    name = "sentence_transformer"

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed; use backend 'tfidf_svd' "
                "or `pip install sentence-transformers`"
            ) from exc
        if len(texts) == 0:
            raise EmbeddingError("no texts to embed")
        model = SentenceTransformer(
            self.cfg.model_name, revision=self.cfg.model_revision, device=self.cfg.device
        )
        vectors = model.encode(
            [t.replace("<*>", " ") for t in texts],
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float64))

    def describe(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "backend": self.name,
            "model_name": c.model_name,
            "model_revision": c.model_revision,
            "batch_size": c.batch_size,
            "device": c.device,
        }


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.backend == "tfidf_svd":
        return TfidfSvdEmbedder(cfg)
    if cfg.backend == "sentence_transformer":
        return SentenceTransformerEmbedder(cfg)
    raise ValueError(f"Unknown embedding backend: {cfg.backend}")

# ============================================================================
# CLUSTERING
# ============================================================================
NOISE_CLUSTER_ID = "noise"


class ClusteringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["agglomerative", "kmeans", "dbscan"] = "agglomerative"
    # agglomerative: cosine distance cut-off (0 = identical direction, 2 = opposite)
    distance_threshold: float = Field(default=0.5, gt=0.0, le=2.0)
    linkage: Literal["average", "complete", "single"] = "average"
    # kmeans
    n_clusters: int | None = Field(default=None, ge=1)
    # dbscan (cosine distance)
    eps: float = Field(default=0.3, gt=0.0, le=2.0)
    min_samples: int = Field(default=2, ge=1)
    seed: int = 0

    @model_validator(mode="after")
    def _kmeans_needs_k(self) -> "ClusteringConfig":
        if self.algorithm == "kmeans" and self.n_clusters is None:
            raise ValueError("kmeans requires n_clusters")
        return self


@dataclass(frozen=True)
class ClusterAssignment:
    """Canonical cluster id per unit; ``NOISE_CLUSTER_ID`` marks DBSCAN noise."""

    cluster_ids: tuple[str, ...]

    @property
    def n_clusters(self) -> int:
        return len({c for c in self.cluster_ids if c != NOISE_CLUSTER_ID})


def _canonicalize(raw_labels: np.ndarray, weights: Sequence[int]) -> tuple[str, ...]:
    """Map arbitrary integer labels to ``cluster_000``... ordered by (-weight, first index)."""
    stats: dict[int, list[int]] = {}
    for idx, label in enumerate(raw_labels):
        label = int(label)
        if label < 0:
            continue
        entry = stats.setdefault(label, [0, idx])
        entry[0] += int(weights[idx])
    order = sorted(stats, key=lambda lab: (-stats[lab][0], stats[lab][1]))
    names = {lab: f"cluster_{rank:03d}" for rank, lab in enumerate(order)}
    return tuple(NOISE_CLUSTER_ID if int(l) < 0 else names[int(l)] for l in raw_labels)


def cluster_embeddings(
    embeddings: np.ndarray,
    weights: Sequence[int],
    cfg: ClusteringConfig,
) -> ClusterAssignment:
    """Cluster L2-normalised ``embeddings`` (one row per unit)."""
    n = embeddings.shape[0]
    if n == 0:
        return ClusterAssignment(cluster_ids=())
    if n == 1:
        return ClusterAssignment(cluster_ids=("cluster_000",))

    if cfg.algorithm == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cfg.distance_threshold,
            metric="cosine",
            linkage=cfg.linkage,
        )
        raw = model.fit_predict(embeddings)
    elif cfg.algorithm == "kmeans":
        from sklearn.cluster import KMeans

        k = min(cfg.n_clusters or 1, n)
        model = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed)
        raw = model.fit_predict(embeddings, sample_weight=np.asarray(weights, dtype=float))
    elif cfg.algorithm == "dbscan":
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, metric="cosine")
        raw = model.fit_predict(embeddings, sample_weight=np.asarray(weights, dtype=float))
    else:  # pragma: no cover - guarded by the config Literal
        raise ValueError(f"Unknown algorithm {cfg.algorithm}")
    return ClusterAssignment(cluster_ids=_canonicalize(np.asarray(raw), weights))


def nearest_to_centroid(
    embeddings: np.ndarray, weights: Sequence[int], members: Sequence[int]
) -> int:
    """Index (into ``embeddings``) of the member closest to the weighted centroid."""
    w = np.asarray([weights[i] for i in members], dtype=float)
    pts = embeddings[list(members)]
    centroid = (pts * w[:, None]).sum(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = pts @ centroid
    best = int(np.argmax(sims))  # argmax returns the first max -> deterministic tie-break
    return members[best]

# ============================================================================
# EVALUATION
# ============================================================================
CAVEATS = (
    "Labels describe log statements/templates, not incident root causes.",
    "Clustering quality does not demonstrate that the system can identify root causes.",
    "Template-identity baseline shows how much of the score comes from template extraction alone.",
    "Ground-truth labels were never used to fit embeddings or clusters.",
)


def load_labels_csv(
    path: str | Path, line_column: str = "line_number", label_column: str = "event_id"
) -> dict[int, str]:
    """Load ``line_number -> label``. Raises on duplicate or malformed rows."""
    labels: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {line_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"labels file is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                line = int(row[line_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bad line number {row[line_column]!r}") from exc
            if line in labels:
                raise ValueError(f"duplicate label for line {line}")
            labels[line] = row[label_column]
    return labels


def purity(clusters: Sequence[str], labels: Sequence[str]) -> float:
    by_cluster: dict[str, Counter] = defaultdict(Counter)
    for c, l in zip(clusters, labels):
        by_cluster[c][l] += 1
    return sum(max(cnt.values()) for cnt in by_cluster.values()) / len(labels) if labels else 0.0


def _singleton_noise(clusters: Sequence[str]) -> list[str]:
    """Score each noise point as its own cluster (the conservative choice)."""
    return [f"noise_{i}" if c == NOISE_CLUSTER_ID else c for i, c in enumerate(clusters)]


def score_partition(clusters: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    if len(clusters) != len(labels):
        raise ValueError("clusters and labels must have equal length")
    if not labels:
        return {"n_events": 0}
    cl = _singleton_noise(clusters)
    hom, comp, v = homogeneity_completeness_v_measure(labels, cl)
    return {
        "n_events": len(labels),
        "n_clusters": len(set(cl)),
        "n_labels": len(set(labels)),
        "adjusted_rand_index": round(float(adjusted_rand_score(labels, cl)), 6),
        "normalized_mutual_info": round(float(normalized_mutual_info_score(labels, cl)), 6),
        "homogeneity": round(float(hom), 6),
        "completeness": round(float(comp), 6),
        "v_measure": round(float(v), 6),
        "purity": round(float(purity(cl, labels)), 6),
    }


def evaluate_events(
    event_line_numbers: Sequence[int | None],
    cluster_ids: Sequence[str],
    template_ids: Sequence[str | None],
    labels_by_line: Mapping[int, str],
    seed: int = 0,
) -> dict[str, Any]:
    """Score clusters on events that have a ground-truth label, with baselines."""
    idx = [
        i for i, ln in enumerate(event_line_numbers) if ln is not None and ln in labels_by_line
    ]
    labels = [labels_by_line[event_line_numbers[i]] for i in idx]  # type: ignore[index]
    clusters = [cluster_ids[i] for i in idx]
    templates = [template_ids[i] or f"none_{i}" for i in idx]

    rng = random.Random(seed)
    shuffled = clusters[:]
    rng.shuffle(shuffled)

    per_cluster: dict[str, Counter] = defaultdict(Counter)
    for c, l in zip(clusters, labels):
        per_cluster[c][l] += 1

    return {
        "coverage": {
            "n_events": len(event_line_numbers),
            "n_labelled": len(idx),
            "fraction_labelled": round(len(idx) / len(event_line_numbers), 6)
            if event_line_numbers
            else 0.0,
        },
        "clustering": score_partition(clusters, labels),
        "baselines": {
            "template_identity": score_partition(templates, labels),
            "single_cluster": score_partition(["all"] * len(labels), labels),
            "shuffled_assignments": score_partition(shuffled, labels),
        },
        "cluster_label_distribution": {
            c: dict(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
            for c, cnt in sorted(per_cluster.items())
        },
        "caveats": list(CAVEATS),
    }

# ============================================================================
# PIPELINE
# ============================================================================
class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    #: Which event text is embedded: the masked ``template`` or the full ``message``.
    text_field: Literal["template", "message"] = "template"
    example_events_per_cluster: int = Field(default=3, ge=0)


def canonical_json(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False) + "\n"


def config_hash(config: PipelineConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_for(event: LogEvent, field: str) -> str:
    if field == "template" and event.template:
        return event.template
    return " ".join(event.message.split())


def cluster_events(
    events: list[LogEvent], config: PipelineConfig
) -> tuple[list[str], dict[str, Any], np.ndarray, list[str]]:
    """Return per-event cluster ids, the clusters document, unit embeddings, unit texts."""
    unit_index: dict[str, int] = {}
    unit_texts: list[str] = []
    unit_events: list[list[int]] = []
    event_unit: list[int] = []
    for i, ev in enumerate(events):
        text = _text_for(ev, config.text_field)
        if text not in unit_index:
            unit_index[text] = len(unit_texts)
            unit_texts.append(text)
            unit_events.append([])
        unit_events[unit_index[text]].append(i)
        event_unit.append(unit_index[text])
    weights = [len(m) for m in unit_events]

    embedder = make_embedder(config.embedding)
    skipped: str | None = None
    if not events:
        embeddings = np.zeros((0, 0))
        assignment_ids: tuple[str, ...] = ()
        skipped = "no_events"
    else:
        try:
            embeddings = embedder.embed(unit_texts)
            assignment_ids = cluster_embeddings(embeddings, weights, config.clustering).cluster_ids
        except EmbeddingError as exc:
            skipped = f"embedding_failed: {exc}"
            embeddings = np.zeros((len(unit_texts), 0))
            assignment_ids = tuple("cluster_000" for _ in unit_texts)

    event_clusters = [assignment_ids[u] for u in event_unit]

    members: dict[str, list[int]] = {}
    for u, cid in enumerate(assignment_ids):
        members.setdefault(cid, []).append(u)

    summaries = []
    for cid in sorted(members, key=lambda c: (c == NOISE_CLUSTER_ID, c)):
        units = members[cid]
        evs = [events[e] for u in units for e in unit_events[u]]
        stamps = sorted(e.timestamp_iso for e in evs if e.timestamp_iso)
        if embeddings.shape[1] > 0 and cid != NOISE_CLUSTER_ID:
            rep = nearest_to_centroid(embeddings, weights, units)
        else:
            rep = units[0]
        summaries.append(
            {
                "cluster_id": cid,
                "n_events": len(evs),
                "n_unique_texts": len(units),
                "representative_text": unit_texts[rep],
                "template_ids": sorted({e.template_id for e in evs if e.template_id}),
                "services": dict(sorted(Counter(e.service or "unknown" for e in evs).items())),
                "severities": dict(sorted(Counter(e.severity or "unknown" for e in evs).items())),
                "first_timestamp": stamps[0] if stamps else None,
                "last_timestamp": stamps[-1] if stamps else None,
                "example_event_ids": [
                    e.event_id for e in evs[: config.example_events_per_cluster]
                ],
            }
        )

    doc = {
        "schema_version": SCHEMA_VERSION,
        "text_field": config.text_field,
        "embedder": embedder.describe(),
        "clustering": config.clustering.model_dump(mode="json"),
        "n_events": len(events),
        "n_unique_texts": len(unit_texts),
        "n_clusters": len([c for c in members if c != NOISE_CLUSTER_ID]),
        "n_noise_events": sum(1 for c in event_clusters if c == NOISE_CLUSTER_ID),
        "skipped": skipped,
        "clusters": summaries,
        "unit_assignments": [
            {"text": t, "n_events": weights[i], "cluster_id": assignment_ids[i]}
            for i, t in enumerate(unit_texts)
        ],
    }
    return event_clusters, doc, embeddings, unit_texts


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig = PipelineConfig(),
    *,
    labels_path: str | Path | None = None,
    label_line_column: str = "line_number",
    label_column: str = "event_id",
    source_name: str | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline and write artifacts; returns the run manifest."""
    in_path, out = Path(input_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result: IngestResult = ingest_file(in_path, config.ingest, source_name=source_name)
    events = result.events
    event_clusters, clusters_doc, embeddings, _ = cluster_events(events, config)

    _write(out / "events.jsonl", "".join(canonical_json(e.to_record(), indent=None) for e in events))
    _write(
        out / "parse_failures.jsonl",
        "".join(canonical_json(f.model_dump(mode="json"), indent=None) for f in result.failures),
    )

    tpl_counts: dict[str, dict[str, Any]] = {}
    for ev, cid in zip(events, event_clusters):
        if ev.template_id is None:
            continue
        entry = tpl_counts.setdefault(
            ev.template_id, {"template_id": ev.template_id, "template": ev.template, "n_events": 0}
        )
        entry["n_events"] += 1
        entry.setdefault("cluster_ids", set()).add(cid)
    templates = [
        {**t, "cluster_ids": sorted(t["cluster_ids"])} for t in tpl_counts.values()
    ]
    _write(out / "templates.json", canonical_json({"schema_version": SCHEMA_VERSION, "templates": templates}))

    np.save(out / "template_embeddings.npy", np.round(embeddings, 6))
    _write(out / "clusters.json", canonical_json(clusters_doc))
    _write(
        out / "event_clusters.jsonl",
        "".join(
            canonical_json(
                {"event_id": e.event_id, "template_id": e.template_id, "cluster_id": c}, indent=None
            )
            for e, c in zip(events, event_clusters)
        ),
    )

    artifacts = ["events.jsonl", "parse_failures.jsonl", "templates.json",
                 "template_embeddings.npy", "clusters.json", "event_clusters.jsonl"]

    evaluation = None
    if labels_path is not None:
        labels = load_labels_csv(labels_path, label_line_column, label_column)
        evaluation = evaluate_events(
            [e.line_number for e in events],
            event_clusters,
            [e.template_id for e in events],
            labels,
            seed=config.clustering.seed,
        )
        _write(out / "evaluation.json", canonical_json(evaluation))
        artifacts.append("evaluation.json")

    import sklearn
    manifest = {
        "pipeline_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "config": config.model_dump(mode="json"),
        "config_sha256": config_hash(config),
        "dataset": {
            "source_name": result.source_name,
            "sha256": sha256_file(in_path),
            "labels_sha256": sha256_file(Path(labels_path)) if labels_path else None,
        },
        "ingest": {
            "parser": result.parser,
            "total_lines": result.total_lines,
            "blank_lines": result.blank_lines,
            "continuation_lines": result.continuation_lines,
            "n_events": len(events),
            "n_partial_events": sum(1 for e in events if e.parse_status == "partial"),
            "n_failures": len(result.failures),
            "failure_reasons": dict(sorted(Counter(f.reason for f in result.failures).items())),
            "accounting_ok": result.accounting_ok(),
        },
        "n_unique_templates": len(templates),
        "n_clusters": clusters_doc["n_clusters"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {name: sha256_file(out / name) for name in sorted(artifacts)},
    }
    _write(out / "run_manifest.json", canonical_json(manifest))
    return manifest


def run_sweep(
    input_path: str | Path,
    labels_path: str | Path,
    base: PipelineConfig,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    text_fields: tuple[str, ...] = ("template", "message"),
    *,
    label_line_column: str = "line_number",
    label_column: str = "event_id",
    source_name: str | None = None,
) -> dict[str, Any]:
    """Sensitivity analysis over distance threshold and embedded text field.

    Reported in full so that the default configuration is not presented as the
    best of many tries. ``message`` mode embeds the raw message (variables
    included), a harder setting that does not benefit from template masking.
    """
    result = ingest_file(input_path, base.ingest, source_name=source_name)
    labels = load_labels_csv(labels_path, label_line_column, label_column)
    rows = []
    for text_field in text_fields:
        for th in thresholds:
            cfg = base.model_copy(
                update={
                    "text_field": text_field,
                    "clustering": base.clustering.model_copy(
                        update={"algorithm": "agglomerative", "distance_threshold": th}
                    ),
                }
            )
            clusters, doc, _, _ = cluster_events(result.events, cfg)
            ev = evaluate_events(
                [e.line_number for e in result.events],
                clusters,
                [e.template_id for e in result.events],
                labels,
                seed=cfg.clustering.seed,
            )
            rows.append(
                {"text_field": text_field, "distance_threshold": th, **ev["clustering"]}
            )
    return {"schema_version": SCHEMA_VERSION, "rows": rows}

# ============================================================================
# CLI
# ============================================================================
def load_config(path: str | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return PipelineConfig.model_validate(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="log_intelligence", description="Log intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="raw log -> events -> clusters (+ evaluation)")
    run.add_argument("--input", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--config")
    run.add_argument("--labels", help="CSV with ground-truth labels (optional)")
    run.add_argument("--label-line-column", default="line_number")
    run.add_argument("--label-column", default="event_id")
    run.add_argument("--sweep", action="store_true", help="also write sweep.json (needs --labels)")

    schema = sub.add_parser("schema", help="print the LogEvent JSON Schema")
    schema.add_argument("--out")

    args = parser.parse_args(argv)

    if args.command == "schema":
        text = canonical_json(export_json_schema())
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0

    config = load_config(args.config)
    manifest = run_pipeline(
        args.input,
        args.out,
        config,
        labels_path=args.labels,
        label_line_column=args.label_line_column,
        label_column=args.label_column,
    )
    if args.sweep:
        if not args.labels:
            parser.error("--sweep requires --labels")
        sweep = run_sweep(
            args.input, args.labels, config,
            label_line_column=args.label_line_column, label_column=args.label_column,
        )
        Path(args.out, "sweep.json").write_text(canonical_json(sweep), encoding="utf-8")

    ing = manifest["ingest"]
    print(
        f"parser={ing['parser']} lines={ing['total_lines']} events={ing['n_events']} "
        f"failures={ing['n_failures']} templates={manifest['n_unique_templates']} "
        f"clusters={manifest['n_clusters']}"
    )
    eval_path = Path(args.out, "evaluation.json")
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        c, t = ev["clustering"], ev["baselines"]["template_identity"]
        print(
            f"ARI={c['adjusted_rand_index']} NMI={c['normalized_mutual_info']} "
            f"homog={c['homogeneity']} compl={c['completeness']} purity={c['purity']} "
            f"(template-identity ARI={t['adjusted_rand_index']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
