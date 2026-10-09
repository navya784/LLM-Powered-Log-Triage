"""
Canonical log schema (Phase 1: "Define common log schema").

Every raw log line -- regardless of which parser handled it -- is converted
into a `LogEvent`. This is the shared contract that Phase 2 (event
extraction, embeddings, clustering) and later phases (temporal graph,
evidence builder, LLM integration) read from. See docs/interface_contract.md.

Field groups
  traceability : event_id, dataset_id, source_file, line_number, raw_message
  content      : message (as parsed), normalized_message (after normalization)
  origin       : timestamp_*, host, service, process, pid
  correlation  : request_id, trace_id, exception
  parsing      : parser_name, parse_confidence, log_format, metadata
  derived      : severity, event_type, template, entities, semantic_cluster,
                 embedding_index

Rule: a field is only filled when the log line (or explicit dataset
configuration) actually provides it. `None` means "not available", never
"guessed". `service`, `process` and `pid` are three different things.
"""

from __future__ import annotations

import json
import typing
from dataclasses import MISSING as dataclasses_MISSING, dataclass, field, asdict, fields
from enum import Enum
from typing import Optional

# Version of the LogEvent contract (docs/interface_contract.md,
# docs/logevent.schema.json). Bump the major part for a breaking change
# (removed/renamed field, changed meaning); the minor part for additive ones.
LOGEVENT_SCHEMA_VERSION = "1.0"


class LogFormat(str, Enum):
    BRACKET = "bracket"            # the original "[LEVEL] service: message" format
    SYSLOG = "syslog"
    JSON_LOG = "json_log"
    APACHE_ACCESS = "apache_access"
    KEY_VALUE = "key_value"
    HDFS = "hdfs"                  # LogHub HDFS: "yymmdd HHMMSS tid LEVEL component: msg"
    PLAIN_TEXT = "plain_text"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class ParseStatus(str, Enum):
    PARSED = "parsed"                          # a structured parser matched
    FALLBACK_PLAIN_TEXT = "fallback_plain_text"  # no format matched
    FAILED = "failed"                          # a parser raised; line kept as plain text


@dataclass
class LogEvent:
    """One canonical, normalized log record."""

    # --- traceability -------------------------------------------------------
    event_id: str
    source_file: str            # path relative to the dataset directory
    line_number: int            # 1-based line in source_file
    raw_message: str            # the exact original line (never modified)
    message: str                # message body as extracted by the parser
    normalized_message: Optional[str] = None   # message after normalization
    dataset_id: Optional[str] = None
    template: Optional[str] = None

    # --- origin ---------------------------------------------------------------
    timestamp_raw: Optional[str] = None
    timestamp_iso: Optional[str] = None
    host: Optional[str] = None
    service: Optional[str] = None     # logical service (from the line or dataset config)
    process: Optional[str] = None     # OS process/program name, only if the line has one
    pid: Optional[str] = None         # OS process id, only if the line has one

    # --- severity / classification -----------------------------------------
    severity: str = Severity.UNKNOWN.value
    severity_raw: Optional[str] = None
    log_format: str = LogFormat.UNKNOWN.value
    event_type: Optional[str] = None  # rule-based AUXILIARY label, not ground truth

    # --- correlation ----------------------------------------------------------
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    exception: Optional[str] = None

    # --- parsing provenance -----------------------------------------------
    parser_name: Optional[str] = None
    parse_confidence: Optional[float] = None  # structural completeness in [0, 1], NOT a probability
    metadata: dict = field(default_factory=dict)

    entities: dict = field(default_factory=dict)

    # --- Phase 2 outputs ------------------------------------------------------
    semantic_cluster: Optional[int] = None
    embedding_index: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "LogEvent":
        """Tolerant of older/newer files: unknown keys are ignored and
        missing optional keys take their defaults."""
        known = {f.name for f in fields(LogEvent)}
        return LogEvent(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------

_NOT_NULL_AFTER_PIPELINE = (
    "normalized_message", "template", "event_type", "parser_name",
    "parse_confidence", "semantic_cluster", "embedding_index",
)


def validate_event(event: "LogEvent", final: bool = True) -> list[str]:
    """Checks one event against the contract; returns a list of problems
    (empty = valid). `final=True` is for events that went through the whole
    pipeline (all derived fields must be set); `final=False` only checks the
    fields a parser must always provide."""
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(f"{event.event_id or '<no id>'}: {msg}")

    for name in ("event_id", "source_file"):
        if not isinstance(getattr(event, name), str) or not getattr(event, name):
            bad(f"{name} must be a non-empty string")
    if not isinstance(event.raw_message, str):
        bad("raw_message must be a string")
    if not isinstance(event.message, str):
        bad("message must be a string")
    if not isinstance(event.line_number, int) or isinstance(event.line_number, bool) or event.line_number < 1:
        bad("line_number must be an int >= 1")
    if event.severity not in {s.value for s in Severity}:
        bad(f"severity '{event.severity}' is not canonical")
    if event.log_format not in {f.value for f in LogFormat}:
        bad(f"log_format '{event.log_format}' is unknown")
    if event.parse_confidence is not None and not (0.0 <= event.parse_confidence <= 1.0):
        bad("parse_confidence must be in [0, 1]")
    if not isinstance(event.metadata, dict):
        bad("metadata must be a dict")
    elif event.metadata.get("parse_status") not in {p.value for p in ParseStatus}:
        bad("metadata.parse_status must be parsed / fallback_plain_text / failed")
    if not isinstance(event.entities, dict):
        bad("entities must be a dict")
    for name in ("timestamp_raw", "timestamp_iso", "host", "service", "process", "pid",
                 "request_id", "trace_id", "exception", "normalized_message", "template"):
        value = getattr(event, name)
        if value is not None and not isinstance(value, str):
            bad(f"{name} must be a string or None")
    if event.timestamp_iso is not None and event.timestamp_raw is None:
        bad("timestamp_iso is set but timestamp_raw is None")
    if event.process is not None and event.process == event.service and event.metadata.get("service_source") != "syslog_tag":
        bad("process equals service outside a syslog TAG (process must not be copied from service)")
    if final:
        for name in _NOT_NULL_AFTER_PIPELINE:
            if getattr(event, name) is None:
                bad(f"{name} is None after the pipeline")
    return problems


def logevent_json_schema() -> dict:
    """JSON Schema (draft 2020-12) generated from the dataclass, so the
    published schema cannot drift from the code."""
    hints = typing.get_type_hints(LogEvent)
    enums = {
        "severity": [s.value for s in Severity],
        "log_format": [f.value for f in LogFormat],
    }
    base = {str: "string", int: "integer", float: "number", dict: "object", bool: "boolean"}
    props: dict = {}
    required: list[str] = []
    for f in fields(LogEvent):
        hint = hints[f.name]
        nullable = False
        if typing.get_origin(hint) is typing.Union:
            args = [a for a in typing.get_args(hint) if a is not type(None)]
            nullable = len(args) != len(typing.get_args(hint))
            hint = args[0]
        json_type = base[hint]
        prop: dict = {"type": [json_type, "null"] if nullable else json_type}
        if f.name in enums:
            prop["enum"] = enums[f.name]
        props[f.name] = prop
        if f.default is dataclasses_MISSING and f.default_factory is dataclasses_MISSING:
            required.append(f.name)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LogEvent",
        "description": f"Canonical log event, contract version {LOGEVENT_SCHEMA_VERSION}. "
                       "See docs/interface_contract.md for field semantics.",
        "x-schema-version": LOGEVENT_SCHEMA_VERSION,
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
