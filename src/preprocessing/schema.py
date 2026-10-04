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
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Optional


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
