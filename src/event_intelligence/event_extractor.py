"""
Event extraction (Phase 2: "Event extraction").

1. Template mining: masks variable tokens (IPs, numbers, UUIDs, paths) so
   structurally-identical log lines collapse to the same template.
2. Rule-based AUXILIARY event classification (`event_type`): keyword rules
   that give a cheap categorical feature. It is a heuristic, not semantic
   understanding and not ground truth. It only *adds* a label -- it never
   rewrites `message`, `normalized_message` or `template`.

Both steps read `normalized_message` (falling back to `message` if the event
has not been normalized). The original `raw_message` is never touched.

This is a lightweight Drain-inspired masking pass, not a full Drain/Spell
streaming implementation -- swap in `drain3` here later if the paper's
baselines need the real algorithm; only `LogEvent.template` needs to keep
being filled.
"""

from __future__ import annotations

import re

from src import config
from src.preprocessing.schema import LogEvent


_MASK_PATTERNS = {name: re.compile(pat) for name, pat in config.TEMPLATE_MASK_PATTERNS.items()}

# Order = priority (first match wins). Rules describe *failure/topic
# wording*, so a bare keyword ("database", "timeout") is not enough:
# "database connection established" is NOT a db_error.
_EVENT_TYPE_RULES: list[tuple[str, re.Pattern]] = [
    ("connection_failure", re.compile(
        r"failed to connect|connection (?:refused|reset|failed|timed? ?out|timeout)"
        r"|connect(?:ion)? timeout|timeout after|timed out", re.I)),
    ("auth_event", re.compile(r"logged in|login|authentication|unauthorized", re.I)),
    ("resource_exhaustion", re.compile(r"out of memory|capacity|disk full|pool at \d+%", re.I)),
    ("db_error", re.compile(
        r"deadlock|query failed"
        r"|database\b.{0,40}\b(?:error|failed|failure|unavailable|down|unreachable)"
        r"|(?:error|failed|failure)\b.{0,40}\bdatabase", re.I)),
    ("payment_failure", re.compile(r"payment declined|insufficient funds|charge failed", re.I)),
    ("http_error", re.compile(
        r"(?:status[= :]+|returned |-> |HTTP/\d(?:\.\d)?\"? )[45]\d{2}\b", re.I)),
    ("health_check", re.compile(r"health check", re.I)),
    ("restart", re.compile(r"restart", re.I)),
]


def mine_template(message: str) -> str:
    template = message
    for name in ["UUID", "IP", "EMAIL", "HEX", "PATH", "NUM"]:
        template = _MASK_PATTERNS[name].sub(f"<{name}>", template)
    return template


def classify_event_type(message: str) -> str:
    """Rule-based auxiliary label; "other" when no rule matches."""
    for label, pattern in _EVENT_TYPE_RULES:
        if pattern.search(message):
            return label
    return "other"


def extract_event(event: LogEvent) -> LogEvent:
    text = event.normalized_message if event.normalized_message is not None else event.message
    event.template = mine_template(text)
    event.event_type = classify_event_type(text)
    return event


def extract_events(events: list[LogEvent]) -> list[LogEvent]:
    return [extract_event(e) for e in events]


def template_frequency(events: list[LogEvent]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for e in events:
        freq[e.template] = freq.get(e.template, 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: -kv[1]))
