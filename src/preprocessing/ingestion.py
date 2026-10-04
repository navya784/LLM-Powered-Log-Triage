"""
Log ingestion (Phase 1: "Add ingestion" -> "Multi-format ingestion").

Discovers a dataset's log files, parses every line into a `LogEvent`, and
reports parser-quality statistics.

This module NEVER generates data. Synthetic fixtures live in
`synthetic_generator.py` and are only used by the pipeline in explicit DEMO
mode. If a dataset's files are not on disk, ingestion raises
`DatasetMissingError` instead of substituting anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from src.preprocessing.datasets import DatasetMissingError, DatasetSpec
from src.preprocessing.parsers import parse_line
from src.preprocessing.schema import LogEvent, ParseStatus


def iter_raw_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yields (1-based line number, line without its line terminator).
    Blank lines are skipped but still counted in the line numbering."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if line.strip():
                yield i, line


def discover_log_files(directory: Path, patterns: tuple[str, ...] = ("*.log", "*.txt", "*.json", "*.jsonl")) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    # de-duplicate while keeping order (patterns can overlap)
    return list(dict.fromkeys(files))


def find_dataset_files(spec: DatasetSpec, directory: Optional[Path] = None) -> List[Path]:
    """Files of `spec` on disk. Raises DatasetMissingError -- never falls back
    to another directory or to generated data."""
    directory = directory or spec.resolve_dir()
    if not directory.is_dir():
        hint = f" Fetch it with: {spec.fetch_hint}" if spec.fetch_hint else ""
        raise DatasetMissingError(
            f"Dataset '{spec.dataset_id}' not found: directory {directory} does not exist.{hint}"
        )
    files = discover_log_files(directory, spec.file_patterns)
    if not files:
        hint = f" Fetch it with: {spec.fetch_hint}" if spec.fetch_hint else ""
        raise DatasetMissingError(
            f"Dataset '{spec.dataset_id}' has no log files in {directory} "
            f"(looked for {list(spec.file_patterns)}).{hint}"
        )
    return files


# ---------------------------------------------------------------------------
# Parser-quality statistics
# ---------------------------------------------------------------------------

@dataclass
class ParseStats:
    """Every non-blank line lands in exactly one bucket:
    parsed + fallback_plain_text + failed == lines_read."""

    lines_read: int = 0
    parsed: int = 0                 # a structured parser matched
    fallback_plain_text: int = 0    # no format matched; kept as plain text
    failed: int = 0                 # a parser raised; kept as plain text
    blank_lines_skipped: int = 0
    formats: Counter = field(default_factory=Counter)      # structured formats only
    per_file: dict = field(default_factory=dict)

    def record(self, event: LogEvent) -> None:
        status = event.metadata.get("parse_status")
        self.lines_read += 1
        if status == ParseStatus.PARSED.value:
            self.parsed += 1
            self.formats[event.log_format] += 1
        elif status == ParseStatus.FAILED.value:
            self.failed += 1
        else:
            self.fallback_plain_text += 1

    @property
    def parse_failure_rate(self) -> float:
        return round(self.failed / self.lines_read, 6) if self.lines_read else 0.0

    @property
    def fallback_rate(self) -> float:
        return round(self.fallback_plain_text / self.lines_read, 6) if self.lines_read else 0.0

    def to_dict(self) -> dict:
        return {
            "lines_read": self.lines_read,
            "parsed": self.parsed,
            "fallback_plain_text": self.fallback_plain_text,
            "failed": self.failed,
            "parse_failure_rate": self.parse_failure_rate,
            "fallback_rate": self.fallback_rate,
            "blank_lines_skipped": self.blank_lines_skipped,
            "formats": dict(sorted(self.formats.items())),
            "per_file": self.per_file,
        }


def _count_blank_lines(path: Path, non_blank: int) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        total = sum(1 for _ in f)
    return total - non_blank


def ingest_files(
    files: List[Path],
    base_dir: Path,
    spec: Optional[DatasetSpec] = None,
    dataset_id: Optional[str] = None,
) -> tuple[List[LogEvent], ParseStats]:
    """Parses `files` into events. `source_file` is stored relative to
    `base_dir` so ids and ground-truth keys do not depend on the machine."""
    stats = ParseStats()
    events: List[LogEvent] = []
    dataset_id = dataset_id or (spec.dataset_id if spec else None)
    for path in files:
        rel = path.relative_to(base_dir).as_posix() if path.is_relative_to(base_dir) else path.name
        service_hint = spec.service_for(rel) if spec else None
        n_before = stats.lines_read
        for line_no, raw_line in iter_raw_lines(path):
            event = parse_line(rel, line_no, raw_line, dataset_id=dataset_id, service_hint=service_hint)
            stats.record(event)
            events.append(event)
        n_file = stats.lines_read - n_before
        blanks = _count_blank_lines(path, n_file)
        stats.blank_lines_skipped += blanks
        stats.per_file[rel] = {"lines_read": n_file, "blank_lines_skipped": blanks}
    return events, stats


def ingest_dataset(spec: DatasetSpec, directory: Optional[Path] = None) -> tuple[List[LogEvent], ParseStats, List[Path]]:
    directory = directory or spec.resolve_dir()
    files = find_dataset_files(spec, directory)
    events, stats = ingest_files(files, directory, spec)
    return events, stats, files
