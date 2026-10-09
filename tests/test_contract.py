"""The LogEvent contract: dataclass <-> JSON Schema <-> documentation <-> real output."""

import json
import re
import subprocess
import sys
from dataclasses import fields

import pytest

from src import config
from src.pipeline import run_pipeline
from src.preprocessing.parsers import parse_line
from src.preprocessing.normalizer import normalize_events
from src.preprocessing.schema import (
    LOGEVENT_SCHEMA_VERSION, LogEvent, logevent_json_schema, validate_event,
)

ROOT = config.ROOT_DIR
DOC = ROOT / "docs" / "interface_contract.md"


def _good_event():
    e = parse_line("f.log", 1, "2026-09-20 10:30:00 [ERROR] api: boom", dataset_id="d")
    normalize_events([e])
    e.template, e.event_type, e.semantic_cluster, e.embedding_index = "boom", "other", 0, 0
    return e


# --- the published JSON Schema cannot drift from the code ---------------------------------

def test_json_schema_file_matches_the_dataclass():
    on_disk = json.loads((ROOT / "docs" / "logevent.schema.json").read_text())
    assert on_disk == logevent_json_schema(), "run: python scripts/export_schema.py"
    assert on_disk["x-schema-version"] == LOGEVENT_SCHEMA_VERSION


def test_export_script_check_mode_passes():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_schema.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_schema_lists_every_dataclass_field_and_forbids_extras():
    schema = logevent_json_schema()
    assert set(schema["properties"]) == {f.name for f in fields(LogEvent)}
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"event_id", "source_file", "line_number", "raw_message", "message"}


def test_a_real_event_validates_against_the_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json.loads(_good_event().to_json()), logevent_json_schema())


# --- the documentation lists exactly the fields that exist --------------------------------

def test_interface_contract_documents_every_field():
    text = DOC.read_text()
    section = text.split("## `LogEvent` fields")[1]
    documented = set()
    for line in section.splitlines():
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| field"):
            first_cell = line.split("|")[1]
            documented |= set(re.findall(r"`([a-z_]+)`", first_cell))
    actual = {f.name for f in fields(LogEvent)}
    assert actual - documented == set(), f"undocumented fields: {actual - documented}"
    assert documented - actual == set(), f"documented but not in LogEvent: {documented - actual}"


def test_interface_contract_states_the_schema_version():
    assert f"`{LOGEVENT_SCHEMA_VERSION}`" in DOC.read_text()


# --- validate_event catches contract violations -------------------------------------------------

def test_a_good_event_is_valid():
    assert validate_event(_good_event()) == []


@pytest.mark.parametrize("mutate,expected", [
    (lambda e: setattr(e, "severity", "WARNING"), "severity"),
    (lambda e: setattr(e, "line_number", 0), "line_number"),
    (lambda e: setattr(e, "event_id", ""), "event_id"),
    (lambda e: setattr(e, "parse_confidence", 1.5), "parse_confidence"),
    (lambda e: setattr(e, "metadata", {}), "parse_status"),
    (lambda e: setattr(e, "message", 123), "message"),
    (lambda e: setattr(e, "timestamp_raw", None), "timestamp_iso"),
    (lambda e: (setattr(e, "process", "api"), setattr(e, "service", "api")), "process equals service"),
    (lambda e: setattr(e, "template", None), "template"),
    (lambda e: setattr(e, "log_format", "carrier-pigeon"), "log_format"),
])
def test_validate_event_reports_violations(mutate, expected):
    e = _good_event()
    mutate(e)
    problems = validate_event(e)
    assert problems and any(expected in p for p in problems), problems


def test_syslog_process_equal_to_service_is_allowed():
    e = parse_line("f.log", 1, "Sep 20 08:01:03 10.0.0.11 ERROR auth-service[1]: boom")
    assert e.process == e.service == "auth-service"
    assert validate_event(e, final=False) == []


def test_non_final_validation_does_not_require_pipeline_fields():
    e = parse_line("f.log", 1, "plain")
    assert validate_event(e, final=False) == []
    assert validate_event(e, final=True)          # normalized_message, template, ... are still None


# --- real pipeline output satisfies the contract ---------------------------------------------------

def test_every_event_in_the_demo_artifacts_validates_and_roundtrips(tmp_path):
    run_pipeline(demo=True, data_dir=tmp_path / "s", output_dir=tmp_path / "o", embedding_backend="tfidf")
    lines = (tmp_path / "o" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 300
    for line in lines:
        e = LogEvent.from_dict(json.loads(line))
        assert validate_event(e) == []
        assert e.to_json() == line                      # lossless round trip


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "loghub" / "HDFS_2k" / "HDFS_2k.log").exists(),
                    reason="run: python scripts/fetch_loghub.py")
def test_every_event_of_the_research_dataset_validates(tmp_path):
    result = run_pipeline(dataset="loghub_hdfs_2k", output_dir=tmp_path, embedding_backend="tfidf", save_outputs=False)
    assert all(validate_event(e) == [] for e in result["events"])
    assert result["report"]["stages"]["schema_validation"]["violations"] == 0
