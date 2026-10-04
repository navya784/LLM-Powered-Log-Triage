from src.preprocessing.schema import LogEvent


def _event(**kw):
    return LogEvent(event_id="e1", source_file="f.log", line_number=1, raw_message="raw", message="msg", **kw)


def test_schema_has_the_extended_fields_with_safe_defaults():
    e = _event()
    for name in ("request_id", "trace_id", "exception", "parser_name",
                 "parse_confidence", "dataset_id", "normalized_message"):
        assert getattr(e, name) is None
    assert e.metadata == {} and e.entities == {}
    assert e.event_id and e.source_file and e.line_number and e.raw_message


def test_metadata_default_is_not_shared_between_instances():
    a, b = _event(), _event()
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_roundtrip_through_json():
    e = _event(dataset_id="d", request_id="r", metadata={"k": [1, 2]}, parse_confidence=0.5)
    import json
    assert LogEvent.from_dict(json.loads(e.to_json())) == e


def test_from_dict_ignores_unknown_and_tolerates_missing_keys():
    d = _event().to_dict()
    d["some_future_field"] = 1
    del d["trace_id"]
    assert LogEvent.from_dict(d).trace_id is None
