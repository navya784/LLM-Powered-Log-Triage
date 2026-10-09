# Review follow-up: what changed and where

| # | Change | Where |
|---|---|---|
| 1, 16 | Synthetic data only in explicit DEMO mode; RESEARCH needs `--dataset`; no dataset ⇒ "dataset missing" error | `pipeline.py` (`RunMode`, CLI), `preprocessing/ingestion.py` (no generator), `preprocessing/synthetic_generator.py` |
| 2 | Real dataset (LogHub HDFS_2k) integrated + documented | `preprocessing/datasets.py`, `data/raw/loghub/HDFS_2k/`, `scripts/fetch_loghub.py`, `docs/dataset_card.md`, `hdfs` parser |
| 3 | Chronological synthetic timestamps | `synthetic_generator.py` (`_advance`), `tests/test_synthetic_generator.py` |
| 4 | Severity aliases normalized before enum validation; original kept in `severity_raw` | `parsers.parse_line`, syslog/HDFS regexes accept all aliases |
| 5 | No hardcoded `gateway`; service from dataset config, provenance in `metadata.service_source` | `parsers.py`, `datasets.py` (`service_map`, `default_service`) |
| 6 | `service` / `process` / `pid` separated; nothing invented | `parsers.py` |
| 7 | `LogEvent` extended (+ `normalized_message`, `severity_raw`) | `preprocessing/schema.py` |
| 8 | Raw and normalized kept side by side | `normalizer.normalize_event` |
| 9 | Parser-quality statistics | `ingestion.ParseStats` → `pipeline_report.json` |
| 10 | Tests for every format + severity + event id | `tests/test_parsers.py` |
| 11 | Clustering tests; length checks instead of silent `zip` truncation | `tests/test_clustering.py`, `clustering.assign_clusters` |
| 12 | Embedding backend never switched silently; recorded in every result | `event_intelligence/embeddings.py` |
| 13 | Clustering evaluation + recorded configuration | `event_intelligence/cluster_evaluation.py`, `src/evaluate_clustering.py`, `clustering.run_clustering` |
| 14 | message / template / hybrid representations | `embeddings.event_text`, `--representation` |
| 15 | Dataset-aware event ids | `parsers.make_event_id` |
| 17 | Event classifier documented as auxiliary; rules tightened; negative tests | `event_extractor.py`, `tests/test_event_extractor.py` |
| 18 | Hand-off artifacts (`events.jsonl/.parquet`, `embeddings.npy`, `semantic_groups.json`) | `pipeline.py` |
| 19 | Interface documentation | `docs/interface_contract.md` |

Deviations / additions worth knowing about: outputs go to `data/processed/<dataset_id>/` (one folder per dataset, so runs cannot overwrite each other) and demo output to `outputs/demo/`; `semantic_groups.json` is an envelope `{schema_version, run, groups}` instead of a bare list so it carries its own provenance; `PARSERS` is now a list of `(name, function)` pairs.

Note: `parsers.py` still imports `extract_log_fields` from your existing `src/preprocessing/log_preprocessor.py`, which is not part of this add-on. `tests/conftest.py` installs a minimal stand-in only if that module cannot be imported, so the add-on's tests also run standalone; with your real module present it is never used.

## Review 1 additions

- **Bug fix:** a JSON log with a non-string field (for example `"msg": 123` or a numeric epoch `ts`) used to crash normalization; all JSON fields are now coerced to text, and a normalization error on any single event is recorded in `metadata.normalize_error` instead of stopping the run.
- **Timestamps:** epoch seconds/milliseconds are understood; `metadata.timestamp_unparseable` distinguishes an invalid timestamp from a missing one; the report counts missing, unparseable and assumed-year timestamps separately.
- **Contract:** `LOGEVENT_SCHEMA_VERSION = "1.0"`, `validate_event`, generated `docs/logevent.schema.json`, checked on every run and in tests.
- **Reproducibility:** `--config` JSON files (`configs/loghub_hdfs_2k.json`), resolved config + package versions recorded in the report, determinism tests.
- **Evaluation:** purity, homogeneity, completeness and a mixed-cluster listing added; every evaluation states that it measures grouping, not RCA.
- **Review material:** `scripts/review1_report.py` generates `docs/review1_report.md`; `docs/review1_checklist.md` maps each Review 1 item to its evidence.
- **Tests:** `tests/test_robustness.py`, `tests/test_reproducibility.py`, `tests/test_contract.py`.
