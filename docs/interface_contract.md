# Interface contract: Member 1 → Member 2

Member 1 (ingestion, normalization, event intelligence) hands Member 2 a set of
files. **Member 2 never needs to re-read or re-parse raw logs.**

```
Input (raw log files of a registered dataset)
   ↓  parsers.parse_line            one LogEvent per non-blank line
LogEvent                            raw_message, message, source location, parsed fields
   ↓  normalizer.normalize_events   fills normalized_message, timestamp_iso, canonical severity, entities
Normalized LogEvent
   ↓  event_extractor               template (masked message) + rule-based auxiliary event_type
   ↓  embeddings.embed_events       one row per event
Embedding matrix   (N × D, float32, L2-normalized; row i ↔ events[i])
   ↓  clustering.run_clustering     one label per event; -1 = noise
Cluster labels
   ↓  clustering.summarize_clusters
Semantic-group artifact   (semantic_groups.json)
```

## Running

```
python -m src.pipeline --dataset loghub_hdfs_2k     # RESEARCH  -> data/processed/loghub_hdfs_2k/
python -m src.pipeline --demo                       # DEMO      -> outputs/demo/   (synthetic; not a result)
```
No `--dataset` and no `--demo` → the pipeline reports the dataset as missing (exit code 2). It never generates data on its own. Research runs also fail (exit 3) if the requested embedding model is unavailable; `--embedding-backend tfidf` is an explicit, recorded choice. Other switches: `--representation {message,template,hybrid}`, `--algorithm`, `--distance-threshold`, `--n-clusters`, `--seed`, `--save-intermediate`.

Python: `from src.pipeline import run_pipeline` → `run_pipeline(dataset="loghub_hdfs_2k")` returns `{"events": [LogEvent], "embeddings": ndarray, "semantic_groups": [...], "report": {...}, "output_dir": Path}`.

Extra dependency: `pyarrow` (parquet). Optional: `hdbscan` (otherwise `sklearn.cluster.HDBSCAN` is used and the implementation is recorded).

## Artifacts (`data/processed/<dataset_id>/`)

| file | content |
|---|---|
| `events.jsonl` | one `LogEvent` per line, in source order — final state after all stages |
| `events.parquet` | same rows; `entities` and `metadata` are JSON *strings* |
| `embeddings.npy` | `(N, D)` float32; row `i` belongs to the event at line `i` of `events.jsonl` (`embedding_index == i`) |
| `semantic_groups.json` | `{"schema_version", "run", "groups"}` — see below |
| `pipeline_report.json` | parse statistics, normalization/extraction counts, embedding + clustering configuration, clustering evaluation |

`--save-intermediate` additionally writes `parsed_events.jsonl`, `normalized_events.jsonl`, `extracted_events.jsonl`.

### `semantic_groups.json`

```json
{
  "schema_version": 1,
  "run": {
    "mode": "RESEARCH", "dataset_id": "loghub_hdfs_2k", "is_synthetic": false, "n_events": 2000,
    "embedding":  {"embedding_backend": "sentence-transformers", "embedding_model": "…", "embedding_dimension": 384,
                   "representation": "template", "fallback_from": null, "…": "…"},
    "clustering": {"algorithm": "agglomerative", "distance_threshold": 0.35, "seed": 42,
                   "n_clusters": 14, "n_noise": 0, "…": "…"}
  },
  "groups": [
    {
      "cluster_id": 3,
      "is_noise": false,
      "size": 314,
      "event_ids": ["…"],
      "representative_template": "…",
      "distinct_templates": 2,
      "services": ["hdfs"],
      "hosts": [],
      "event_types": ["other"],
      "severity_distribution": {"INFO": 314},
      "time_span": {"start": "2008-11-09T20:40:05+00:00", "end": "2008-11-11T10:13:16+00:00"}
    }
  ]
}
```
Groups are ordered by size (largest first), the noise group (`cluster_id: -1`, HDBSCAN only) last. Every event appears in exactly one group. `time_span` values are `null` when no member has a parseable timestamp.

## `LogEvent` fields — what you can rely on

**Rule: a field is filled only if the log line (or explicit dataset configuration) provides it. `null` means "not available", never "guessed".** Check `metadata` for provenance.

| field | guarantee |
|---|---|
| `event_id` | 16-hex id = SHA-1 of `dataset_id`, `source_file`, `line_number`, `raw_message` (unit-separator joined). Deterministic and dataset-aware; unique in practice across datasets. |
| `dataset_id`, `source_file`, `line_number` | Where the event came from. `source_file` is relative to the dataset directory; `line_number` is 1-based. Always set (`dataset_id` is `null` only for events built outside the pipeline). |
| `raw_message` | The exact original line (without its line terminator). **Never modified.** Use it to show the evidence behind an event id. |
| `message` | Message body as extracted by the parser (unmodified by normalization). |
| `normalized_message` | Message after normalization (e.g. embedded `service[pid]:` prefix removed). Set for every event that went through the pipeline. |
| `template` | `normalized_message` with IPs, numbers, paths, … masked. |
| `timestamp_raw` / `timestamp_iso` | Raw string as parsed / ISO-8601 UTC, or `null` if absent or unparseable. **Timestamps without a timezone are assumed UTC**, and syslog-style stamps without a year get an assumed year; the latter is flagged with `metadata.timestamp_year_assumed = true`. |
| `severity` / `severity_raw` | Canonical `DEBUG, INFO, WARN, ERROR, CRITICAL` or `UNKNOWN`; aliases (`WARNING, CRIT, FATAL, ERR, …`) are mapped before validation. `severity_raw` is the token in the line. Apache-style access lines have no severity token: it is derived from the HTTP status (`metadata.severity_source`). |
| `host` | Host/IP if the line carries one, else `null`. |
| `service` | Logical service. From the line if it has one (`metadata.service_source = "log_line"` or `"syslog_tag"`), otherwise from **dataset configuration** (`"dataset_config"`), otherwise `null`. A `null` service is legitimate (e.g. a raw access log with no mapping) — do not assume one. |
| `process`, `pid` | OS process name / id, only when the line has them (syslog `tag[pid]`; JSON/key-value `process`/`pid` fields). `service`, `process` and `pid` are different concepts. `process` is **not** a copy of `service` except that a syslog TAG is by definition the program name. |
| `request_id`, `trace_id`, `exception` | Read from JSON / key-value fields when present, else `null`. |
| `parser_name`, `log_format` | Which parser handled the line (`bracket, json, syslog, hdfs, apache, key_value, plain_text`). |
| `parse_confidence` | Structural completeness in [0, 1] (share of timestamp / severity / message extracted); `0.0` for plain-text fallback. A diagnostic heuristic, **not a probability**. |
| `metadata` | Free-form provenance. Always has `parse_status` (`parsed`, `fallback_plain_text` or `failed`). May hold `service_source`, `component`, `thread_id`, `http_status`, `extra_fields`, `parse_error`, … |
| `entities` | Extracted `ips`, `status_code`, `latency_ms`, `error_code` when present (from `normalized_message`). |
| `event_type` | **Rule-based auxiliary label** (keyword rules; `other` if none match). A cheap extra feature, not ground truth and not semantic understanding. It never changes `message` / `template`. |
| `semantic_cluster` | Cluster id from the run (`-1` = noise). Meaningful only together with the run's `clustering` configuration. |
| `embedding_index` | Row of this event in `embeddings.npy`. |

## Provenance you should carry forward

Two runs are comparable only if `run.embedding` (backend, model, dimension, representation) and `run.clustering` (algorithm, parameters, seed) match. Results from `mode: DEMO` / `is_synthetic: true` must never be reported as research results.
