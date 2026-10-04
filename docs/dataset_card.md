# Dataset card — LogHub HDFS_2k

Registered as `loghub_hdfs_2k` in `src/preprocessing/datasets.py`.
Run: `python -m src.pipeline --dataset loghub_hdfs_2k`

| | |
|---|---|
| **Dataset name** | LogHub — HDFS, 2k-line sample (`HDFS_2k`) |
| **Source** | https://github.com/logpai/loghub (`HDFS/HDFS_2k.log`, `HDFS_2k.log_structured.csv`, `HDFS_2k.log_templates.csv`, branch `master`) |
| **Version** | No release tag upstream. Pinned by SHA-256 (below); files fetched 2026-09-29 and re-verified against upstream on 2026-09-30. The upstream commit hash could not be recorded (GitHub API rate-limited at that time) — add it here when you next can. |
| **License** | LogHub states the datasets are "freely available for research or academic work" and asks that usage/distribution refer to the LogHub repository and cite the LogHub paper. No SPDX license file is provided — **confirm the terms with the LogHub README before redistributing the data or publishing derived artifacts.** |
| **Number of logs** | 2,000 lines (0 blank), one file, 2008-11-09 20:36:15 → 2008-11-11 10:20:17 |
| **Log formats** | 1 format: `yymmdd HHMMSS <thread-id> <LEVEL> <java-logger>: <message>` (parser: `hdfs`) |
| **Available labels** | Per line: `EventId` (E1–E14) and `EventTemplate`, from LogHub's `HDFS_2k.log_structured.csv`. **No anomaly / root-cause labels in this sample.** |
| **Ground-truth definition** | A line's ground-truth class is its LogHub `EventId`: the template (event type) LogHub assigns to it in `HDFS_2k.log_structured.csv`. It defines *what counts as the same event*. It is used for ARI / NMI of the semantic grouping. It is **not** an RCA label. |
| **Train/test split** | The pipeline is unsupervised (nothing is fit on labels), but hyperparameters are chosen by looking at metrics, so a split is used to avoid tuning on what we report. Deterministic, by position in log order: **tune** = lines 1–1000 (12 event types), **heldout** = lines 1001–2000 (13 event types; 11 shared). Choose configurations with `python -m src.evaluate_clustering --dataset loghub_hdfs_2k --split tune`; report them with `--split heldout`. `--split all` is exploration only. |
| **Preprocessing performed** | Parsed with the `hdfs` parser (line-terminator stripped; `raw_message` = exact line). Timestamp `yymmdd HHMMSS` → ISO-8601, **assumed UTC** (the logs carry no timezone). Thread id and Java logger name kept in `metadata` (`thread_id`, `component`), *not* in `pid` / `process`. `service` = `hdfs` for every line, attributed by dataset config (`metadata.service_source = "dataset_config"`), because the lines carry none. Template mining masks IPs, numbers, paths. No lines dropped. |

### Files and checksums (SHA-256)

```
7c967000980c086ed55fa6544ba4f05fe66d44622795e890c68caf8bbb635035  HDFS_2k.log
729df59774e3dde934044028546d2a55d5e3d4370b9d12fcebbe4c087b2bf7b4  HDFS_2k.log_structured.csv
a07307511f67c9dc1f41ae730ae60dcce8360f2c72742f0b8a3a9cf1a403d1db  HDFS_2k.log_templates.csv
```
Place them in `data/raw/loghub/HDFS_2k/` (or run `python scripts/fetch_loghub.py`, which downloads and verifies them).

### Contents

- Severity: INFO 1,920 · WARN 80 (no ERROR/CRITICAL).
- Components: `dfs.FSNamesystem` 659 · `dfs.DataNode$PacketResponder` 603 · `dfs.DataNode$DataXceiver` 454 · `dfs.FSDataset` 263 · `dfs.DataBlockScanner` 20 · `dfs.DataNode` 1.
- Event types by size: E6 314 · E10 311 · E11 292 · E13 292 · E9 263 · E8 224 · E7 115 · E1 80 · E3 80 · E14 20 · E4 5 · E12 2 · E2 1 · E5 1 (14 total).

### Baseline numbers (for orientation, not a result)

TF-IDF+SVD embeddings (the real sentence-transformers model was not available where this was run), seed 42, `tune` split, agglomerative @ 0.35:

| representation | clusters | ARI | NMI |
|---|---|---|---|
| message | 180 | 0.476 | 0.645 |
| template | 12 | 1.000 | 1.000 |
| hybrid | 77 | 0.885 | 0.864 |

The representation matters a lot on this data — which is why it is an experimental switch, not an assumption. Rerun with the sentence-transformers backend before quoting anything.

### Known limitations

1. **Not an RCA dataset.** The 2k sample has no anomaly or root-cause labels; ground truth here only supports evaluating *semantic grouping*. Block-level normal/anomaly labels exist for the full HDFS_v1 set (LogHub / Xu et al., SOSP'09) and are not integrated yet. No claim about RCA performance can be made from this dataset.
2. **Easy for grouping.** 14 near-deterministic templates; with template embeddings ARI ≈ 1.0, so it cannot separate good methods from mediocre ones. Add a harder dataset before drawing conclusions.
3. **Little structure for downstream work.** One system, one service (`hdfs`), no hostnames, no request/trace ids, only INFO/WARN. Dependency-graph, severity-scoring and cross-service correlation cannot be exercised on it.
4. **Template masking leaves the minus sign of negative block ids** (`blk_-123` → `blk_-<NUM>`), so 14 true event types become 29 distinct templates. Clustering still merges them; a dedicated block-id mask would remove the artifact.
5. **Timestamps** have no timezone (UTC assumed) and no year in the raw form (2000 + yy).
6. Small sample (2,000 lines; the full HDFS_v1 set has millions); how LogHub drew it is not documented here.

### Citation

Zhu, He, He, Liu, Lyu. *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.* ISSRE 2023. https://arxiv.org/abs/2008.06448
Xu, Huang, Fox, Patterson, Jordan. *Detecting Large-Scale System Problems by Mining Console Logs.* SOSP 2009.

---

## Synthetic demo data (not a research dataset)

`python -m src.pipeline --demo` reads generated fixtures (`src/preprocessing/synthetic_generator.py`, 5 formats × 60 lines, seeded, strictly increasing timestamps). They exist to test parsers and pipeline connectivity. Nothing produced from them may be reported as a result; the pipeline labels every such run `mode: DEMO`, `is_synthetic: true`.
