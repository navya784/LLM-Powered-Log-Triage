"""
Phase 1 + Phase 2 pipeline orchestrator:

Raw logs -> Parsing -> Normalization -> Event extraction -> Embeddings
         -> Clustering (+ evaluation) -> Hand-off artifacts

Two explicit run modes -- the pipeline always knows which one it is in:

  RESEARCH   python -m src.pipeline --dataset loghub_hdfs_2k
             Real registered dataset only. No dataset / missing files /
             unavailable embedding model => it stops with an error. Nothing
             synthetic is ever generated or substituted.
             Output: data/processed/<dataset_id>/

  DEMO       python -m src.pipeline --demo
             Synthetic fixture logs (src/preprocessing/synthetic_generator.py)
             for testing parsers and pipeline connectivity. Results are NOT
             research results. Output: outputs/demo/

Every run writes `mode`, `dataset_id` and `is_synthetic` into its report and
into semantic_groups.json, so demo artifacts cannot be mistaken for research
ones. Programmatic use: `run_pipeline(dataset=..., ...)` /
`run_pipeline(demo=True, ...)` -- the public interface for Phase 3 and 4 (see
docs/interface_contract.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from src import config
from src.event_intelligence.cluster_evaluation import evaluate_clustering
from src.event_intelligence.clustering import assign_clusters_with_params, summarize_clusters
from src.event_intelligence.embeddings import EmbeddingBackendUnavailable, embed_events
from src.event_intelligence.event_extractor import extract_events, template_frequency
from src.preprocessing.datasets import DEMO_SYNTHETIC, DatasetMissingError, DatasetSpec, get_dataset
from src.preprocessing.ingestion import find_dataset_files, ingest_files
from src.preprocessing.normalizer import normalize_events
from src.preprocessing.schema import LogEvent

SCHEMA_VERSION = 1


class RunMode(str, Enum):
    RESEARCH = "RESEARCH"
    DEMO = "DEMO"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_jsonl(events: list[LogEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(e.to_json() + "\n")


def _write_parquet(events: list[LogEvent], path: Path) -> bool:
    """entities / metadata are stored as JSON strings (parquet cannot store
    empty or heterogeneous dicts). Returns False if pyarrow is missing."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    rows = []
    for e in events:
        row = e.to_dict()
        row["entities"] = json.dumps(row["entities"], ensure_ascii=False)
        row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return True


def _load_ground_truth(spec: DatasetSpec, directory: Path, events: list[LogEvent]) -> Optional[dict]:
    """Aligns the dataset's ground-truth labels with `events` (by
    (source_file, line_number)). Returns None if the dataset has no labels."""
    if spec.ground_truth_loader is None:
        return None
    labels = spec.ground_truth_loader(directory)
    truth = [labels.get((e.source_file, e.line_number)) for e in events]
    n_missing = sum(t is None for t in truth)
    if n_missing:
        return {"labels": None, "coverage": round(1 - n_missing / len(events), 6),
                "note": f"{n_missing} events have no ground-truth label; external metrics skipped"}
    return {"labels": truth, "coverage": 1.0}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset: Optional[str] = None,
    *,
    demo: bool = False,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_outputs: bool = True,
    save_intermediate: bool = False,
    embedding_backend: Optional[str] = None,
    embedding_model: Optional[str] = None,
    representation: Optional[str] = None,
    algorithm: Optional[str] = None,
    clustering_params: Optional[dict] = None,
    seed: Optional[int] = None,
) -> dict:
    """Runs the pipeline. Exactly one of `dataset` (RESEARCH) or `demo=True`
    (DEMO) must be given; with neither, DatasetMissingError is raised.

    `data_dir` / `output_dir` override where logs are read from / artifacts
    are written (mainly for tests)."""
    if demo and dataset:
        raise ValueError("Pass either a dataset (research) or demo=True, not both.")
    if not demo and not dataset:
        raise DatasetMissingError(
            "No dataset supplied. Use dataset='<name>' for a research run "
            f"(registered: {sorted(n for n in _research_names())}) or demo=True for the "
            "synthetic demo. The pipeline does not generate data on its own."
        )

    mode = RunMode.DEMO if demo else RunMode.RESEARCH
    if mode is RunMode.DEMO:
        spec = DEMO_SYNTHETIC
        directory = data_dir or config.SAMPLE_LOG_DIR
        from src.preprocessing.synthetic_generator import generate_sample_dataset

        from src.preprocessing.ingestion import discover_log_files

        if not (directory.is_dir() and discover_log_files(directory, spec.file_patterns)):
            print(f"[pipeline] DEMO mode: generating synthetic fixture logs in {directory}")
            generate_sample_dataset(out_dir=directory)
        out_dir = output_dir or (config.OUTPUT_DIR / "demo")
    else:
        spec = get_dataset(dataset)
        if spec.is_synthetic:
            raise ValueError(f"'{spec.dataset_id}' is synthetic and cannot be used for research. Use demo=True.")
        directory = data_dir or spec.resolve_dir()
        out_dir = output_dir or (config.PROCESSED_DIR / spec.dataset_id)

    # Research runs never silently change methods; demo runs may fall back
    # (and the fallback is recorded).
    allow_fallback = mode is RunMode.DEMO
    seed = seed if seed is not None else config.RANDOM_SEED

    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode.value,
        "dataset_id": spec.dataset_id,
        "dataset_name": spec.name,
        "is_synthetic": spec.is_synthetic,
        "stages": {},
    }
    if spec.is_synthetic:
        report["warning"] = "DEMO run on synthetic data: not a research result."
    t0 = time.time()

    # 1. Ingestion + parsing ------------------------------------------------
    files = find_dataset_files(spec, directory)
    events, parse_stats = ingest_files(files, directory, spec)
    report["stages"]["ingestion"] = {
        "directory": str(directory), "n_files": len(files),
        "files": [p.relative_to(directory).as_posix() for p in files],
    }
    report["stages"]["parsing"] = parse_stats.to_dict()
    if not events:
        raise DatasetMissingError(f"Dataset '{spec.dataset_id}' contains no non-blank log lines in {directory}.")
    if save_outputs and save_intermediate:
        _write_jsonl(events, out_dir / config.INTERMEDIATE_NAMES["parsed"])

    # 2. Normalization ------------------------------------------------------
    events = normalize_events(events)
    report["stages"]["normalization"] = {
        "n_events": len(events),
        "unresolved_timestamps": sum(1 for e in events if e.timestamp_iso is None),
        "unresolved_severities": sum(1 for e in events if e.severity == "UNKNOWN"),
        "events_without_service": sum(1 for e in events if e.service is None),
        "service_attributed_by_dataset_config": sum(
            1 for e in events if e.metadata.get("service_source") == "dataset_config"),
    }
    if save_outputs and save_intermediate:
        _write_jsonl(events, out_dir / config.INTERMEDIATE_NAMES["normalized"])

    # 3. Event extraction ---------------------------------------------------
    events = extract_events(events)
    freq = template_frequency(events)
    type_counts: dict[str, int] = {}
    for e in events:
        type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
    report["stages"]["event_extraction"] = {
        "n_events": len(events),
        "n_distinct_templates": len(freq),
        "top_templates": list(freq.items())[:10],
        "event_type_counts (rule-based auxiliary label)": dict(sorted(type_counts.items())),
    }
    if save_outputs and save_intermediate:
        _write_jsonl(events, out_dir / config.INTERMEDIATE_NAMES["extracted"])

    # 4. Embeddings ---------------------------------------------------------
    embeddings, emb_info = embed_events(
        events,
        save_path=(out_dir / config.ARTIFACT_NAMES["embeddings"]) if save_outputs else None,
        representation=representation, backend=embedding_backend,
        model_name=embedding_model, allow_fallback=allow_fallback,
    )
    report["stages"]["embeddings"] = {"shape": list(embeddings.shape), **emb_info.to_dict()}

    # 5. Clustering + evaluation -------------------------------------------
    events, cluster_result = assign_clusters_with_params(
        events, embeddings, algorithm, seed=seed, allow_fallback=allow_fallback,
        **(clustering_params or {}),
    )
    groups = summarize_clusters(events)

    truth_info = _load_ground_truth(spec, directory, events)
    truth = truth_info["labels"] if truth_info else None
    evaluation = evaluate_clustering(embeddings, cluster_result.labels, truth)
    evaluation["ground_truth"] = (
        {"available": False} if truth_info is None
        else {"available": truth is not None, "coverage": truth_info["coverage"],
              **({"note": truth_info["note"]} if "note" in truth_info else {})}
    )
    report["stages"]["semantic_clustering"] = {**cluster_result.params, "evaluation": evaluation}

    report["total_runtime_seconds"] = round(time.time() - t0, 3)
    report["n_events_final"] = len(events)

    # 6. Hand-off artifacts --------------------------------------------------
    if save_outputs:
        names = config.ARTIFACT_NAMES
        _write_jsonl(events, out_dir / names["events_jsonl"])
        report["parquet_written"] = _write_parquet(events, out_dir / names["events_parquet"])
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "mode": mode.value, "dataset_id": spec.dataset_id, "is_synthetic": spec.is_synthetic,
                "n_events": len(events),
                "embedding": emb_info.to_dict(),
                "clustering": cluster_result.params,
            },
            "groups": groups,
        }
        with open(out_dir / names["semantic_groups"], "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
        report["output_dir"] = str(out_dir)
        with open(out_dir / names["report"], "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return {"events": events, "embeddings": embeddings, "semantic_groups": groups,
            "report": report, "output_dir": out_dir if save_outputs else None}


def _research_names() -> list[str]:
    from src.preprocessing.datasets import research_dataset_names
    return research_dataset_names()


def _print_report(report: dict) -> None:
    print("\n" + "=" * 64)
    print(f"PIPELINE RUN REPORT  [{report['mode']}]  dataset={report['dataset_id']}")
    if report.get("warning"):
        print(f"!! {report['warning']}")
    print("=" * 64)
    for stage, info in report["stages"].items():
        print(f"\n[{stage}]")
        for k, v in info.items():
            if k == "top_templates":
                print("  top_templates:")
                for tmpl, count in v:
                    print(f"    ({count:>4}x) {tmpl}")
            elif k in ("per_file", "files", "evaluation"):
                print(f"  {k}: {json.dumps(v)[:300]}")
            else:
                print(f"  {k}: {v}")
    print(f"\nTotal runtime: {report['total_runtime_seconds']}s")
    print(f"Final event count: {report['n_events_final']}")
    print("=" * 64)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Log preprocessing + event-intelligence pipeline. Choose --dataset (research) or --demo (synthetic).",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dataset", help="registered research dataset id, e.g. loghub_hdfs_2k")
    mode.add_argument("--demo", action="store_true", help="run on synthetic fixture logs (NOT a research run)")
    ap.add_argument("--embedding-backend", choices=["sentence-transformers", "tfidf"], default=None)
    ap.add_argument("--representation", choices=list(config.EMBEDDING_REPRESENTATIONS), default=None)
    ap.add_argument("--algorithm", choices=list(config.CLUSTERING_ALGORITHMS), default=None)
    ap.add_argument("--distance-threshold", type=float, default=None, help="agglomerative cosine distance threshold")
    ap.add_argument("--n-clusters", type=int, default=None, help="kmeans k")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-intermediate", action="store_true")
    args = ap.parse_args(argv)

    params = {k: v for k, v in {"distance_threshold": args.distance_threshold,
                                "n_clusters": args.n_clusters}.items() if v is not None}
    try:
        result = run_pipeline(
            dataset=args.dataset, demo=args.demo, embedding_backend=args.embedding_backend,
            representation=args.representation, algorithm=args.algorithm,
            clustering_params=params, seed=args.seed, save_intermediate=args.save_intermediate,
        )
    except DatasetMissingError as exc:
        print(f"[pipeline] DATASET MISSING: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[pipeline] INVALID REQUEST: {exc}", file=sys.stderr)
        return 2
    except EmbeddingBackendUnavailable as exc:
        print(f"[pipeline] EMBEDDING BACKEND UNAVAILABLE: {exc}", file=sys.stderr)
        return 3
    _print_report(result["report"])
    print(f"\nOutputs written to: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
