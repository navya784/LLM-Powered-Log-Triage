"""
Clustering evaluation harness (Phase 2).

Answers "which configuration works best on this dataset?" with numbers
instead of assumptions. For a registered research dataset it sweeps
embedding representation (message / template / hybrid) x clustering
configuration and records, per row:

  embedding backend/model/dimension/representation,
  algorithm + its parameters + seed,
  number of clusters, noise %, cluster-size distribution,
  silhouette (cosine), Davies-Bouldin,
  ARI / NMI against the dataset's ground truth (when it has one).

Protocol (see docs/dataset_card.md, "Train/test split"): the pipeline is
unsupervised, but hyperparameters are still chosen by looking at metrics.
So choose them with `--split tune` (first half of the events in log order)
and report the chosen configuration on `--split heldout` (second half).
`--split all` is for exploration only.

    python -m src.evaluate_clustering --dataset loghub_hdfs_2k --split tune
    python -m src.evaluate_clustering --dataset loghub_hdfs_2k --split heldout

Uses the real embedding backend by default and fails if it is unavailable;
pass --embedding-backend tfidf to run the TF-IDF baseline *explicitly*.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src import config
from src.event_intelligence.cluster_evaluation import evaluate_clustering
from src.event_intelligence.clustering import run_clustering
from src.event_intelligence.embeddings import EmbeddingBackendUnavailable, embed_events
from src.event_intelligence.event_extractor import extract_events
from src.pipeline import _load_ground_truth
from src.preprocessing.datasets import DatasetMissingError, get_dataset
from src.preprocessing.ingestion import ingest_dataset
from src.preprocessing.normalizer import normalize_events

DEFAULT_GRID = (
    [{"algorithm": "agglomerative", "distance_threshold": t} for t in (0.10, 0.20, 0.35, 0.50)]
    + [{"algorithm": "kmeans", "n_clusters": k} for k in (8, 12, 16)]
    + [{"algorithm": "hdbscan", "min_cluster_size": m, "min_samples": 3} for m in (5, 10)]
)


def select_split(items: list, split: str) -> list:
    """Deterministic split by position in log order: 'tune' = first half,
    'heldout' = second half, 'all' = everything."""
    if split == "all":
        return list(items)
    mid = len(items) // 2
    if split == "tune":
        return list(items[:mid])
    if split == "heldout":
        return list(items[mid:])
    raise ValueError(f"Unknown split '{split}'. Choose tune, heldout or all.")


def run_evaluation(
    dataset: str,
    split: str = "tune",
    embedding_backend: Optional[str] = None,
    representations: Optional[list[str]] = None,
    grid: Optional[list[dict]] = None,
    seed: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> dict:
    spec = get_dataset(dataset)
    if spec.is_synthetic:
        raise ValueError("Clustering evaluation is for research datasets, not the synthetic demo data.")
    seed = seed if seed is not None else config.RANDOM_SEED
    directory = data_dir or spec.resolve_dir()

    events, parse_stats, _ = ingest_dataset(spec, directory)
    events = extract_events(normalize_events(events))
    truth_info = _load_ground_truth(spec, directory, events)
    truth_all = truth_info["labels"] if truth_info else None

    idx = select_split(list(range(len(events))), split)
    subset = [events[i] for i in idx]
    truth = [truth_all[i] for i in idx] if truth_all else None

    rows = []
    for rep in representations or list(config.EMBEDDING_REPRESENTATIONS):
        embeddings, info = embed_events(
            subset, representation=rep, backend=embedding_backend, allow_fallback=False,
        )
        for cfg in grid or DEFAULT_GRID:
            cfg = dict(cfg)
            result = run_clustering(embeddings, cfg.pop("algorithm"), seed=seed, **cfg)
            rows.append({
                "embedding": info.to_dict(),
                "clustering": result.params,
                "evaluation": evaluate_clustering(embeddings, result.labels, truth),
            })
    return {
        "dataset_id": spec.dataset_id, "split": split, "n_events": len(subset),
        "seed": seed, "ground_truth_available": truth is not None,
        "parse_stats": {k: parse_stats.to_dict()[k] for k in ("lines_read", "parsed", "parse_failure_rate")},
        "rows": rows,
    }


def _print_table(report: dict) -> None:
    print(f"\nDataset {report['dataset_id']} | split={report['split']} | n={report['n_events']} | seed={report['seed']}")
    hdr = f"{'repr':9} {'algorithm':14} {'params':30} {'k':>4} {'noise%':>7} {'silh':>7} {'DB':>7} {'ARI':>7} {'NMI':>7}"
    print(hdr + "\n" + "-" * len(hdr))
    fmt = lambda v: f"{v:7.3f}" if isinstance(v, (int, float)) else f"{'-':>7}"
    for r in report["rows"]:
        c, e = r["clustering"], r["evaluation"]
        params = {k: c[k] for k in ("distance_threshold", "n_clusters_requested", "min_cluster_size") if k in c}
        ext = e["external"] or {}
        print(f"{r['embedding']['representation']:9} {c['algorithm']:14} {json.dumps(params):30} "
              f"{c['n_clusters']:>4} {100 * e['size_distribution']['noise_fraction']:7.1f} "
              f"{fmt(e['internal']['silhouette_cosine'])} {fmt(e['internal']['davies_bouldin'])} "
              f"{fmt(ext.get('ari'))} {fmt(ext.get('nmi'))}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.evaluate_clustering", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["tune", "heldout", "all"], default="tune")
    ap.add_argument("--embedding-backend", choices=["sentence-transformers", "tfidf"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="where to write the JSON report")
    args = ap.parse_args(argv)
    try:
        report = run_evaluation(args.dataset, args.split, args.embedding_backend, seed=args.seed)
    except (DatasetMissingError, ValueError) as exc:
        print(f"[evaluate] {exc}", file=sys.stderr)
        return 2
    except EmbeddingBackendUnavailable as exc:
        print(f"[evaluate] EMBEDDING BACKEND UNAVAILABLE: {exc}", file=sys.stderr)
        return 3
    _print_table(report)
    out = args.out or (config.OUTPUT_DIR / "evaluation" / f"{args.dataset}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
