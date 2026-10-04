"""
Central configuration for the preprocessing + event-intelligence pipeline
(Phases 1 & 2 of the project roadmap).

Two run modes exist (see src/pipeline.py):
  * RESEARCH -- reads a registered dataset (src/preprocessing/datasets.py),
                writes to data/processed/<dataset_id>/, fails loudly if
                anything required is missing.
  * DEMO     -- reads synthetic fixture logs, writes to outputs/demo/.
                Never a source of research results.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_LOG_DIR = DATA_DIR / "raw"
SAMPLE_LOG_DIR = DATA_DIR / "sample"          # synthetic fixtures (DEMO only)
PROCESSED_DIR = DATA_DIR / "processed"        # RESEARCH hand-off artifacts
OUTPUT_DIR = ROOT_DIR / "outputs"             # DEMO artifacts + evaluation reports
DOCS_DIR = ROOT_DIR / "docs"

# Hand-off artifact file names (written into the run's output directory).
ARTIFACT_NAMES = {
    "events_jsonl": "events.jsonl",
    "events_parquet": "events.parquet",
    "embeddings": "embeddings.npy",
    "semantic_groups": "semantic_groups.json",
    "report": "pipeline_report.json",
}
# Optional per-stage dumps (only written with save_intermediate=True).
INTERMEDIATE_NAMES = {
    "parsed": "parsed_events.jsonl",
    "normalized": "normalized_events.jsonl",
    "extracted": "extracted_events.jsonl",
}

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_BACKEND = "sentence-transformers"   # "sentence-transformers" | "tfidf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
TFIDF_SVD_COMPONENTS = 64
# What text gets embedded. Deliberately a switch, not an assumption -- the
# comparison is part of the experimental design:
#   "message"  -> normalized_message (variable tokens kept)
#   "template" -> template (variable tokens masked)
#   "hybrid"   -> "<template> | <normalized_message>"
EMBEDDING_REPRESENTATION = "template"
EMBEDDING_REPRESENTATIONS = ("message", "template", "hybrid")

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
CLUSTERING_ALGORITHM = "agglomerative"  # "hdbscan" | "kmeans" | "agglomerative"
CLUSTERING_ALGORITHMS = ("agglomerative", "kmeans", "hdbscan")
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3
KMEANS_N_CLUSTERS = 12
# NOTE: 0.35 is a starting value, not a research assumption. Choose it with
# `python -m src.evaluate_clustering` (see docs/dataset_card.md, "Split").
AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.35

TEMPLATE_MASK_PATTERNS = {
    "IP": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    "NUM": r"\d+",
    "UUID": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
    "HEX": r"\b0x[0-9a-fA-F]+\b",
    "PATH": r"(?:/[\w\-.]+){2,}",
    "EMAIL": r"\b[\w.-]+@[\w.-]+\.\w+\b",
}

SEVERITY_CANONICAL_ORDER = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
SEVERITY_ALIASES = {
    "DEBUG": "DEBUG", "TRACE": "DEBUG",
    "INFO": "INFO", "NOTICE": "INFO",
    "WARN": "WARN", "WARNING": "WARN",
    "ERROR": "ERROR", "ERR": "ERROR",
    "CRITICAL": "CRITICAL", "CRIT": "CRITICAL", "FATAL": "CRITICAL",
    "EMERGENCY": "CRITICAL", "ALERT": "CRITICAL",
}

RANDOM_SEED = 42
