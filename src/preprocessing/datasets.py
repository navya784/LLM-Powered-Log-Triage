"""
Dataset registry (separates RESEARCH datasets from DEMO fixtures).

A `DatasetSpec` says where a dataset's logs live, how to attribute a service
to sources whose lines carry none, and (optionally) where its ground-truth
labels are. Every real dataset used in experiments must be registered here
AND documented in docs/dataset_card.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from src import config


class DatasetMissingError(RuntimeError):
    """No dataset was supplied, or its files are not on disk."""


@dataclass
class DatasetSpec:
    dataset_id: str
    name: str
    directory: Callable[[], Path] | Path
    file_patterns: tuple = ("*.log", "*.txt", "*.json", "*.jsonl")
    # source file name -> service, for sources whose raw lines carry no service
    # (e.g. web access logs). Explicit, per-dataset metadata -- never a parser default.
    service_map: Dict[str, str] = field(default_factory=dict)
    default_service: Optional[str] = None
    # ground truth: labels for template/event grouping, keyed (source_file, line_number)
    ground_truth_loader: Optional[Callable[[Path], Dict[tuple, str]]] = None
    is_synthetic: bool = False
    fetch_hint: str = ""

    def resolve_dir(self) -> Path:
        return self.directory() if callable(self.directory) else Path(self.directory)

    def service_for(self, source_file: str) -> Optional[str]:
        """Service the dataset configuration attributes to `source_file`
        (a path relative to the dataset directory), or None if unknown."""
        if source_file in self.service_map:
            return self.service_map[source_file]
        return self.service_map.get(Path(source_file).name, self.default_service)


# ---------------------------------------------------------------------------
# LogHub HDFS (2k sample) -- ground truth = EventId of each line
# ---------------------------------------------------------------------------

def _loghub_hdfs_ground_truth(directory: Path) -> Dict[tuple, str]:
    csv_path = directory / "HDFS_2k.log_structured.csv"
    if not csv_path.exists():
        raise DatasetMissingError(f"Ground-truth file not found: {csv_path}")
    labels: Dict[tuple, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[("HDFS_2k.log", int(row["LineId"]))] = row["EventId"]
    return labels


LOGHUB_HDFS_2K = DatasetSpec(
    dataset_id="loghub_hdfs_2k",
    name="LogHub HDFS (2k sample)",
    directory=lambda: config.RAW_LOG_DIR / "loghub" / "HDFS_2k",
    file_patterns=("HDFS_2k.log",),
    # HDFS lines carry no service field. The whole dataset is one system, so
    # the dataset config attributes every line to "hdfs" (recorded in
    # metadata["service_source"] = "dataset_config"). The finer-grained
    # component (e.g. dfs.DataNode$PacketResponder) is kept in metadata.
    default_service="hdfs",
    ground_truth_loader=_loghub_hdfs_ground_truth,
    fetch_hint="python scripts/fetch_loghub.py",
)

# Synthetic fixtures: tests / CI / demo only. Not a research dataset.
DEMO_SYNTHETIC = DatasetSpec(
    dataset_id="demo_synthetic",
    name="Synthetic demo logs (NOT a research dataset)",
    directory=lambda: config.SAMPLE_LOG_DIR,
    # The synthetic access log is defined to come from the gateway; that is
    # dataset metadata, stated here rather than baked into the Apache parser.
    service_map={"gateway_access.log": "gateway"},
    is_synthetic=True,
)

DATASETS: Dict[str, DatasetSpec] = {
    LOGHUB_HDFS_2K.dataset_id: LOGHUB_HDFS_2K,
    DEMO_SYNTHETIC.dataset_id: DEMO_SYNTHETIC,
}


def research_dataset_names() -> list[str]:
    return sorted(k for k, v in DATASETS.items() if not v.is_synthetic)


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise DatasetMissingError(
            f"Unknown dataset '{name}'. Registered research datasets: "
            f"{research_dataset_names()}. Use --demo for synthetic fixture data."
        )
    return DATASETS[name]
