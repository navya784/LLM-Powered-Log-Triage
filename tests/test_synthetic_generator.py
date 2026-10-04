import json
import re
from datetime import datetime

from src.preprocessing.synthetic_generator import generate_sample_dataset


def test_sample_dataset_generation(tmp_path):
    files = generate_sample_dataset(out_dir=tmp_path, n_per_format=10)
    assert len(files) == 5  # bracket, syslog, apache, json, kv
    for f in files:
        assert f.exists() and f.stat().st_size > 0


def test_generation_is_deterministic_for_a_seed(tmp_path):
    a = generate_sample_dataset(out_dir=tmp_path / "a", n_per_format=20, seed=7)
    b = generate_sample_dataset(out_dir=tmp_path / "b", n_per_format=20, seed=7)
    for fa, fb in zip(a, b):
        assert fa.read_text() == fb.read_text()


def _timestamps(path):
    text = path.read_text().splitlines()
    name = path.name
    out = []
    for line in text:
        if name == "app_bracket.log":
            out.append(datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        elif name == "app_syslog.log":
            out.append(datetime.strptime("2026 " + line[:15], "%Y %b %d %H:%M:%S"))
        elif name == "gateway_access.log":
            out.append(datetime.strptime(re.search(r"\[([^\]]+) \+0000\]", line).group(1), "%d/%b/%Y:%H:%M:%S"))
        elif name == "services.jsonl":
            out.append(datetime.fromisoformat(json.loads(line)["timestamp"].rstrip("Z")))
        else:  # infra_metrics.log
            out.append(datetime.fromisoformat(re.search(r"ts=(\S+)", line).group(1)))
    return out


def test_timestamps_are_strictly_increasing_in_every_file(tmp_path):
    for seed in (1, 2, 3):
        for f in generate_sample_dataset(out_dir=tmp_path / str(seed), n_per_format=200, seed=seed):
            ts = _timestamps(f)
            assert len(ts) == 200
            assert all(later > earlier for earlier, later in zip(ts, ts[1:])), f"{f.name} not chronological"
