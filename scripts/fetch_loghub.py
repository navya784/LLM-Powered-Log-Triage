"""
Fetches the LogHub HDFS_2k sample into data/raw/loghub/HDFS_2k/ and verifies
each file against the SHA-256 recorded in docs/dataset_card.md.

    python scripts/fetch_loghub.py            # download missing files
    python scripts/fetch_loghub.py --force    # re-download everything

Source: https://github.com/logpai/loghub  (HDFS/HDFS_2k.log*, branch master)
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/"
FILES = {
    "HDFS_2k.log": "7c967000980c086ed55fa6544ba4f05fe66d44622795e890c68caf8bbb635035",
    "HDFS_2k.log_structured.csv": "729df59774e3dde934044028546d2a55d5e3d4370b9d12fcebbe4c087b2bf7b4",
    "HDFS_2k.log_templates.csv": "a07307511f67c9dc1f41ae730ae60dcce8360f2c72742f0b8a3a9cf1a403d1db",
}
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "loghub" / "HDFS_2k"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, expected in FILES.items():
        path = args.dest / name
        if path.exists() and not args.force and sha256(path) == expected:
            print(f"ok       {name} (already present, checksum matches)")
            continue
        print(f"fetching {name} ...")
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as resp:
                path.write_bytes(resp.read())
        except OSError as exc:
            print(f"FAILED   {name}: {exc}", file=sys.stderr)
            ok = False
            continue
        actual = sha256(path)
        if actual != expected:
            print(f"CHECKSUM MISMATCH {name}: expected {expected}, got {actual}. "
                  f"Upstream may have changed; do not use this file for reported results "
                  f"until docs/dataset_card.md is updated.", file=sys.stderr)
            ok = False
        else:
            print(f"ok       {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
