"""
Writes docs/logevent.schema.json from the LogEvent dataclass.

    python scripts/export_schema.py          # (re)generate the file
    python scripts/export_schema.py --check  # exit 1 if the file is out of date

A test runs the --check logic, so the published schema cannot drift from the code.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.schema import logevent_json_schema  # noqa: E402

TARGET = ROOT / "docs" / "logevent.schema.json"


def render() -> str:
    return json.dumps(logevent_json_schema(), indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        ok = TARGET.exists() and TARGET.read_text(encoding="utf-8") == text
        print("up to date" if ok else f"{TARGET} is out of date; run python scripts/export_schema.py")
        return 0 if ok else 1
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
