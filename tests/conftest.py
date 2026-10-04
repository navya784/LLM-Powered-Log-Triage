"""
Test bootstrap.

* Puts the repo root on sys.path so `import src...` works from any cwd.
* If the project's existing `src/preprocessing/log_preprocessor.py` (the
  bracket-format parser this add-on builds on) is not present, installs a
  minimal stand-in so the add-on's tests can run on their own. When the real
  module exists it is used and this stand-in is never created.
"""

import importlib
import re
import sys
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    importlib.import_module("src.preprocessing.log_preprocessor")
except ImportError:
    _BRACKET = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] ([\w\-.]+): (.*)$")

    def extract_log_fields(line):
        m = _BRACKET.match(line)
        if not m:
            return None
        return {
            "timestamp": datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
            "level": m.group(2), "service": m.group(3), "message": m.group(4),
        }

    stub = types.ModuleType("src.preprocessing.log_preprocessor")
    stub.extract_log_fields = extract_log_fields
    sys.modules["src.preprocessing.log_preprocessor"] = stub
