"""
Synthetic multi-format log generator -- TESTS, CI, DEMO AND LOCAL DEV ONLY.

This data exists to exercise the parsers and check pipeline connectivity.
It is NOT a research dataset: nothing generated here may be used to claim
parsing, clustering or RCA performance. The pipeline only reads it when run
explicitly in DEMO mode (`python -m src.pipeline --demo`).

Timestamps within each generated file are strictly increasing (each line is
1-8 seconds after the previous one).
"""

from __future__ import annotations

import json as _json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from src import config


_SERVICES = ["auth-service", "payment-service", "order-service", "inventory-service", "gateway"]
_HOSTS = ["10.0.0.11", "10.0.0.12", "10.0.0.13", "10.0.0.21", "10.0.0.22"]

_BRACKET_TEMPLATES = [
    ("INFO", "user logged in successfully"),
    ("WARN", "connection pool approaching capacity"),
    ("ERROR", "Connection failed"),
    ("ERROR", "Database connection timeout"),
    ("CRITICAL", "out of memory, restarting worker"),
]

_SYSLOG_TEMPLATES = [
    ("INFO", "{service}[{pid}]: request completed in {ms}ms status={status}"),
    ("WARN", "{service}[{pid}]: connection pool at {pct}% capacity"),
    ("ERROR", "{service}[{pid}]: failed to connect to {dep} at {ip}: timeout after {ms}ms"),
    ("ERROR", "{service}[{pid}]: database query failed: deadlock detected on table orders"),
    ("CRITICAL", "{service}[{pid}]: out of memory, restarting worker"),
    ("INFO", "{service}[{pid}]: health check passed"),
]

_APACHE_TEMPLATE = '{ip} - - [{ts}] "GET /api/v1/{resource} HTTP/1.1" {status} {bytes}'

_JSON_TEMPLATES = [
    {"level": "INFO", "msg": "user {user_id} logged in", "service": "auth-service"},
    {"level": "ERROR", "msg": "payment declined for order {order_id}: insufficient funds", "service": "payment-service"},
    {"level": "WARN", "msg": "retrying inventory lookup for sku {sku}", "service": "inventory-service"},
    {"level": "ERROR", "msg": "upstream gateway returned 502 for {ip}", "service": "gateway"},
]

_KV_TEMPLATE = 'ts={ts} level={level} service={service} host={ip} msg="{msg}"'


def _advance(current: datetime) -> datetime:
    """Move time forward by 1-8 seconds, so timestamps are strictly increasing."""
    return current + timedelta(seconds=random.randint(1, 8))


def generate_sample_dataset(out_dir: Path | None = None, n_per_format: int = 60, seed: int = 42) -> List[Path]:
    random.seed(seed)
    out_dir = out_dir or config.SAMPLE_LOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2026, 9, 20, 8, 0, 0)
    written: List[Path] = []

    # bracket format (matches the existing log_preprocessor's expected shape)
    bracket_path = out_dir / "app_bracket.log"
    with open(bracket_path, "w") as f:
        current_time = base_time
        for _ in range(n_per_format):
            level, msg = random.choice(_BRACKET_TEMPLATES)
            current_time = _advance(current_time)
            ts = current_time
            service = random.choice(_SERVICES)
            f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {service}: {msg}\n")
    written.append(bracket_path)

    # syslog-style
    syslog_path = out_dir / "app_syslog.log"
    with open(syslog_path, "w") as f:
        current_time = base_time
        for _ in range(n_per_format):
            level, tmpl = random.choice(_SYSLOG_TEMPLATES)
            current_time = _advance(current_time)
            ts = current_time
            line = tmpl.format(
                service=random.choice(_SERVICES), pid=random.randint(1000, 9999),
                ms=random.randint(5, 4000), status=random.choice([200, 200, 200, 500, 503]),
                pct=random.randint(60, 99), dep=random.choice(_SERVICES), ip=random.choice(_HOSTS),
            )
            f.write(f"{ts.strftime('%b %d %H:%M:%S')} {random.choice(_HOSTS)} {level} {line}\n")
    written.append(syslog_path)

    # Apache-access-style
    apache_path = out_dir / "gateway_access.log"
    with open(apache_path, "w") as f:
        current_time = base_time
        for _ in range(n_per_format):
            current_time = _advance(current_time)
            ts = current_time
            line = _APACHE_TEMPLATE.format(
                ip=random.choice(_HOSTS), ts=ts.strftime("%d/%b/%Y:%H:%M:%S +0000"),
                resource=random.choice(["orders", "users", "inventory", "payments"]),
                status=random.choice([200, 201, 400, 404, 500, 502]), bytes=random.randint(120, 15000),
            )
            f.write(line + "\n")
    written.append(apache_path)

    # JSON-lines
    json_path = out_dir / "services.jsonl"
    with open(json_path, "w") as f:
        current_time = base_time
        for _ in range(n_per_format):
            current_time = _advance(current_time)
            ts = current_time
            rec = dict(random.choice(_JSON_TEMPLATES))
            rec["msg"] = rec["msg"].format(
                user_id=random.randint(1000, 9999), order_id=f"ORD-{random.randint(10000,99999)}",
                sku=f"SKU-{random.randint(100,999)}", ip=random.choice(_HOSTS),
            )
            rec["timestamp"] = ts.isoformat() + "Z"
            rec["host"] = random.choice(_HOSTS)
            f.write(_json.dumps(rec) + "\n")
    written.append(json_path)

    # key=value
    kv_path = out_dir / "infra_metrics.log"
    with open(kv_path, "w") as f:
        current_time = base_time
        for _ in range(n_per_format):
            current_time = _advance(current_time)
            ts = current_time
            level, tmpl = random.choice(_SYSLOG_TEMPLATES)
            msg = tmpl.format(
                service=random.choice(_SERVICES), pid=random.randint(1000, 9999),
                ms=random.randint(5, 4000), status=random.choice([200, 500]),
                pct=random.randint(60, 99), dep=random.choice(_SERVICES), ip=random.choice(_HOSTS),
            )
            f.write(_KV_TEMPLATE.format(
                ts=ts.isoformat(), level=level, service=random.choice(_SERVICES),
                ip=random.choice(_HOSTS), msg=msg,
            ) + "\n")
    written.append(kv_path)

    return written


if __name__ == "__main__":
    for p in generate_sample_dataset():
        print(f"wrote {p}")
