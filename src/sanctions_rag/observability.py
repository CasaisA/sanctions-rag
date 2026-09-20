"""Structured JSON logs, request ids and per-stage latency.

A retrieval service fails quietly: results get worse, nothing raises. So every query
is logged with its stage timings and result count, and the process keeps running
counters that /metrics exposes in Prometheus text format.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    return logging.getLogger("sanctions_rag")


log = logging.getLogger("sanctions_rag")


class Metrics:
    """Counters and latency histograms, exported in Prometheus text format."""

    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

    def __init__(self) -> None:
        self.counters: dict[tuple[str, str], int] = defaultdict(int)
        self.hist: dict[str, list[float]] = defaultdict(list)

    def count(self, name: str, label: str = "") -> None:
        self.counters[(name, label)] += 1

    def observe(self, name: str, seconds: float) -> None:
        self.hist[name].append(seconds)

    def render(self) -> str:
        out: list[str] = []
        for (name, label), value in sorted(self.counters.items()):
            tag = f'{{stage="{label}"}}' if label else ""
            out.append(f"sanctions_rag_{name}_total{tag} {value}")
        for name, values in sorted(self.hist.items()):
            if not values:
                continue
            out.append(f"sanctions_rag_{name}_seconds_count {len(values)}")
            out.append(f"sanctions_rag_{name}_seconds_sum {sum(values):.6f}")
            for b in self.BUCKETS:
                out.append(f'sanctions_rag_{name}_seconds_bucket{{le="{b}"}} '
                           f"{sum(1 for v in values if v <= b)}")
            out.append(f'sanctions_rag_{name}_seconds_bucket{{le="+Inf"}} {len(values)}')
        return "\n".join(out) + "\n"


METRICS = Metrics()


@contextmanager
def stage(name: str, **fields):
    """Time one pipeline stage and log it with the request id."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        METRICS.observe(name, dt)
        METRICS.count("stage", name)
        log.info(f"{name} finished", extra={"extra_fields": {"stage": name,
                                                             "duration_ms": round(dt * 1000, 2), **fields}})


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id.set(rid)
    return rid
