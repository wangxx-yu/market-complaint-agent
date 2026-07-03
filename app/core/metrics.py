"""Prometheus-compatible metrics for Agent observability.

Exposes counters and histograms trackable via /metrics endpoint.
No external dependency — plain text format compliant with Prometheus exposition.

Usage:
    from app.core.metrics import (
        agent_calls_total, agent_errors_total, agent_latency_seconds,
        reviews_total, degradation_total,
    )
    agent_calls_total.labels(agent="classifier", status="success").inc()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import ClassVar


@dataclass
class Counter:
    name: str
    help: str
    labels: dict[str, int] = field(default_factory=dict)

    def inc(self, **label_values: str) -> None:
        key = ",".join(f"{k}={v}" for k, v in sorted(label_values.items()))
        self.labels[key] = self.labels.get(key, 0) + 1


@dataclass
class Histogram:
    name: str
    help: str
    buckets: ClassVar[list[float]] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    _values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self._values.append(value)


# ── Metric definitions ──

agent_calls_total = Counter("mca_agent_calls_total", "Total agent invocations")
agent_errors_total = Counter("mca_agent_errors_total", "Agent invocation errors")
agent_latency_seconds = Histogram("mca_agent_latency_seconds", "Agent step latency")
reviews_total = Counter("mca_reviews_total", "Human review confirmations")
degradation_total = Counter("mca_degradation_total", "Degradation events")
