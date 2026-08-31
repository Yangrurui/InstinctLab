"""Stable lifecycle benchmark report and release-threshold evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class BenchmarkThresholdError(RuntimeError):
    """A benchmark report does not satisfy its declared release thresholds."""


def duration_statistics(samples_s: Sequence[float]) -> dict[str, float | int]:
    """Summarize positive duration samples in milliseconds."""
    samples = sorted(float(value) for value in samples_s)
    if not samples:
        raise ValueError("A benchmark series must contain at least one sample.")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("Benchmark duration samples must be finite and non-negative.")
    return {
        "samples": len(samples),
        "min_ms": samples[0] * 1_000.0,
        "median_ms": _percentile(samples, 0.5) * 1_000.0,
        "p95_ms": _percentile(samples, 0.95) * 1_000.0,
        "max_ms": samples[-1] * 1_000.0,
        "mean_ms": (sum(samples) / len(samples)) * 1_000.0,
    }


def evaluate_thresholds(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return fail-closed threshold violations for one benchmark report.

    Threshold documents may contain ``match`` metadata plus ``minimum`` and
    ``maximum`` metric mappings. Unknown metrics and mismatched metadata are
    failures rather than silently ignored baseline drift.
    """
    allowed = {"schema_version", "match", "minimum", "maximum"}
    unknown_sections = sorted(set(thresholds) - allowed)
    failures: list[str] = []
    if unknown_sections:
        failures.append(f"unknown threshold document sections: {unknown_sections}")
    match = thresholds.get("match", {})
    if not isinstance(match, Mapping):
        failures.append("threshold match section must be a mapping")
    else:
        for name, expected in match.items():
            actual = report.get(name)
            if actual != expected:
                failures.append(
                    f"metadata {name!r} is {actual!r}, expected {expected!r}"
                )
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        failures.append("benchmark report has no metrics mapping")
        return tuple(failures)
    for direction in ("minimum", "maximum"):
        limits = thresholds.get(direction, {})
        if not isinstance(limits, Mapping):
            failures.append(f"threshold {direction} section must be a mapping")
            continue
        for name, raw_limit in limits.items():
            if name not in metrics:
                failures.append(f"threshold names unknown metric {name!r}")
                continue
            actual = metrics[name]
            if not _is_finite_number(actual) or not _is_finite_number(raw_limit):
                failures.append(f"metric {name!r} and its limit must be finite numbers")
                continue
            limit = float(raw_limit)
            value = float(actual)
            if direction == "minimum" and value < limit:
                failures.append(f"metric {name!r}={value} is below minimum {limit}")
            if direction == "maximum" and value > limit:
                failures.append(f"metric {name!r}={value} exceeds maximum {limit}")
    return tuple(failures)


def require_thresholds(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> None:
    """Raise one readable error when release thresholds are not met."""
    failures = evaluate_thresholds(report, thresholds)
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise BenchmarkThresholdError(f"Lifecycle benchmark thresholds failed:\n{details}")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = [
    "BenchmarkThresholdError",
    "duration_statistics",
    "evaluate_thresholds",
    "require_thresholds",
]
