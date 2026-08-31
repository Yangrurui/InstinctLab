from __future__ import annotations

import pytest
from instinctlab_engine.lifecycle import (
    BenchmarkThresholdError,
    duration_statistics,
    evaluate_thresholds,
    require_thresholds,
)


def _report():
    return {
        "engine": "mjlab",
        "task_id": "Task-v0",
        "num_envs": 128,
        "metrics": {
            "throughput_env_steps_per_s": 50_000.0,
            "full_reset_median_ms": 3.0,
        },
    }


def test_duration_statistics_are_stable_and_use_linear_p95() -> None:
    stats = duration_statistics((0.001, 0.002, 0.003, 0.004, 0.005))
    assert stats == {
        "samples": 5,
        "min_ms": 1.0,
        "median_ms": 3.0,
        "p95_ms": pytest.approx(4.8),
        "max_ms": 5.0,
        "mean_ms": 3.0,
    }


def test_thresholds_check_direction_metadata_and_unknown_metrics() -> None:
    failures = evaluate_thresholds(
        _report(),
        {
            "match": {"engine": "isaacsim", "num_envs": 128},
            "minimum": {
                "throughput_env_steps_per_s": 60_000.0,
                "missing": 1.0,
            },
            "maximum": {"full_reset_median_ms": 2.0},
        },
    )
    assert len(failures) == 4
    assert any("metadata 'engine'" in failure for failure in failures)
    assert any("unknown metric 'missing'" in failure for failure in failures)


def test_require_thresholds_is_a_release_gate() -> None:
    passing = {
        "match": {"engine": "mjlab", "task_id": "Task-v0"},
        "minimum": {"throughput_env_steps_per_s": 40_000.0},
        "maximum": {"full_reset_median_ms": 4.0},
    }
    require_thresholds(_report(), passing)

    with pytest.raises(BenchmarkThresholdError, match="thresholds failed"):
        require_thresholds(
            _report(),
            {"maximum": {"full_reset_median_ms": 1.0}},
        )
