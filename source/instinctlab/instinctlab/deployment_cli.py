"""Command-line interface for single-file ONNX policy verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .deployment import (
    DEPLOYMENT_REPORT_SCHEMA,
    DeploymentVerificationError,
    verify_deployment_policy,
    write_verification_report,
)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify policy.onnx before it is promoted to a robot runtime."
    )
    parser.add_argument("policy", type=Path, help="Path to the single-file policy.onnx artifact.")
    parser.add_argument(
        "--runtime",
        choices=("onnxruntime", "reference", "auto"),
        default="onnxruntime",
        help="Inference implementation. Release evidence must use onnxruntime (default).",
    )
    parser.add_argument("--atol", type=float, default=None, help="Override the embedded absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=None, help="Override the embedded relative tolerance.")
    parser.add_argument("--sha256", default=None, help="Expected release SHA-256 for policy.onnx.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed inference calls before measurement.")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed inference calls.")
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=None,
        help="Fail if ONNX Runtime p95 latency exceeds this target-hardware limit.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. No sidecar is written by default.",
    )
    return parser.parse_args(argv)


def _failure(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": DEPLOYMENT_REPORT_SCHEMA,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    report_path = args.report.expanduser().resolve() if args.report is not None else None
    try:
        report = verify_deployment_policy(
            args.policy,
            runtime=args.runtime,
            expected_sha256=args.sha256,
            atol=args.atol,
            rtol=args.rtol,
            warmup=args.warmup,
            runs=args.runs,
            max_p95_latency_ms=args.max_p95_latency_ms,
        )
    except (DeploymentVerificationError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        report = _failure(error)
        if report_path is not None:
            write_verification_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    if report_path is not None:
        write_verification_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
