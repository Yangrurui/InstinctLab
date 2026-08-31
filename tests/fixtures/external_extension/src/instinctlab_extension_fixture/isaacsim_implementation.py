"""SDK-free Isaac sensor and terrain fixture implementations."""

from __future__ import annotations

from .runtime import ImuRuntime


def build_imu(sensor, context):
    return ImuRuntime(sensor, context)


def build_terrain(spec, profile):
    return {"engine": "isaacsim", "kind": spec.kind, "profile": dict(profile)}
