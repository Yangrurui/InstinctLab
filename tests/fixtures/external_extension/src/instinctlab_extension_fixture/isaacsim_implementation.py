"""Lazy Isaac-side fixture implementations; deliberately SDK-free for the kit."""

from __future__ import annotations

from .runtime import ImuRuntime, RuntimeAdapter, StatefulActuatorCfgBase


class StatefulActuatorCfg(StatefulActuatorCfgBase):
    pass


RUNTIME_ADAPTER = RuntimeAdapter()


def build_imu(sensor, context):
    return ImuRuntime(sensor, context)


def build_terrain(spec, profile):
    return {"engine": "isaacsim", "kind": spec.kind, "profile": dict(profile)}
