"""SDK-free entry-point registration for every public extension seam."""

from __future__ import annotations

from importlib import import_module


def resolve_asset(engine: str, variant: str):
    if variant != "v1":
        raise KeyError(f"fixture_bot has no variant {variant!r}")
    if engine not in {"isaacsim", "mjlab"}:
        raise KeyError(f"fixture_bot has no native module for {engine!r}")
    return import_module(f"instinctlab_extension_fixture.{engine}_asset"), variant


resolve_asset.instinctlab_engine_api = ">=0.1,<0.2"


def _register_actuator(registry, engine: str) -> None:
    registry.register(
        model_id="fixture.stateful.v1",
        config_factory=(
            f"instinctlab_extension_fixture.{engine}_actuator:StatefulActuatorCfg"
        ),
        runtime_adapter="instinctlab_extension_fixture.runtime:RUNTIME_ADAPTER",
        capabilities={
            "joint_position_command",
            "applied_effort",
            "effort_limits",
            "stiffness",
            "gain_randomization",
            "stateful_reset",
        },
    )


def register_isaacsim_actuator(registry) -> None:
    _register_actuator(registry, "isaacsim")


def register_mjlab_actuator(registry) -> None:
    _register_actuator(registry, "mjlab")


def _register_sensor(registry, engine: str) -> None:
    registry.register(
        kind="fixture.imu",
        builder=f"instinctlab_extension_fixture.{engine}_implementation:build_imu",
        capabilities={
            "attached_frame",
            "sample_timestamp",
            "latency_history",
            "device_placement",
            "partial_reset",
        },
    )


def register_isaacsim_sensor(registry) -> None:
    _register_sensor(registry, "isaacsim")


def register_mjlab_sensor(registry) -> None:
    _register_sensor(registry, "mjlab")


def register_terrains(registry) -> None:
    registry.register_terrain(
        "isaacsim",
        "fixture_plane",
        "instinctlab_extension_fixture.isaacsim_implementation:build_terrain",
    )
    registry.register_terrain(
        "mjlab",
        "fixture_plane",
        "instinctlab_extension_fixture.mjlab_implementation:build_terrain",
    )


for _registrar in (
    register_isaacsim_actuator,
    register_mjlab_actuator,
    register_isaacsim_sensor,
    register_mjlab_sensor,
    register_terrains,
):
    _registrar.instinctlab_engine_api = ">=0.1,<0.2"
