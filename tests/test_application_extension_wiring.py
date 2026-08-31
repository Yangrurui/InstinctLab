"""The application distribution owns and publishes its native extension seams."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from instinctlab.actuators.registration import (
    DELAYED_PD_MODEL_ID,
    register_isaacsim,
    register_mjlab,
)
from instinctlab.actuators.runtime import (
    ISAACSIM_DELAYED_PD_RUNTIME,
    MJLAB_DELAYED_PD_RUNTIME,
    DeclaredModelRuntimeAdapter,
)
from instinctlab.assets.unitree_g1 import isaacsim, mjlab
from instinctlab.terrains.registration import (
    PERLIN_WAVE_KIND,
    ROUGH_TILE_KINDS,
    register_terrains,
)
from instinctlab_engine.actuators import (
    ACTUATOR_CAPABILITIES,
    STIFFNESS,
    ActuatorRegistry,
)
from instinctlab_engine.registry import TerrainExtensionRegistry

ROOT = Path(__file__).resolve().parents[1]
APPLICATION = ROOT / "source" / "instinctlab"


def _application_entry_points() -> dict[str, list[str]]:
    tree = ast.parse((APPLICATION / "setup.py").read_text())
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    value = next(
        keyword.value
        for keyword in setup_call.keywords
        if keyword.arg == "entry_points"
    )
    return ast.literal_eval(value)


def _register_actuator(engine: str, registrar) -> ActuatorRegistry:
    registry = ActuatorRegistry(load_entry_points=False)
    registry._active_engine = engine
    try:
        registrar(registry)
    finally:
        registry._active_engine = None
    return registry


def test_application_wheel_publishes_terrain_and_actuator_registrars() -> None:
    entry_points = _application_entry_points()
    assert entry_points["instinctlab.actuators"] == [
        (
            "isaacsim.instinctlab_delayed_pd = "
            "instinctlab.actuators.registration:register_isaacsim"
        ),
        (
            "mjlab.instinctlab_delayed_pd = "
            "instinctlab.actuators.registration:register_mjlab"
        ),
    ]
    assert entry_points["instinctlab.terrains"] == [
        "instinctlab_rough = instinctlab.terrains.registration:register_terrains"
    ]


def test_application_registrars_are_sdk_free_and_lazy() -> None:
    imported_before = set(sys.modules)

    terrains = TerrainExtensionRegistry(load_entry_points=False)
    register_terrains(terrains)
    isaac_actuators = _register_actuator("isaacsim", register_isaacsim)
    mjlab_actuators = _register_actuator("mjlab", register_mjlab)

    imported = set(sys.modules) - imported_before
    sdk_roots = ("isaaclab", "isaacsim", "mjlab", "mujoco", "omni", "pxr")
    assert not [
        name
        for name in imported
        if name in sdk_roots or name.startswith(tuple(f"{root}." for root in sdk_roots))
    ]
    assert terrains.sub_terrain_kinds("isaacsim") == frozenset(ROUGH_TILE_KINDS)
    assert terrains.sub_terrain_kinds("mjlab") == frozenset(ROUGH_TILE_KINDS)

    isaac = isaac_actuators.registrations("isaacsim")[DELAYED_PD_MODEL_ID]
    mjlab_registration = mjlab_actuators.registrations("mjlab")[DELAYED_PD_MODEL_ID]
    assert isaac.config_factory == "isaaclab.actuators:DelayedPDActuatorCfg"
    assert mjlab_registration.config_factory == "mjlab.actuator:BuiltinPdActuatorCfg"
    assert (
        isaac.capabilities == mjlab_registration.capabilities == ACTUATOR_CAPABILITIES
    )


def test_application_runtime_aliases_do_not_claim_unlabelled_native_models() -> None:
    unlabelled = SimpleNamespace(cfg=SimpleNamespace())
    other = SimpleNamespace(
        cfg=SimpleNamespace(instinctlab_model_id="another.model.v1")
    )
    for adapter in (ISAACSIM_DELAYED_PD_RUNTIME, MJLAB_DELAYED_PD_RUNTIME):
        assert adapter.matches(unlabelled) is False
        assert adapter.matches(other) is False


def test_identity_adapter_can_scope_a_second_application_model(monkeypatch) -> None:
    class RuntimeDelegate:
        @staticmethod
        def matches(actuator) -> bool:
            return getattr(actuator, "native_family", None) == "review_motor"

    module = ModuleType("instinctlab_review_actuator_runtime")
    module.RUNTIME = RuntimeDelegate()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    adapter = DeclaredModelRuntimeAdapter(
        model_id="review.custom_motor.v1",
        delegate_path=f"{module.__name__}:RUNTIME",
    )
    module.APP_ADAPTER = adapter
    selected = SimpleNamespace(
        native_family="review_motor",
        cfg=SimpleNamespace(instinctlab_model_id="review.custom_motor.v1"),
    )
    delayed = SimpleNamespace(
        native_family="review_motor",
        cfg=SimpleNamespace(instinctlab_model_id=DELAYED_PD_MODEL_ID),
    )

    assert adapter.matches(selected) is True
    assert adapter.matches(delayed) is False

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="isaacsim",
        model_id="review.custom_motor.v1",
        config_factory=lambda **values: values,
        runtime_adapter=f"{module.__name__}:APP_ADAPTER",
        capabilities={STIFFNESS},
    )
    registration, selected_adapter = registry.runtime_adapter(
        "isaacsim",
        selected,
        capability=STIFFNESS,
        native_group="review_motor",
        requesting_term="extension conformance",
    )
    assert registration.model_id == "review.custom_motor.v1"
    assert selected_adapter is adapter


def test_perlin_wave_has_one_dedicated_lazy_builder_per_engine() -> None:
    terrains = TerrainExtensionRegistry(load_entry_points=False)
    register_terrains(terrains)

    expected = {
        "isaacsim": "instinctlab_engine_isaacsim.terrain_builders",
        "mjlab": "instinctlab_engine_mjlab.terrain_builders",
    }
    for engine, module_name in expected.items():
        builder = terrains.sub_terrain(engine, PERLIN_WAVE_KIND)
        assert builder is not None
        assert builder.__module__ == module_name
        assert builder.__name__ == "build_perlin_wave_tile"


def test_rough_kinds_are_application_owned_not_backend_registered() -> None:
    for project in ("instinctlab_engine_isaacsim", "instinctlab_engine_mjlab"):
        plugin = (ROOT / "source" / project / "src" / project / "plugin.py").read_text()
        assert not any(kind in plugin for kind in ROUGH_TILE_KINDS)


def test_parkour_native_assets_select_the_application_actuator_model() -> None:
    for module in (isaacsim, mjlab):
        config = module.NATIVE_CONFIGS["popsicle_torsobase_parkour_v1"]
        assert config.actuator_model_ids == (DELAYED_PD_MODEL_ID,)
        assert {group.model_id for group in config.actuator_groups} == {
            DELAYED_PD_MODEL_ID
        }

        for variant in (
            "popsicle_torsobase_v1",
            "popsicle_torsobase_shadowing_v1",
        ):
            assert (
                DELAYED_PD_MODEL_ID
                not in module.NATIVE_CONFIGS[variant].actuator_model_ids
            )


def test_application_registrars_declare_core_api_compatibility() -> None:
    for registrar in (register_isaacsim, register_mjlab, register_terrains):
        assert registrar.instinctlab_engine_api == ">=0.1,<0.2"
