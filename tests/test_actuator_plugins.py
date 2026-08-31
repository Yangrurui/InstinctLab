"""Native actuator plugins stay lazy, atomic, attributable, and fail closed."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from instinctlab_engine import actuators as actuator_module
from instinctlab_engine import plugins as plugin_module
from instinctlab_engine.actuators import (
    EFFORT_LIMITS,
    JOINT_POSITION_COMMAND,
    STIFFNESS,
    ActuatorRegistry,
)
from instinctlab_engine.plugins import (
    PluginDiscoveryError,
    plugin_provenance_since,
    plugin_usage_snapshot,
)


class _Distribution:
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version


class _EntryPoint:
    def __init__(self, name: str, registrar, *, distribution: str):
        self.name = name
        self._registrar = registrar
        self.loads = 0
        self.dist = _Distribution(distribution)
        self.value = f"{distribution}.registration:register"

    def load(self):
        self.loads += 1
        return self._registrar


@pytest.fixture(autouse=True)
def _restore_plugin_provenance():
    snapshot = plugin_module._snapshot_provenance()
    yield
    plugin_module._restore_provenance(snapshot)


def _register_isaac(registry: ActuatorRegistry) -> None:
    registry.register(
        model_id="test.pd.v1",
        config_factory="test_isaac_actuator:Config",
        runtime_adapter="test_isaac_actuator:RUNTIME",
        capabilities={JOINT_POSITION_COMMAND, STIFFNESS},
    )


_register_isaac.instinctlab_engine_api = ">=0.1,<0.2"


def test_metadata_discovery_loads_only_the_selected_engine_and_no_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isaac = _EntryPoint(
        "isaacsim.test_pd", _register_isaac, distribution="test-isaac-actuator"
    )
    mjlab = _EntryPoint(
        "mjlab.test_pd",
        lambda _registry: pytest.fail("unselected actuator provider was loaded"),
        distribution="test-mjlab-actuator",
    )
    monkeypatch.setattr(
        actuator_module.metadata,
        "entry_points",
        lambda *, group: (
            [mjlab, isaac] if group == "instinctlab.actuators" else []
        ),
    )
    before = set(sys.modules)

    registry = ActuatorRegistry()
    metadata_by_model = registry.registrations("isaacsim")

    assert tuple(metadata_by_model) == ("test.pd.v1",)
    assert metadata_by_model["test.pd.v1"].capabilities == frozenset(
        {JOINT_POSITION_COMMAND, STIFFNESS}
    )
    assert isaac.loads == 1
    assert mjlab.loads == 0
    imported = set(sys.modules) - before
    assert "test_isaac_actuator" not in imported
    assert not any(name == "isaaclab" or name.startswith("isaaclab.") for name in imported)
    assert not any(name == "mjlab" or name.startswith("mjlab.") for name in imported)


def test_factory_resolution_is_lazy_and_records_the_used_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ModuleType("test_isaac_actuator")

    class Config:
        pass

    provider.Config = Config
    provider.RUNTIME = object()
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    entry_point = _EntryPoint(
        "isaacsim.test_pd", _register_isaac, distribution="test-isaac-actuator"
    )
    monkeypatch.setattr(
        actuator_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.actuators" else [],
    )
    registry = ActuatorRegistry()
    usage_start = plugin_usage_snapshot()

    assert registry.config_factory("isaacsim", "test.pd.v1") is Config
    provenance = plugin_provenance_since(usage_start, engine="isaacsim")

    assert len(provenance) == 1
    assert provenance[0]["distribution"] == "test-isaac-actuator"
    assert provenance[0]["registered_keys"] == ["isaacsim:test.pd.v1"]


def test_partial_registration_rolls_back_and_reraises_the_same_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register_broken(registry: ActuatorRegistry) -> None:
        registry.register(
            model_id="test.partial.v1",
            config_factory="broken_actuator:Config",
            capabilities={JOINT_POSITION_COMMAND},
        )
        raise RuntimeError("registrar exploded")

    entry_point = _EntryPoint(
        "isaacsim.broken", register_broken, distribution="broken-actuators"
    )
    monkeypatch.setattr(
        actuator_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.actuators" else [],
    )
    registry = ActuatorRegistry()

    with pytest.raises(PluginDiscoveryError, match="broken-actuators") as first:
        registry.registrations("isaacsim")
    assert registry._registrations == {}
    assert registry._origins == {}

    with pytest.raises(PluginDiscoveryError) as second:
        registry.registrations("isaacsim")
    assert second.value is first.value
    assert entry_point.loads == 1


def test_duplicate_models_name_both_distributions_and_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _EntryPoint(
        "mjlab.alpha", _register_isaac, distribution="first-actuators"
    )
    second = _EntryPoint(
        "mjlab.beta", _register_isaac, distribution="second-actuators"
    )
    monkeypatch.setattr(
        actuator_module.metadata,
        "entry_points",
        lambda *, group: [first, second] if group == "instinctlab.actuators" else [],
    )
    registry = ActuatorRegistry()

    with pytest.raises(PluginDiscoveryError) as error:
        registry.registrations("mjlab")

    assert "first-actuators" in str(error.value)
    assert "second-actuators" in str(error.value)
    assert registry._registrations == {}


def test_wrong_core_api_is_rejected_without_partial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def future(registry: ActuatorRegistry) -> None:
        _register_isaac(registry)

    future.instinctlab_engine_api = ">=9,<10"
    entry_point = _EntryPoint(
        "isaacsim.future", future, distribution="future-actuators"
    )
    monkeypatch.setattr(
        actuator_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.actuators" else [],
    )
    registry = ActuatorRegistry()

    with pytest.raises(PluginDiscoveryError, match=">=9,<10"):
        registry.registrations("isaacsim")
    assert registry._registrations == {}


def test_runtime_capability_is_checked_against_the_matched_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ModuleType("test_runtime_actuator")

    class NativeActuator:
        pass

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

        def stiffness_groups(self, _env: object, _asset: object, _actuator: object):
            return (((0,), 1.0),)

    provider.Config = type("Config", (), {})
    provider.RUNTIME = RuntimeAdapter()
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.runtime.v1",
        config_factory="test_runtime_actuator:Config",
        runtime_adapter="test_runtime_actuator:RUNTIME",
        capabilities={JOINT_POSITION_COMMAND, STIFFNESS},
    )

    registration, adapter = registry.runtime_adapter(
        "mjlab",
        NativeActuator(),
        capability=STIFFNESS,
        native_group="legs",
        requesting_term="joint_power",
    )
    assert registration.model_id == "test.runtime.v1"
    assert isinstance(adapter, RuntimeAdapter)

    with pytest.raises(RuntimeError, match="effort_limits") as error:
        registry.runtime_adapter(
            "mjlab",
            NativeActuator(),
            capability=EFFORT_LIMITS,
            native_group="legs",
            requesting_term="joint_torque_limit",
        )
    assert "test.runtime.v1" in str(error.value)
    assert "joint_torque_limit" in str(error.value)


def test_runtime_capability_requires_its_declared_adapter_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ModuleType("test_incomplete_runtime_actuator")

    class NativeActuator:
        instinctlab_model_id = "test.incomplete.v1"

    class RuntimeAdapter:
        def matches(self, actuator: object) -> bool:
            return isinstance(actuator, NativeActuator)

    provider.Config = type("Config", (), {})
    provider.RUNTIME = RuntimeAdapter()
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="test.incomplete.v1",
        config_factory="test_incomplete_runtime_actuator:Config",
        runtime_adapter="test_incomplete_runtime_actuator:RUNTIME",
        capabilities={EFFORT_LIMITS},
    )

    with pytest.raises(RuntimeError, match="effort_limit_for_joint"):
        registry.runtime_adapter(
            "mjlab",
            NativeActuator(),
            capability=EFFORT_LIMITS,
            native_group="joint",
            requesting_term="joint effort limit reader",
        )
