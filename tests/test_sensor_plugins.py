"""Native sensor extensions are lazy and obey explicit lifecycle semantics."""

from __future__ import annotations

from collections import deque
import sys
from types import ModuleType, SimpleNamespace

import pytest

from instinctlab_engine import plugins as plugin_module
from instinctlab_engine import sensors as sensor_module
from instinctlab_engine.plugins import (
    PluginDiscoveryError,
    plugin_provenance_since,
    plugin_usage_snapshot,
)
from instinctlab_engine.sensors import (
    ATTACHED_FRAME,
    DEVICE_PLACEMENT,
    LATENCY_HISTORY,
    PARTIAL_RESET,
    SAMPLE_TIMESTAMP,
    NativeSensorBuildContext,
    SensorRegistry,
)
from instinctlab_engine.spec import NativeSensorRef


ALL_CAPABILITIES = {
    ATTACHED_FRAME,
    DEVICE_PLACEMENT,
    LATENCY_HISTORY,
    PARTIAL_RESET,
    SAMPLE_TIMESTAMP,
}


class _Distribution:
    def __init__(self, name: str):
        self.name = name
        self.version = "1.0"


class _EntryPoint:
    def __init__(self, name: str, registrar, distribution: str):
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


def _register_imu(registry: SensorRegistry) -> None:
    registry.register(
        kind="imu",
        builder="external_imu.native:build_sensor",
        capabilities=ALL_CAPABILITIES,
    )


_register_imu.instinctlab_engine_api = ">=0.1,<0.2"


def _entry_points(monkeypatch: pytest.MonkeyPatch, entries) -> None:
    monkeypatch.setattr(
        sensor_module.metadata,
        "entry_points",
        lambda *, group: list(entries) if group == "instinctlab.sensors" else [],
    )


def test_sensor_metadata_loads_only_for_the_selected_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isaac = _EntryPoint("isaacsim.imu", _register_imu, "external-isaac-imu")
    mjlab = _EntryPoint(
        "mjlab.imu",
        lambda _registry: pytest.fail("unselected sensor provider was loaded"),
        "external-mjlab-imu",
    )
    _entry_points(monkeypatch, (mjlab, isaac))

    registrations = SensorRegistry().registrations("isaacsim")

    assert tuple(registrations) == ("imu",)
    assert registrations["imu"].capabilities == ALL_CAPABILITIES
    assert isaac.loads == 1
    assert mjlab.loads == 0


def test_resolved_sensor_builder_records_only_its_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = ModuleType("external_imu.native")
    native.build_sensor = lambda sensor, context: (sensor, context)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    entry = _EntryPoint("mjlab.imu", _register_imu, "external-mjlab-imu")
    _entry_points(monkeypatch, (entry,))
    registry = SensorRegistry()
    usage_start = plugin_usage_snapshot()

    builder = registry.builder(
        "mjlab",
        NativeSensorRef(name="imu", kind="imu", attach="pelvis"),
    )

    assert builder is native.build_sensor
    provenance = plugin_provenance_since(usage_start, engine="mjlab")
    assert [record["distribution"] for record in provenance] == [
        "external-mjlab-imu"
    ]
    assert provenance[0]["registered_keys"] == ["mjlab:imu"]


def test_partial_sensor_registration_rolls_back_and_failure_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(registry: SensorRegistry) -> None:
        _register_imu(registry)
        raise RuntimeError("registrar exploded")

    entry = _EntryPoint("mjlab.broken", broken, "broken-sensors")
    _entry_points(monkeypatch, (entry,))
    registry = SensorRegistry()

    with pytest.raises(PluginDiscoveryError, match="broken-sensors") as first:
        registry.registrations("mjlab")
    assert registry._registrations == {}
    with pytest.raises(PluginDiscoveryError) as second:
        registry.registrations("mjlab")
    assert second.value is first.value
    assert entry.loads == 1


def test_duplicate_sensor_kinds_name_both_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _EntryPoint("isaacsim.alpha", _register_imu, "first-sensors")
    second = _EntryPoint("isaacsim.beta", _register_imu, "second-sensors")
    _entry_points(monkeypatch, (first, second))

    with pytest.raises(PluginDiscoveryError) as error:
        SensorRegistry().registrations("isaacsim")
    assert "first-sensors" in str(error.value)
    assert "second-sensors" in str(error.value)


def test_wrong_sensor_plugin_api_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def future(registry: SensorRegistry) -> None:
        _register_imu(registry)

    future.instinctlab_engine_api = ">=9,<10"
    _entry_points(
        monkeypatch,
        (_EntryPoint("mjlab.future", future, "future-sensors"),),
    )

    with pytest.raises(PluginDiscoveryError, match=">=9,<10"):
        SensorRegistry().registrations("mjlab")


def test_missing_sensor_lifecycle_capability_fails_before_builder_import() -> None:
    registry = SensorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        kind="imu",
        builder="module_that_must_not_import:build",
        capabilities={ATTACHED_FRAME, DEVICE_PLACEMENT, SAMPLE_TIMESTAMP},
    )
    sensor = NativeSensorRef(
        name="imu",
        kind="imu",
        attach="pelvis",
        latency=0.01,
        history_length=2,
    )

    with pytest.raises(RuntimeError, match="latency_history") as error:
        registry.builder("mjlab", sensor)
    assert "partial_reset" in str(error.value)


class _FixtureImu:
    """External stateful fixture with one-tick latency and selected-env reset."""

    def __init__(self, num_envs: int, history_length: int):
        self.history_length = history_length
        self.pending = deque()
        self.data = [[0.0] for _ in range(num_envs)]
        self.sample_time = [0.0] * num_envs
        self.history = [[] for _ in range(num_envs)]

    def update(self, values: list[float], sample_time: float) -> None:
        self.pending.append((list(values), sample_time))
        if len(self.pending) < 2:
            return
        delivered, timestamp = self.pending.popleft()
        for env_id, value in enumerate(delivered):
            self.data[env_id] = [value]
            self.sample_time[env_id] = timestamp
            self.history[env_id] = (
                self.history[env_id] + [value]
            )[-self.history_length :]

    def reset(self, env_ids: list[int]) -> None:
        for env_id in env_ids:
            self.data[env_id] = [0.0]
            self.sample_time[env_id] = 0.0
            self.history[env_id] = []


@pytest.mark.parametrize("engine", ("isaacsim", "mjlab"))
def test_external_stateful_imu_temporal_and_partial_reset_contract(engine: str) -> None:
    registry = SensorRegistry(load_entry_points=False)

    def build(sensor: NativeSensorRef, context: NativeSensorBuildContext):
        assert context.engine == engine
        return _FixtureImu(context.num_envs, sensor.history_length)

    registry.register(
        engine=engine,
        kind="imu",
        builder=build,
        capabilities=ALL_CAPABILITIES,
    )
    sensor = NativeSensorRef(
        name="imu",
        kind="imu",
        attach="pelvis",
        update_period=0.01,
        latency=0.01,
        history_length=2,
    )
    builder = registry.builder(engine, sensor)
    runtime = builder(
        sensor,
        NativeSensorBuildContext(
            engine=engine,
            robot=SimpleNamespace(),
            sensor_period=0.005,
            profile={},
            num_envs=2,
        ),
    )

    runtime.update([1.0, 10.0], 0.01)
    assert runtime.data == [[0.0], [0.0]]
    runtime.update([2.0, 20.0], 0.02)
    assert runtime.data == [[1.0], [10.0]]
    assert runtime.sample_time == [0.01, 0.01]
    runtime.update([3.0, 30.0], 0.03)
    assert runtime.history == [[1.0, 2.0], [10.0, 20.0]]

    runtime.reset([0])
    assert runtime.history[0] == []
    assert runtime.history[1] == [10.0, 20.0]
