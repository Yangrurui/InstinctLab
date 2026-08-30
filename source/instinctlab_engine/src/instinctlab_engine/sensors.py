"""Lazy native sensor builders for additive third-party sensor families."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any

from instinctlab_engine.plugins import (
    PluginDiscoveryError,
    _PLUGIN_LOCK,
    _plugin_locked,
    _restore_provenance,
    _snapshot_provenance,
    entry_point_description,
    load_plugin_callable,
    mark_plugin_used,
    record_plugin,
)

ENTRY_POINT_GROUP = "instinctlab.sensors"

ATTACHED_FRAME = "attached_frame"
SAMPLE_TIMESTAMP = "sample_timestamp"
LATENCY_HISTORY = "latency_history"
DEVICE_PLACEMENT = "device_placement"
PARTIAL_RESET = "partial_reset"
SENSOR_CAPABILITIES = frozenset(
    {
        ATTACHED_FRAME,
        SAMPLE_TIMESTAMP,
        LATENCY_HISTORY,
        DEVICE_PLACEMENT,
        PARTIAL_RESET,
    }
)


@dataclass(frozen=True, slots=True)
class NativeSensorBuildContext:
    """Backend-selected context passed to a lazy native sensor builder."""

    engine: str
    robot: Any
    sensor_period: float
    profile: Mapping[str, Any]
    num_envs: int


@dataclass(frozen=True, slots=True)
class SensorRegistration:
    engine: str
    kind: str
    builder: str | Callable[..., Any]
    capabilities: frozenset[str]


def _validate_builder(builder: str | Callable[..., Any]) -> None:
    if callable(builder):
        return
    if not isinstance(builder, str):
        raise TypeError("native sensor builder must be callable or a 'module:attribute' path")
    module, separator, attribute = builder.partition(":")
    if not separator or not module or not attribute.isidentifier():
        raise ValueError(
            f"invalid native sensor builder {builder!r}; expected 'module:attribute'"
        )


def _resolve(builder: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(builder):
        return builder
    module_name, _, attribute = builder.partition(":")
    resolved = getattr(import_module(module_name), attribute)
    if not callable(resolved):
        raise TypeError(f"native sensor builder {builder!r} resolved to a non-callable")
    return resolved


class SensorRegistry:
    """Transactional SDK-free discovery keyed by ``(engine, sensor kind)``."""

    ENTRY_POINT_GROUP = ENTRY_POINT_GROUP

    def __init__(self, *, load_entry_points: bool = True):
        self._registrations: dict[tuple[str, str], SensorRegistration] = {}
        self._origins: dict[tuple[str, str], str] = {}
        self._load_entry_points = load_entry_points
        self._loaded_engines: set[str] = set()
        self._engine_errors: dict[str, PluginDiscoveryError] = {}
        self._active_engine: str | None = None
        self._active_plugin: str | None = None

    @staticmethod
    def _validate_key(engine: str, kind: str) -> None:
        if not engine or not engine.isidentifier():
            raise ValueError(f"invalid sensor engine {engine!r}")
        if not kind or any(not part.isidentifier() for part in kind.split(".")):
            raise ValueError(
                f"invalid sensor kind {kind!r}; expected dotted identifiers"
            )

    @_plugin_locked
    def register(
        self,
        *,
        kind: str,
        builder: str | Callable[..., Any],
        capabilities: Iterable[str],
        engine: str | None = None,
    ) -> None:
        selected_engine = engine or self._active_engine
        if selected_engine is None:
            raise ValueError(
                "direct sensor registration must state engine=; plugin registrars "
                "receive it from their entry-point name"
            )
        self._validate_key(selected_engine, kind)
        if self._active_engine is not None and selected_engine != self._active_engine:
            raise ValueError(
                f"sensor plugin for {self._active_engine!r} tried to register "
                f"{kind!r} for {selected_engine!r}"
            )
        _validate_builder(builder)
        capability_set = frozenset(capabilities)
        unknown = capability_set - SENSOR_CAPABILITIES
        if unknown:
            raise ValueError(
                f"sensor {selected_engine}/{kind} declares unknown capabilities: "
                f"{', '.join(sorted(unknown))}"
            )
        key = (selected_engine, kind)
        registration = SensorRegistration(
            engine=selected_engine,
            kind=kind,
            builder=builder,
            capabilities=capability_set,
        )
        existing = self._registrations.get(key)
        if existing is not None and (
            existing != registration
            or (
                self._active_plugin is not None
                and self._origins.get(key) != self._active_plugin
            )
        ):
            existing_source = self._origins.get(key, "a direct registration")
            incoming_source = self._active_plugin or "a direct registration"
            raise ValueError(
                f"sensor {selected_engine}/{kind} is already registered by "
                f"{existing_source}; conflicting registration is from {incoming_source}"
            )
        self._registrations[key] = registration
        if existing is None and self._active_plugin is not None:
            self._origins[key] = self._active_plugin

    @_plugin_locked
    def _load_installed(self, engine: str) -> None:
        if engine in self._engine_errors:
            raise self._engine_errors[engine]
        if engine in self._loaded_engines:
            return
        if not engine or not engine.isidentifier():
            raise ValueError(f"invalid sensor engine {engine!r}")
        if not self._load_entry_points:
            self._loaded_engines.add(engine)
            return
        registration_snapshot = dict(self._registrations)
        origin_snapshot = dict(self._origins)
        provenance_snapshot = _snapshot_provenance()
        try:
            installed = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
            selected = []
            for entry_point in installed:
                entry_engine, separator, extension = entry_point.name.partition(".")
                if not separator or not extension:
                    raise PluginDiscoveryError(
                        "Sensor entry-point name must be '<engine>.<extension>' "
                        f"({entry_point_description(self.ENTRY_POINT_GROUP, entry_point)})"
                    )
                if entry_engine == engine:
                    selected.append(entry_point)
            for entry_point in sorted(selected, key=lambda item: item.name):
                before = set(self._registrations)
                description = entry_point_description(
                    self.ENTRY_POINT_GROUP, entry_point
                )
                try:
                    self._active_engine = engine
                    self._active_plugin = description
                    registrar = load_plugin_callable(
                        self.ENTRY_POINT_GROUP, entry_point
                    )
                    registrar(self)
                except Exception as exc:
                    raise PluginDiscoveryError(
                        f"Sensor plugin registration failed ({description}): {exc}"
                    ) from exc
                finally:
                    self._active_engine = None
                    self._active_plugin = None
                registered = {
                    key
                    for key in set(self._registrations) - before
                    if key[0] == engine
                }
                if not registered:
                    raise PluginDiscoveryError(
                        "Sensor plugin did not register a kind for its selected engine "
                        f"({description})"
                    )
                for key in registered:
                    self._origins[key] = description
                record_plugin(
                    self.ENTRY_POINT_GROUP,
                    entry_point,
                    (f"{item_engine}:{kind}" for item_engine, kind in registered),
                )
        except Exception as exc:  # noqa: BLE001 - discovery must be atomic
            self._registrations = registration_snapshot
            self._origins = origin_snapshot
            self._active_engine = None
            self._active_plugin = None
            _restore_provenance(provenance_snapshot)
            error = (
                exc
                if isinstance(exc, PluginDiscoveryError)
                else PluginDiscoveryError(
                    f"Sensor plugin discovery failed for {engine!r}: {exc}"
                )
            )
            self._engine_errors[engine] = error
            raise error
        self._loaded_engines.add(engine)

    def registrations(self, engine: str) -> Mapping[str, SensorRegistration]:
        with _PLUGIN_LOCK:
            self._load_installed(engine)
            return {
                kind: registration
                for (item_engine, kind), registration in sorted(
                    self._registrations.items()
                )
                if item_engine == engine
            }

    def builder(self, engine: str, sensor: Any) -> Callable[..., Any]:
        with _PLUGIN_LOCK:
            self._load_installed(engine)
            try:
                registration = self._registrations[(engine, sensor.kind)]
            except KeyError:
                known = ", ".join(self.registrations(engine)) or "none"
                raise RuntimeError(
                    f"Engine {engine!r} has no native sensor builder for kind "
                    f"{sensor.kind!r}; registered kinds are {known}."
                ) from None
        required = {ATTACHED_FRAME, SAMPLE_TIMESTAMP, DEVICE_PLACEMENT}
        if sensor.latency > 0.0 or sensor.history_length > 0:
            required.add(LATENCY_HISTORY)
        if sensor.partial_reset:
            required.add(PARTIAL_RESET)
        missing = required - registration.capabilities
        if missing:
            raise RuntimeError(
                f"Engine {engine!r} sensor kind {sensor.kind!r} cannot satisfy sensor "
                f"{sensor.name!r}; missing capabilities: {', '.join(sorted(missing))}."
            )
        mark_plugin_used(self.ENTRY_POINT_GROUP, f"{engine}:{sensor.kind}")
        return _resolve(registration.builder)


SENSORS = SensorRegistry()


def register_sensor(
    engine: str,
    kind: str,
    builder: str | Callable[..., Any],
    *,
    capabilities: Iterable[str],
) -> None:
    SENSORS.register(
        engine=engine,
        kind=kind,
        builder=builder,
        capabilities=capabilities,
    )


def native_sensor_builder(engine: str, sensor: Any) -> Callable[..., Any]:
    return SENSORS.builder(engine, sensor)


__all__ = [
    "ATTACHED_FRAME",
    "DEVICE_PLACEMENT",
    "LATENCY_HISTORY",
    "NativeSensorBuildContext",
    "PARTIAL_RESET",
    "SAMPLE_TIMESTAMP",
    "SENSORS",
    "SENSOR_CAPABILITIES",
    "SensorRegistration",
    "SensorRegistry",
    "native_sensor_builder",
    "register_sensor",
]
