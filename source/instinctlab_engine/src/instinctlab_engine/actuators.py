"""Lazy native actuator models and their observable runtime capabilities.

Actuator physics remains owned by the selected simulator and concrete asset.
This registry carries only a model identity, lazy implementation paths, and the
small runtime interface that shared task formulas are allowed to request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any, Protocol, runtime_checkable

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

ENTRY_POINT_GROUP = "instinctlab.actuators"

JOINT_POSITION_COMMAND = "joint_position_command"
APPLIED_EFFORT = "applied_effort"
EFFORT_LIMITS = "effort_limits"
STIFFNESS = "stiffness"
GAIN_RANDOMIZATION = "gain_randomization"
STATEFUL_RESET = "stateful_reset"

ACTUATOR_CAPABILITIES = frozenset(
    {
        JOINT_POSITION_COMMAND,
        APPLIED_EFFORT,
        EFFORT_LIMITS,
        STIFFNESS,
        GAIN_RANDOMIZATION,
        STATEFUL_RESET,
    }
)

LazyObject = str | Any


@runtime_checkable
class ActuatorRuntimeAdapter(Protocol):
    """Optional observable interface for one native actuator model."""

    def matches(self, actuator: object) -> bool: ...


@dataclass(frozen=True, slots=True)
class ActuatorRegistration:
    """SDK-free metadata for one engine-native actuator model."""

    engine: str
    model_id: str
    config_factory: LazyObject
    runtime_adapter: LazyObject | None
    capabilities: frozenset[str]


def _validate_lazy_object(value: LazyObject, *, field: str) -> None:
    if callable(value):
        return
    if not isinstance(value, str):
        raise TypeError(f"{field} must be callable or a 'module:attribute' path")
    module, separator, attribute = value.partition(":")
    if not separator or not module or not attribute.isidentifier():
        raise ValueError(
            f"invalid {field} {value!r}; expected a callable or 'module:attribute'"
        )


def _resolve(value: LazyObject) -> Any:
    if not isinstance(value, str):
        return value
    module_name, _, attribute = value.partition(":")
    return getattr(import_module(module_name), attribute)


class ActuatorRegistry:
    """Discover actuator providers for only the selected engine.

    Registrars receive this object and call :meth:`register` without an engine
    argument. The engine is bound by the ``<engine>.<extension>`` entry-point
    name, which prevents a provider from registering into another backend's
    namespace during discovery.
    """

    ENTRY_POINT_GROUP = ENTRY_POINT_GROUP

    def __init__(self, *, load_entry_points: bool = True):
        self._registrations: dict[tuple[str, str], ActuatorRegistration] = {}
        self._origins: dict[tuple[str, str], str] = {}
        self._load_entry_points = load_entry_points
        self._loaded_engines: set[str] = set()
        self._engine_errors: dict[str, PluginDiscoveryError] = {}
        self._active_engine: str | None = None
        self._active_plugin: str | None = None

    @staticmethod
    def _validate_engine(engine: str) -> None:
        if not engine or not engine.isidentifier():
            raise ValueError(f"invalid actuator engine name {engine!r}")

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        if not model_id or any(not part.isidentifier() for part in model_id.split(".")):
            raise ValueError(
                f"invalid actuator model id {model_id!r}; expected dotted identifiers"
            )

    @_plugin_locked
    def register(
        self,
        *,
        model_id: str,
        config_factory: LazyObject,
        runtime_adapter: LazyObject | None = None,
        capabilities: Iterable[str] = (),
        engine: str | None = None,
    ) -> None:
        """Register lazy native implementation metadata, never parameter values."""
        selected_engine = engine or self._active_engine
        if selected_engine is None:
            raise ValueError(
                "direct actuator registration must state engine=; plugin registrars "
                "receive it from their entry-point name"
            )
        self._validate_engine(selected_engine)
        if self._active_engine is not None and selected_engine != self._active_engine:
            raise ValueError(
                f"actuator plugin for {self._active_engine!r} tried to register "
                f"model {model_id!r} for {selected_engine!r}"
            )
        self._validate_model_id(model_id)
        _validate_lazy_object(config_factory, field="actuator config_factory")
        if runtime_adapter is not None:
            _validate_lazy_object(runtime_adapter, field="actuator runtime_adapter")
        capability_set = frozenset(capabilities)
        unknown = capability_set - ACTUATOR_CAPABILITIES
        if unknown:
            raise ValueError(
                f"actuator {selected_engine}/{model_id} declares unknown capabilities: "
                f"{', '.join(sorted(unknown))}"
            )
        if runtime_adapter is None and capability_set.intersection(
            {APPLIED_EFFORT, EFFORT_LIMITS, STIFFNESS, GAIN_RANDOMIZATION, STATEFUL_RESET}
        ):
            raise ValueError(
                f"actuator {selected_engine}/{model_id} declares runtime capabilities "
                "without a runtime_adapter"
            )

        key = (selected_engine, model_id)
        registration = ActuatorRegistration(
            engine=selected_engine,
            model_id=model_id,
            config_factory=config_factory,
            runtime_adapter=runtime_adapter,
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
                f"actuator model {selected_engine}/{model_id} is already registered by "
                f"{existing_source}; conflicting registration is from {incoming_source}"
            )
        self._registrations[key] = registration
        if existing is None and self._active_plugin is not None:
            self._origins[key] = self._active_plugin

    @_plugin_locked
    def _load_installed(self, engine: str) -> None:
        self._validate_engine(engine)
        if engine in self._engine_errors:
            raise self._engine_errors[engine]
        if engine in self._loaded_engines:
            return
        if not self._load_entry_points:
            self._loaded_engines.add(engine)
            return

        registration_snapshot = dict(self._registrations)
        origin_snapshot = dict(self._origins)
        provenance_snapshot = _snapshot_provenance()
        try:
            entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
            selected = []
            for entry_point in entry_points:
                entry_engine, separator, extension = entry_point.name.partition(".")
                if not separator or not extension:
                    raise PluginDiscoveryError(
                        "Actuator entry-point name must be '<engine>.<extension>' "
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
                        f"Actuator plugin registration failed ({description}): {exc}"
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
                        "Actuator plugin did not register a model for its selected engine "
                        f"({description})"
                    )
                for key in registered:
                    self._origins[key] = description
                record_plugin(
                    self.ENTRY_POINT_GROUP,
                    entry_point,
                    (f"{item_engine}:{model_id}" for item_engine, model_id in registered),
                )
        except Exception as exc:  # noqa: BLE001 - discovery is one atomic transaction
            self._registrations = registration_snapshot
            self._origins = origin_snapshot
            self._active_engine = None
            self._active_plugin = None
            _restore_provenance(provenance_snapshot)
            error = (
                exc
                if isinstance(exc, PluginDiscoveryError)
                else PluginDiscoveryError(
                    f"Actuator plugin discovery failed for {engine!r}: {exc}"
                )
            )
            self._engine_errors[engine] = error
            raise error
        self._loaded_engines.add(engine)

    def registrations(self, engine: str) -> Mapping[str, ActuatorRegistration]:
        """Return SDK-free metadata for the selected engine."""
        with _PLUGIN_LOCK:
            self._load_installed(engine)
            return {
                model_id: registration
                for (item_engine, model_id), registration in sorted(
                    self._registrations.items()
                )
                if item_engine == engine
            }

    def registration(self, engine: str, model_id: str) -> ActuatorRegistration:
        with _PLUGIN_LOCK:
            self._load_installed(engine)
            try:
                return self._registrations[(engine, model_id)]
            except KeyError:
                known = ", ".join(self.registrations(engine)) or "none"
                raise KeyError(
                    f"unknown actuator model {engine}/{model_id}; registered models are {known}"
                ) from None

    def config_factory(self, engine: str, model_id: str) -> Callable[..., Any]:
        registration = self.registration(engine, model_id)
        factory = _resolve(registration.config_factory)
        if not callable(factory):
            raise TypeError(
                f"actuator config_factory for {engine}/{model_id} resolved to "
                f"non-callable {type(factory).__name__}"
            )
        mark_plugin_used(self.ENTRY_POINT_GROUP, f"{engine}:{model_id}")
        return factory

    def runtime_adapter(
        self,
        engine: str,
        actuator: object,
        *,
        capability: str,
        native_group: str,
        requesting_term: str,
    ) -> tuple[ActuatorRegistration, ActuatorRuntimeAdapter]:
        if capability not in ACTUATOR_CAPABILITIES:
            raise ValueError(f"unknown actuator runtime capability {capability!r}")
        matches: list[tuple[ActuatorRegistration, ActuatorRuntimeAdapter]] = []
        for registration in self.registrations(engine).values():
            if registration.runtime_adapter is None:
                continue
            adapter = _resolve(registration.runtime_adapter)
            if isinstance(adapter, type):
                adapter = adapter()
            if not isinstance(adapter, ActuatorRuntimeAdapter):
                raise TypeError(
                    f"runtime adapter for {engine}/{registration.model_id} does not "
                    "provide matches(actuator)"
                )
            if adapter.matches(actuator):
                matches.append((registration, adapter))
        if not matches:
            raise RuntimeError(
                f"Engine {engine!r} has no registered runtime adapter for native actuator "
                f"group {native_group!r} ({type(actuator).__name__}); term "
                f"{requesting_term!r} requires {capability!r}."
            )
        if len(matches) > 1:
            model_ids = ", ".join(item[0].model_id for item in matches)
            raise RuntimeError(
                f"Engine {engine!r} native actuator group {native_group!r} matches "
                f"multiple model ids: {model_ids}"
            )
        registration, adapter = matches[0]
        if capability not in registration.capabilities:
            raise RuntimeError(
                f"Engine {engine!r} actuator model {registration.model_id!r}, native "
                f"group {native_group!r}, does not declare capability {capability!r} "
                f"required by term {requesting_term!r}."
            )
        mark_plugin_used(
            self.ENTRY_POINT_GROUP, f"{engine}:{registration.model_id}"
        )
        return registration, adapter


ACTUATORS = ActuatorRegistry()


def register_actuator(
    engine: str,
    model_id: str,
    config_factory: LazyObject,
    *,
    runtime_adapter: LazyObject | None = None,
    capabilities: Iterable[str] = (),
) -> None:
    """Directly register one native actuator model without importing its SDK."""
    ACTUATORS.register(
        engine=engine,
        model_id=model_id,
        config_factory=config_factory,
        runtime_adapter=runtime_adapter,
        capabilities=capabilities,
    )


def actuator_models(engine: str) -> Mapping[str, ActuatorRegistration]:
    """Return installed SDK-free actuator metadata for one engine."""
    return ACTUATORS.registrations(engine)


def native_actuator_factory(engine: str, model_id: str) -> Callable[..., Any]:
    """Resolve a selected native config factory after backend bootstrap."""
    return ACTUATORS.config_factory(engine, model_id)


__all__ = [
    "ACTUATOR_CAPABILITIES",
    "ACTUATORS",
    "APPLIED_EFFORT",
    "ActuatorRegistration",
    "ActuatorRegistry",
    "ActuatorRuntimeAdapter",
    "EFFORT_LIMITS",
    "GAIN_RANDOMIZATION",
    "JOINT_POSITION_COMMAND",
    "STATEFUL_RESET",
    "STIFFNESS",
    "actuator_models",
    "native_actuator_factory",
    "register_actuator",
]
