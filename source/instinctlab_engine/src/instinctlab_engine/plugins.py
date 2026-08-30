"""Shared plugin discovery errors, API checks, and run provenance."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import wraps
from threading import RLock, local
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

ENGINE_CORE_API_VERSION = "0.1"


class PluginDiscoveryError(RuntimeError):
    """An installed plugin failed before its registrations could be committed."""


_records: dict[str, dict[str, Any]] = {}
_key_origins: dict[tuple[str, str], str] = {}
_PLUGIN_LOCK = RLock()
_usage_state = local()


def _usage_log() -> list[str]:
    log = getattr(_usage_state, "log", None)
    if log is None:
        log = []
        _usage_state.log = log
    return log


def _plugin_locked(function):
    """Serialize discovery transactions and their shared provenance state."""

    @wraps(function)
    def locked(*args, **kwargs):
        with _PLUGIN_LOCK:
            return function(*args, **kwargs)

    return locked


def _distribution_details(entry_point: Any) -> tuple[str, str]:
    distribution = getattr(entry_point, "dist", None)
    name = getattr(distribution, "name", None)
    if not name and distribution is not None:
        metadata = getattr(distribution, "metadata", {})
        name = metadata.get("Name") if hasattr(metadata, "get") else None
    version = getattr(distribution, "version", None)
    return str(name or "<unknown distribution>"), str(version or "<unknown version>")


def entry_point_description(group: str, entry_point: Any) -> str:
    distribution, version = _distribution_details(entry_point)
    value = getattr(entry_point, "value", "<unknown value>")
    return (
        f"group={group!r}, entry_point={entry_point.name!r}, value={value!r}, "
        f"distribution={distribution!r}, version={version!r}"
    )


def load_plugin_callable(group: str, entry_point: Any) -> Any:
    """Load one entry point and validate its optional core-API constraint."""
    try:
        plugin = entry_point.load()
        if not callable(plugin):
            raise TypeError("entry point did not resolve to a callable")
        api_constraint = getattr(plugin, "instinctlab_engine_api", None)
        if api_constraint is not None:
            try:
                supported = SpecifierSet(str(api_constraint))
            except InvalidSpecifier as exc:
                raise ValueError(
                    f"invalid instinctlab_engine_api constraint {api_constraint!r}"
                ) from exc
            if Version(ENGINE_CORE_API_VERSION) not in supported:
                raise RuntimeError(
                    f"plugin requires engine-core API {api_constraint!r}, "
                    f"but this process provides {ENGINE_CORE_API_VERSION!r}"
                )
        return plugin
    except Exception as exc:
        raise PluginDiscoveryError(
            f"Plugin discovery failed ({entry_point_description(group, entry_point)}): {exc}"
        ) from exc


@_plugin_locked
def record_plugin(group: str, entry_point: Any, keys: Iterable[str]) -> str:
    """Record keys attributed to an entry point and return its stable record ID."""
    distribution, version = _distribution_details(entry_point)
    value = str(getattr(entry_point, "value", "<unknown value>"))
    record_id = f"{group}|{distribution}|{version}|{entry_point.name}|{value}"
    normalized_keys = tuple(sorted(set(keys)))
    _records[record_id] = {
        "group": group,
        "distribution": distribution,
        "version": version,
        "entry_point": entry_point.name,
        "value": value,
        "registered_keys": list(normalized_keys),
    }
    for key in normalized_keys:
        _key_origins[(group, key)] = record_id
    return record_id


@_plugin_locked
def mark_plugin_used(group: str, key: str) -> None:
    """Mark the plugin that owns ``group/key`` as used by the current thread."""
    record_id = _key_origins.get((group, key))
    if record_id is not None:
        _usage_log().append(record_id)


def _provenance(
    record_ids: set[str], *, engine: str | None = None
) -> list[dict[str, Any]]:

    def affects_engine(record: dict[str, Any]) -> bool:
        if engine is None or record["group"] == "instinctlab.assets":
            return True
        keys = record["registered_keys"]
        if record["group"] == "instinctlab.engines":
            return engine in keys
        if record["group"] == "instinctlab.engine_terms":
            return any(key.startswith(f"{engine}:") for key in keys)
        if record["group"] == "instinctlab.actuators":
            return any(key.startswith(f"{engine}:") for key in keys)
        if record["group"] == "instinctlab.terrains":
            return any(f":{engine}:" in key for key in keys)
        return True

    return [
        deepcopy(_records[record_id])
        for record_id in sorted(record_ids)
        if affects_engine(_records[record_id])
    ]


@_plugin_locked
def plugin_provenance(*, engine: str | None = None) -> list[dict[str, Any]]:
    """Metadata for plugins used by the current compilation thread."""
    return _provenance(set(_usage_log()), engine=engine)


@_plugin_locked
def plugin_usage_snapshot() -> int:
    """Return a thread-local cursor used to scope one compilation's provenance."""
    return len(_usage_log())


@_plugin_locked
def plugin_provenance_since(
    usage_start: int,
    *,
    engine: str | None = None,
    include_keys: Iterable[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Metadata for uses after ``usage_start`` plus explicitly selected keys."""
    record_ids = set(_usage_log()[usage_start:])
    record_ids.update(
        record_id
        for group, key in include_keys
        if (record_id := _key_origins.get((group, key))) is not None
    )
    return _provenance(record_ids, engine=engine)


@_plugin_locked
def _snapshot_provenance() -> tuple[dict, dict, list]:
    return dict(_records), dict(_key_origins), list(_usage_log())


@_plugin_locked
def _restore_provenance(snapshot: tuple[dict, dict, list]) -> None:
    records, origins, used = snapshot
    _records.clear()
    _records.update(records)
    _key_origins.clear()
    _key_origins.update(origins)
    log = _usage_log()
    log.clear()
    log.extend(used)


__all__ = [
    "ENGINE_CORE_API_VERSION",
    "PluginDiscoveryError",
    "entry_point_description",
    "load_plugin_callable",
    "mark_plugin_used",
    "plugin_provenance",
    "plugin_provenance_since",
    "plugin_usage_snapshot",
    "record_plugin",
]
