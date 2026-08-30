"""Report the independent dimensions that make a task portable or native."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .task import TaskSpec


def _keys_by_engine(values: Mapping[str, Any]) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for engine, overrides in sorted(values.items()):
        if isinstance(overrides, Mapping):
            keys[engine] = sorted(str(key) for key in overrides)
        else:
            keys[engine] = []
    return keys


def _semantic_overlays(spec: TaskSpec) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []

    def record(path: str, values: Mapping[str, Any]) -> None:
        if values:
            overlays.append(
                {
                    "path": path,
                    "engines": sorted(values),
                    "keys": _keys_by_engine(values),
                }
            )

    record("sim.profiles", spec.sim.profiles)
    for term_path, term in spec.mdp.terms().items():
        record(f"mdp.{term_path}.engine_params", term.engine_params)
    record("agent.engine_overrides", spec.agent.engine_overrides)
    return sorted(overlays, key=lambda entry: entry["path"])


def _native_extras(spec: TaskSpec) -> list[dict[str, Any]]:
    return [
        {
            "engine": engine,
            "keys": sorted(str(key) for key in extras),
        }
        for engine, extras in sorted(spec.engine_extras.items())
        if extras
    ]


def _clean_resolution(
    resolution_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fields = ("skipped", "emulated", "omitted")
    known = resolution_manifest is not None and all(
        field in resolution_manifest for field in fields
    )
    if not known:
        return {
            "known": False,
            "clean": None,
            "skipped": None,
            "emulated": None,
            "omitted": None,
        }
    counts = {field: len(resolution_manifest.get(field, {})) for field in fields}
    return {
        "known": True,
        "clean": not any(counts.values()),
        **counts,
    }


def portability_report(
    spec: TaskSpec,
    resolution_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return contract, overlay, native-extra, and resolution dimensions.

    A task may have a portable multi-engine contract while intentionally
    retaining native solver or randomization values. Conversely, a declaration
    with no overlays can still compile non-cleanly on a backend. Keeping these
    dimensions separate prevents one optimistic boolean from hiding either
    condition.
    """
    semantic_overlays = _semantic_overlays(spec)
    native_extras = _native_extras(spec)
    return {
        "contract_portability": {
            "multi_engine": len(spec.engines) > 1,
            "declared_engines": list(spec.engines),
        },
        "semantic_overlays": {
            "count": len(semantic_overlays),
            "entries": semantic_overlays,
        },
        "native_extras": {
            "count": len(native_extras),
            "entries": native_extras,
        },
        "clean_resolution": _clean_resolution(resolution_manifest),
    }


__all__ = ["portability_report"]
