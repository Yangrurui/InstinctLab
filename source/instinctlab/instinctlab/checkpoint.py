"""Checkpoint compatibility metadata for the unified training entry points."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from instinctlab_engine.spec import TaskSpec, portability_report
from instinctlab_engine.spec.robot import BackendAsset

_CONTRACT_VERSION = "task_contract_v2"
_LEGACY_CONTRACT_VERSION = "task_spec_v1"
_POLICY_IO_VERSION = "policy_io_v2"
_EXPERIMENT_SEMANTICS_VERSION = "experiment_semantics_v1"
_PROVENANCE_VERSION = "provenance_v1"
_DEFAULT_CHECKPOINT_PATTERN = r"model_.*\.pt"
_PUBLIC_TYPE_MODULES = {
    "instinctlab_engine.spec.motion_reference": "instinctlab_engine.spec.sensor",
    "instinctlab_engine.spec.volume": "instinctlab_engine.spec.sensor",
}


def _type_name(value: Any) -> str:
    value_type = type(value)
    # Contract v1 records the public declaration path, not its physical source file. Keeping this
    # mapping stable lets the implementation be split into cohesive modules without invalidating
    # checkpoints whose tensor contract did not change.
    module = _PUBLIC_TYPE_MODULES.get(value_type.__module__, value_type.__module__)
    return f"{module}.{value_type.__qualname__}"


def _canonical(
    value: Any,
    *,
    include_asset_paths: bool = False,
    include_omitted_fields: bool = False,
) -> Any:
    """Convert a declaration to deterministic JSON without importing an engine."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Checkpoint contracts cannot contain non-finite value {value!r}.")
        return value
    if isinstance(value, Enum):
        return {
            "type": _type_name(value),
            "value": _canonical(
                value.value,
                include_asset_paths=include_asset_paths,
                include_omitted_fields=include_omitted_fields,
            ),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BackendAsset):
        # ``asset_id`` identifies the robot. Absolute source paths vary between clones and
        # machines and must not make an otherwise identical checkpoint unloadable.
        return {
            "type": _type_name(value),
            "fields": [
                [
                    field.name,
                    _canonical(
                        getattr(value, field.name),
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                ]
                for field in fields(value)
                if include_asset_paths or field.name != "path"
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):

        def include(field) -> bool:
            if field.metadata.get("contract_omit", False) and not include_omitted_fields:
                return False
            if not field.metadata.get("contract_omit_if_default", False):
                return True
            default = field.default
            if default is MISSING:
                default = field.default_factory()
            return getattr(value, field.name) != default

        return {
            "type": _type_name(value),
            "fields": [
                [
                    field.name,
                    _canonical(
                        getattr(value, field.name),
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                ]
                for field in fields(value)
                if include(field)
            ],
        }
    if isinstance(value, Mapping):
        # Mapping order is part of the tensor contract for observations, actions and rewards.
        return {
            "mapping": [
                [
                    _canonical(
                        key,
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                    _canonical(
                        item,
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                ]
                for key, item in value.items()
            ]
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical(
                item,
                include_asset_paths=include_asset_paths,
                include_omitted_fields=include_omitted_fields,
            )
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        items = [
            _canonical(
                item,
                include_asset_paths=include_asset_paths,
                include_omitted_fields=include_omitted_fields,
            )
            for item in value
        ]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module and qualname and "<locals>" not in qualname:
            return {"callable": f"{module}.{qualname}"}
    raise TypeError(f"Task contract cannot serialize {_type_name(value)}: {value!r}.")


def _fingerprint(version: str, payload: Any, **canonical_options: bool) -> dict[str, str]:
    canonical = {
        "version": version,
        "payload": _canonical(payload, **canonical_options),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return {
        "version": version,
        "hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


_RUNNER_LIFECYCLE_FIELDS = frozenset(
    {
        "device",
        "experiment_name",
        "load_checkpoint",
        "load_run",
        "log_interval",
        "max_iterations",
        "num_steps_per_env",
        "resume",
        "run_name",
        "save_interval",
        "seed",
    }
)
"""Runner controls that neither construct nor restore checkpoint state."""


def _as_agent_config(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            f"Agent config {type(config).__module__}.{type(config).__qualname__} "
            "must be a mapping or expose to_dict() for checkpoint contracts."
        )
    return to_dict()


def _agent_config(spec: TaskSpec) -> Mapping[str, Any]:
    config = spec.agent.resolve()(**dict(spec.agent.overrides))
    return _as_agent_config(config)


def _policy_agent_contract(agent_config: Mapping[str, Any], runner: str) -> dict[str, Any]:
    """Configuration of every component whose tensors the runner restores.

    This deliberately excludes only runner lifecycle controls. Policy, critic,
    normalizers, algorithm-owned discriminators, teacher/student networks, VAE
    modules, optimizer-sensitive topology, checkpoint manipulators, and future
    auxiliary components therefore enter the mandatory hash automatically.
    """
    return {
        "runner": runner,
        "restored_components": {
            key: value
            for key, value in agent_config.items()
            if key not in _RUNNER_LIFECYCLE_FIELDS
        },
    }


def _policy_io_payload(spec: TaskSpec, agent_config: Mapping[str, Any]) -> dict[str, Any]:
    motion_schemas = [
        {
            "name": reference.name,
            "joints": reference.joints,
            "links": reference.links,
            "symmetric_augmentation": reference.symmetric_augmentation,
        }
        for reference in spec.scene.motion_references
    ]
    return {
        "robot": {
            "schema_version": spec.robot.schema_version,
            "asset_id": spec.robot.asset_id,
            "joint_names": spec.robot.joint_names,
            "body_names": spec.robot.body_names,
            "joint_defaults_and_action_scale": tuple(
                (item.name, item.default_pos, item.action_scale)
                for item in spec.robot.joint_properties
            ),
        },
        "observations": spec.mdp.observations,
        "actions": spec.mdp.actions,
        "motion_reference_schemas": motion_schemas,
        "agent": _policy_agent_contract(agent_config, spec.agent.runner),
    }


def _experiment_semantics_payload(
    spec: TaskSpec, agent_config: Mapping[str, Any]
) -> dict[str, Any]:
    agent_training_keys = (
        "seed",
        "num_steps_per_env",
        "max_iterations",
        "policy",
        "algorithm",
        "normalizers",
    )
    return {
        "robot": spec.robot,
        "scene": spec.scene,
        "sim": spec.sim,
        "rewards": spec.mdp.rewards,
        "terminations": spec.mdp.terminations,
        "commands": spec.mdp.commands,
        "events": spec.mdp.events,
        "curriculum": spec.mdp.curriculum,
        "agent_training": {
            key: agent_config[key] for key in agent_training_keys if key in agent_config
        },
    }


def task_contract(
    spec: TaskSpec, *, agent_config: Mapping[str, Any] | object | None = None
) -> dict[str, Any]:
    """Separate checkpoint compatibility, experiment semantics, and provenance."""
    resolved_agent_config = (
        _agent_config(spec)
        if agent_config is None
        else _as_agent_config(agent_config)
    )
    return {
        "version": _CONTRACT_VERSION,
        "task_id": spec.task_id,
        "robot_schema_version": spec.robot.schema_version,
        "asset_id": spec.robot.asset_id,
        "joint_names": list(spec.robot.joint_names),
        "body_names": list(spec.robot.body_names),
        "portability": portability_report(spec),
        "policy_io": _fingerprint(
            _POLICY_IO_VERSION,
            _policy_io_payload(spec, resolved_agent_config),
        ),
        "experiment_semantics": _fingerprint(
            _EXPERIMENT_SEMANTICS_VERSION,
            _experiment_semantics_payload(spec, resolved_agent_config),
        ),
        "provenance": _fingerprint(
            _PROVENANCE_VERSION,
            {"spec": spec, "agent_config": resolved_agent_config},
            include_asset_paths=True,
            include_omitted_fields=True,
        ),
    }


def add_task_contract(
    manifest: Mapping[str, Any],
    spec: TaskSpec,
    *,
    agent_config: Mapping[str, Any] | object | None = None,
) -> dict[str, Any]:
    """Return a compilation manifest carrying checkpoint compatibility metadata."""
    payload = dict(manifest)
    payload["portability"] = portability_report(spec, payload)
    payload["task_contract"] = task_contract(spec, agent_config=agent_config)
    return payload


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort ``model_<iteration>.pt`` numerically, then fall back to the filename."""
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def latest_checkpoint(run_dir: str | Path, pattern: str = _DEFAULT_CHECKPOINT_PATTERN) -> Path:
    """Return the newest numeric checkpoint in one run directory.

    ``pattern`` uses the same full-match regular-expression semantics as the runner config.  This
    keeps checkpoint discovery identical for training resume and playback.
    """
    run_path = Path(run_dir).expanduser().resolve()
    matcher = re.compile(pattern)
    checkpoints = [path for path in run_path.iterdir() if path.is_file() and matcher.fullmatch(path.name)]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint matching {pattern!r} under {run_path}")
    return max(checkpoints, key=checkpoint_sort_key)


def latest_run_checkpoint(
    log_root: str | Path,
    *,
    run_pattern: str = ".*",
    checkpoint_pattern: str = _DEFAULT_CHECKPOINT_PATTERN,
    skip_empty_runs: bool = False,
) -> Path:
    """Select a matching run, then its latest numeric checkpoint.

    Training resume treats the latest run as authoritative and reports a missing checkpoint there.
    Playback can set ``skip_empty_runs`` to walk backwards past freshly-created or interrupted run
    directories that contain no model yet.
    """
    root = Path(log_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint log root does not exist: {root}")
    matcher = re.compile(run_pattern)
    runs = sorted(path for path in root.iterdir() if path.is_dir() and matcher.fullmatch(path.name))
    if not runs:
        raise FileNotFoundError(f"no run matching {run_pattern!r} under {root}")
    if not skip_empty_runs:
        return latest_checkpoint(runs[-1], checkpoint_pattern)
    for run in reversed(runs):
        try:
            return latest_checkpoint(run, checkpoint_pattern)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no checkpoint matching {checkpoint_pattern!r} in runs under {root}")


def validate_checkpoint_contract(
    checkpoint: str | Path,
    spec: TaskSpec,
    *,
    checkpoint_task_id: str | None = None,
    experiment_policy: Literal["require", "warn", "ignore"] = "require",
    agent_config: Mapping[str, Any] | object | None = None,
) -> None:
    """Validate the manifest next to a checkpoint before loading its tensors.

    Legacy runs have no enforceable policy-I/O hash and remain loadable with a warning because
    existing Isaac and InstinctMJ checkpoints predate the unified launcher. New contracts always
    reject policy-I/O drift. ``experiment_policy`` controls whether changed training semantics
    reject resume, warn under an explicit override, or are ignored by play/export. A Play task may pass
    the explicitly registered training task id whose policy it consumes; training
    and resume callers omit it and remain strict about their own task identity.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if experiment_policy not in {"require", "warn", "ignore"}:
        raise ValueError(f"unknown checkpoint experiment policy {experiment_policy!r}")
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        warnings.warn(
            f"Checkpoint {checkpoint_path} has no adjacent manifest.json; compatibility cannot be verified. "
            "In particular, a legacy Isaac policy may use native BFS joint order rather than canonical DFS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    stored = manifest.get("task_contract")
    if not isinstance(stored, dict):
        warnings.warn(
            f"Checkpoint manifest {manifest_path} predates task contracts; compatibility cannot be verified. "
            "In particular, a legacy Isaac policy may use native BFS joint order rather than canonical DFS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    current = task_contract(spec, agent_config=agent_config)
    expected_task_id = checkpoint_task_id or current["task_id"]
    stored_version = stored.get("version")
    if stored.get("task_id") != expected_task_id:
        raise ValueError(
            f"Checkpoint task contract mismatch for {checkpoint_path}: "
            f"checkpoint task={stored.get('task_id')!r}, version={stored_version!r}; "
            f"runtime task={current['task_id']!r}, expected checkpoint task={expected_task_id!r}, "
            f"version={current['version']!r}."
        )
    stored_joint_names = stored.get("joint_names")
    if stored_joint_names != current["joint_names"]:
        raise ValueError(
            f"Checkpoint canonical joint order mismatch for {checkpoint_path}: "
            f"checkpoint={stored_joint_names!r}; runtime={current['joint_names']!r}. "
            "A BFS checkpoint cannot be loaded positionally into the DFS policy interface."
        )
    if stored.get("robot_schema_version") != current["robot_schema_version"]:
        raise ValueError(
            f"Checkpoint robot joint schema mismatch for {checkpoint_path}: "
            f"checkpoint={stored.get('robot_schema_version')!r}; "
            f"runtime={current['robot_schema_version']!r}."
        )
    if stored_version == _LEGACY_CONTRACT_VERSION:
        warnings.warn(
            f"Checkpoint manifest {manifest_path} has legacy task contract "
            f"{_LEGACY_CONTRACT_VERSION!r}; policy I/O compatibility cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if stored_version != current["version"]:
        raise ValueError(
            f"Checkpoint task contract version mismatch for {checkpoint_path}: "
            f"checkpoint={stored_version!r}; runtime={current['version']!r}."
        )
    stored_policy = stored.get("policy_io")
    if not isinstance(stored_policy, dict) or stored_policy != current["policy_io"]:
        raise ValueError(
            f"Checkpoint policy I/O contract mismatch for {checkpoint_path}: "
            f"checkpoint={stored_policy!r}; runtime={current['policy_io']!r}."
        )
    stored_experiment = stored.get("experiment_semantics")
    if stored_experiment != current["experiment_semantics"]:
        message = (
            f"Checkpoint experiment semantics mismatch for {checkpoint_path}: "
            f"checkpoint={stored_experiment!r}; runtime={current['experiment_semantics']!r}."
        )
        if experiment_policy == "require":
            raise ValueError(message)
        if experiment_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)


__all__ = [
    "add_task_contract",
    "checkpoint_sort_key",
    "latest_checkpoint",
    "latest_run_checkpoint",
    "task_contract",
    "validate_checkpoint_contract",
]
