"""One-rollout plant parity probe: InstinctLab vs InstinctMJ parkour mjlab factories.

Measures static runtime configuration and per-step physics from a *single* shared
initial condition, without changing task behaviour. Run once per side and diff:

    python scripts/probe_parkour_plant_parity.py run --side ours --mode dump \\
        --num-envs 2 --steps 0 --seed 42 --device cuda:2 --out /tmp/ours_dump.json
    CUDA_VISIBLE_DEVICES=0 /root/InstinctMJ/.venv/bin/python \\
        scripts/probe_parkour_plant_parity.py run --side instinctmj --mode dump \\
        --state-npz /tmp/ours_dump.state.npz --num-envs 2 --seed 42 --device cuda:0 \\
        --out /tmp/mj_dump.json
    python scripts/probe_parkour_plant_parity.py compare /tmp/ours_dump.json /tmp/mj_dump.json

``state.npz`` carries robot kinematics **and**, by default, the live
``PoseVelocityCommand`` runtime buffers (``vel_command_b``, ``pos_command_w``, …).
Without the command snapshot, independent resets can disagree on yaw commands by
~1.0 raw even when robot state matches — that is a probe artifact, not an MDP
drift. Use ``--no-command-state`` for pure-plant probes; ``compare`` then records
``command_dependent_mdp_parity=false`` and must not be read as same-state reward
or ``velocity_commands`` obs parity.

InstinctMJ **must** use ``/root/InstinctMJ/.venv/bin/python`` (the ``instinct-train``
interpreter) and the training GPU convention ``CUDA_VISIBLE_DEVICES=<n> --device cuda:0``.
The current process's site-packages are the wrong stack; a missing ``coacd`` there is
not a missing project dependency. Passing ``--device cuda:1`` inside that venv is a
Warp device mismatch, not a missing package.

Engine packages are imported only inside ``run`` (never at module import time), so
``--help`` and offline tests stay engine-free.

Camera causal A/B (ours factory only) — native production already uses groups (0,1,2)/no-hop;
``instinctmj_geom_groups`` is a compatibility alias that verifies native matches InstinctMJ::

    python scripts/probe_parkour_plant_parity.py run --side ours --mode policy_eval \\
        --checkpoint logs/mjlab/g1_parkour/.../model_700.pt --camera-semantics native \\
        --seed 42 --num-envs 256 --steps 500 --eval-warmup-steps 50 --device cuda:2 \\
        --out /tmp/cam_ab_native.json
    python scripts/probe_parkour_plant_parity.py run --side ours --mode policy_eval \\
        --checkpoint logs/mjlab/g1_parkour/.../model_700.pt --camera-semantics instinctmj_geom_groups \\
        --seed 42 --num-envs 256 --steps 500 --eval-warmup-steps 50 --device cuda:2 \\
        --out /tmp/cam_ab_instinctmj.json

Production ``engines/mjlab/camera.py`` uses InstinctMJ geom groups with first-hit/no-hop.
``instinctmj_geom_groups`` applies the legacy in-process patch only when native is not yet aligned.

Cross-factory policy_eval (native camera; InstinctMJ must use its training venv)::

    python scripts/probe_parkour_plant_parity.py run --side ours --mode policy_eval \\
        --checkpoint <ckpt> --seed 42 --num-envs 256 --steps 500 --eval-warmup-steps 50 \\
        --device cuda:2 --out /tmp/eval_ours_ours_s42.json
    CUDA_VISIBLE_DEVICES=0 /root/InstinctMJ/.venv/bin/python \\
        scripts/probe_parkour_plant_parity.py run --side instinctmj --mode policy_eval \\
        --checkpoint <ckpt> --seed 42 --num-envs 256 --steps 500 --eval-warmup-steps 50 \\
        --device cuda:0 --out /tmp/eval_ours_ref_s42.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

OURS_TASK_ID = "Instinct-Parkour-Target-G1"
INSTINCTMJ_TASK_ID = "Instinct-Parkour-Target-Amp-G1-v0"
DEFAULT_INSTINCTMJ_ROOT = Path("/root/InstinctMJ")
DEFAULT_INSTINCTMJ_PYTHON = DEFAULT_INSTINCTMJ_ROOT / ".venv/bin/python"
CAMERA_NAME = "camera"
ROOT_HEIGHT_MINIMUM = 0.5
COMMAND_NAME = "base_velocity"
COMMAND_STATE_SCHEMA = "parkour_pose_velocity_command/v1"
COMMAND_STATE_PREFIX = "cmd__"

# Live PoseVelocityCommand tensors that feed rewards, command metrics, and
# velocity_commands / heading-related obs. Missing on a term is recorded, not guessed.
COMMAND_FIELD_NAMES: tuple[str, ...] = (
    "pos_command_w",
    "heading_command_w",
    "pos_command_b",
    "vel_command_b",
    "max_command_b",
    "is_standing_env",
    "lin_vel_x_range",
    "lin_vel_y_range",
    "ang_vel_z_range",
    "random_lin_vel_x_range",
    "random_lin_vel_y_range",
    "random_ang_vel_z_range",
    "random_velocity_indices",
    "random_lin_vel_x",
    "random_lin_vel_y",
    "random_ang_vel_z",
    "time_left",
    "command_counter",
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "qpos": 1e-4,
    "root_z": 1e-3,
    "qfrc": 0.05,
    "action": 1e-3,
    "depth": 0.02,
}

# Causal order for first-divergence: observation → command → torque → kinematics.
COMPARE_FIELDS: tuple[tuple[str, str], ...] = (
    ("depth_processed", "depth"),
    ("depth_raw", "depth"),
    ("left_height_scanner_hits", "depth"),
    ("right_height_scanner_hits", "depth"),
    ("raw_action", "action"),
    ("processed_action", "action"),
    ("qfrc_actuator", "qfrc"),
    ("qpos", "qpos"),
    ("qvel", "qpos"),
    ("root_link_pos_w", "root_z"),
)
REQUIRED_STEP_FIELDS: frozenset[str] = frozenset(
    {"raw_action", "processed_action", "qpos", "qfrc_actuator", "root_link_pos_w"}
)


# ---------------------------------------------------------------------------
# Pure helpers (no engine imports — safe for offline tests)
# ---------------------------------------------------------------------------


def root_height_margin(root_z: float, origin_z: float, *, minimum: float = ROOT_HEIGHT_MINIMUM) -> float:
    """``root_z - clamp(origin_z, 0) - minimum_height`` (parkour root_height term)."""
    clamped = max(float(origin_z), 0.0)
    return float(root_z) - clamped - float(minimum)


def align_names_or_fail(left: list[str], right: list[str], *, label: str) -> None:
    if left != right:
        raise ValueError(f"{label} names differ: left={left!r} right={right!r}")


def command_npz_key(field_name: str) -> str:
    """Stable on-disk prefix for command tensors inside ``state.npz``."""
    return f"{COMMAND_STATE_PREFIX}{field_name}"


def command_state_status(*, captured: bool, loaded: bool, loaded_present: bool) -> str:
    """One of ``present`` / ``absent`` / ``loaded_present`` / ``loaded_absent``."""
    if loaded:
        return "loaded_present" if loaded_present else "loaded_absent"
    return "present" if captured else "absent"


def command_state_mdp_comparable(left_status: str, right_status: str) -> bool:
    """Both sides must carry a command snapshot for same-state MDP term parity."""
    ok = {"present", "loaded_present"}
    return left_status in ok and right_status in ok


def _command_snapshot_has_fields(command_state: dict[str, Any] | None) -> bool:
    return bool(command_state and command_state.get("fields"))


def capture_command_state(term: Any) -> dict[str, Any]:
    """Snapshot PoseVelocityCommand runtime buffers as host numpy arrays."""
    fields: dict[str, Any] = {}
    missing_on_term: list[str] = []
    for name in COMMAND_FIELD_NAMES:
        if not hasattr(term, name):
            missing_on_term.append(name)
            continue
        arr = _to_numpy(getattr(term, name))
        if arr is None:
            missing_on_term.append(name)
            continue
        fields[name] = arr
    return {
        "schema": COMMAND_STATE_SCHEMA,
        "command_name": COMMAND_NAME,
        "fields": fields,
        "missing_on_term": missing_on_term,
    }


def apply_command_state(term: Any, command_state: dict[str, Any] | None) -> dict[str, Any]:
    """Write a captured command snapshot into the live term.

    Shape or dtype mismatch raises ``ValueError``. Fields absent on the term or in
    the snapshot are listed in the returned report rather than silently skipped.
    """
    import torch

    if not command_state:
        return {
            "applied": [],
            "missing_on_term": [],
            "missing_in_snapshot": list(COMMAND_FIELD_NAMES),
            "schema": None,
        }
    fields = command_state.get("fields") or {}
    applied: list[str] = []
    missing_on_term: list[str] = []
    missing_in_snapshot: list[str] = []
    for name in COMMAND_FIELD_NAMES:
        if name not in fields:
            missing_in_snapshot.append(name)
            continue
        if not hasattr(term, name):
            missing_on_term.append(name)
            continue
        target = getattr(term, name)
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"{name}: live attribute is {type(target)!r}, expected torch.Tensor")
        source = fields[name]
        tensor = torch.as_tensor(source, device=target.device)
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} != live {tuple(target.shape)}")
        if tensor.dtype != target.dtype:
            tensor = tensor.to(dtype=target.dtype)
        target.copy_(tensor)
        applied.append(name)
    return {
        "applied": applied,
        "missing_on_term": missing_on_term,
        "missing_in_snapshot": missing_in_snapshot,
        "schema": command_state.get("schema"),
        "command_name": command_state.get("command_name"),
    }


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(x) for x in value]


def permute_by_names(values: Any, src_names: list[str], dst_names: list[str]) -> Any:
    """Reorder the last axis of ``values`` from ``src_names`` into ``dst_names``."""
    import numpy as np

    if list(src_names) == list(dst_names):
        return values
    missing = [name for name in dst_names if name not in src_names]
    extra = [name for name in src_names if name not in dst_names]
    if missing or extra:
        raise ValueError(f"name sets differ: missing={missing!r} extra={extra!r}")
    index = [src_names.index(name) for name in dst_names]
    arr = np.asarray(values)
    return arr[..., index]


def _max_abs_diff(left: Any, right: Any, *, treat_nonfinite_as: float | None = None) -> float:
    import numpy as np

    la = np.asarray(left, dtype=np.float64).reshape(-1)
    ra = np.asarray(right, dtype=np.float64).reshape(-1)
    if la.shape != ra.shape:
        raise ValueError(f"shape mismatch: {la.shape} vs {ra.shape}")
    if la.size == 0:
        return 0.0
    if treat_nonfinite_as is not None:
        la = np.where(np.isfinite(la), la, treat_nonfinite_as)
        ra = np.where(np.isfinite(ra), ra, treat_nonfinite_as)
    return float(np.max(np.abs(la - ra)))


def first_consecutive_two_step_exceedance(diffs: list[float], threshold: float) -> int | None:
    """Return the first step index ``i`` where ``diffs[i]`` and ``diffs[i+1]`` both exceed ``threshold``."""
    for index in range(len(diffs) - 1):
        if diffs[index] > threshold and diffs[index + 1] > threshold:
            return index
    return None


def tensor_summary(values: Any, *, digest_bytes: bytes | None = None) -> dict[str, Any]:
    """JSON-safe summary. Pass precomputed ``digest_bytes`` when torch is unavailable."""
    if digest_bytes is None:
        if values is None:
            return {"available": False, "reason": "value is None"}
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
        flat = arr.reshape(-1)
        digest_bytes = flat.tobytes()
        summary: dict[str, Any] = {
            "available": True,
            "shape": list(arr.shape),
            "min": float(flat.min()) if flat.size else None,
            "max": float(flat.max()) if flat.size else None,
            "mean": float(flat.mean()) if flat.size else None,
            "std": float(flat.std()) if flat.size else None,
            "sha256": hashlib.sha256(digest_bytes).hexdigest(),
        }
        return summary
    return {
        "available": True,
        "sha256": hashlib.sha256(digest_bytes).hexdigest(),
    }


def _load_companion_arrays(path: Path | None, key: str) -> Any | None:
    if path is None or not path.is_file():
        return None
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            return None
        return archive[key]


def _step_array(payload: dict[str, Any], field: str, step_index: int) -> Any | None:
    companion = payload.get("companion_npz")
    if companion:
        arr = _load_companion_arrays(Path(companion), f"step_{step_index}_{field}")
        if arr is not None:
            return arr
    step = next((item for item in payload.get("steps", []) if item.get("step_index") == step_index), None)
    if step is None:
        return None
    embedded = step.get(field)
    if embedded is None:
        return None
    if isinstance(embedded, dict) and "values" in embedded:
        return embedded["values"]
    return embedded


def _step_origins_z(payload: dict[str, Any], step_index: int) -> Any | None:
    arr = _step_array(payload, "env_origins_z", step_index)
    if arr is not None:
        return arr
    step = next((item for item in payload.get("steps", []) if item.get("step_index") == step_index), None)
    if step is None:
        return None
    return step.get("env_origins_z")


def _relative_root_z(pos: Any, origins_z: Any | None) -> Any:
    import numpy as np

    z = np.asarray(pos, dtype=float)
    if z.ndim >= 2:
        z = z[:, 2]
    elif z.size >= 3:
        z = np.asarray([z[2]], dtype=float)
    if origins_z is None:
        return z
    return z - np.asarray(origins_z, dtype=float).reshape(-1)[: z.size]


def first_field_exceedance(diffs: list[float], threshold: float, step_indices: list[int]) -> dict[str, Any] | None:
    """Two consecutive steps when possible; a single snapshot (dump) fails on one exceedance."""
    if len(diffs) < 2:
        for i, diff in enumerate(diffs):
            if diff > threshold:
                return {
                    "first_step_index": step_indices[i],
                    "diffs": [diff],
                    "reason": "single-step exceedance (dump/snapshot has no second step)",
                }
        return None
    hit = first_consecutive_two_step_exceedance(diffs, threshold)
    if hit is None:
        return None
    return {"first_step_index": step_indices[hit], "diffs": diffs[hit : hit + 2]}


def compare_rollout_payloads(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare two probe JSON outputs. Raises on name mismatch; reports first consecutive-two-step exceedance."""
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else {**DEFAULT_THRESHOLDS, **thresholds})
    static_l = left.get("static", {})
    static_r = right.get("static", {})
    joint_l = list(static_l.get("joint_names", []))
    joint_r = list(static_r.get("joint_names", []))
    if set(joint_l) != set(joint_r):
        raise ValueError(f"joint name sets differ: left={joint_l!r} right={joint_r!r}")
    action_l = list(static_l.get("action_target_names", []))
    action_r = list(static_r.get("action_target_names", []))
    if set(action_l) != set(action_r):
        raise ValueError(f"action name sets differ: left={action_l!r} right={action_r!r}")

    steps_l = sorted({int(s["step_index"]) for s in left.get("steps", []) if "step_index" in s})
    steps_r = sorted({int(s["step_index"]) for s in right.get("steps", []) if "step_index" in s})
    if steps_l != steps_r:
        raise ValueError(f"step indices differ: left={steps_l} right={steps_r}")

    named_joint = {"qpos", "qvel", "qfrc_actuator"}
    named_action = {"raw_action", "processed_action"}
    per_field_diffs: dict[str, list[float]] = {name: [] for name, _ in COMPARE_FIELDS}
    per_step_report: list[dict[str, Any]] = []
    missing_required: dict[str, Any] | None = None

    for step_index in steps_l:
        row: dict[str, Any] = {"step_index": step_index, "fields": {}}
        for field, thresh_key in COMPARE_FIELDS:
            tol = thresholds[thresh_key]
            lv = _step_array(left, field, step_index)
            rv = _step_array(right, field, step_index)
            if lv is None or rv is None:
                row["fields"][field] = {"status": "missing", "left": lv is not None, "right": rv is not None}
                per_field_diffs[field].append(float("inf") if field in REQUIRED_STEP_FIELDS else 0.0)
                if field in REQUIRED_STEP_FIELDS and missing_required is None and step_index >= 0:
                    missing_required = {
                        "field": field,
                        "first_step_index": step_index,
                        "reason": "required tensor missing; refusing to treat None as agreement",
                    }
                continue
            if field in named_joint:
                rv = permute_by_names(rv, joint_r, joint_l)
            elif field in named_action:
                rv = permute_by_names(rv, action_r, action_l)
            if field.startswith("depth"):
                diff = _max_abs_diff(lv, rv, treat_nonfinite_as=1.0e6)
            elif field == "root_link_pos_w":
                import numpy as np

                lz = _relative_root_z(lv, _step_origins_z(left, step_index))
                rz = _relative_root_z(rv, _step_origins_z(right, step_index))
                diff = float(np.max(np.abs(lz - rz)))
            else:
                diff = _max_abs_diff(lv, rv)
            per_field_diffs[field].append(diff)
            row["fields"][field] = {"max_abs_diff": diff, "threshold": tol, "exceeds": diff > tol}
        per_step_report.append(row)

    first_failure: dict[str, Any] | None = missing_required
    if first_failure is None:
        for field, thresh_key in COMPARE_FIELDS:
            hit = first_field_exceedance(per_field_diffs[field], thresholds[thresh_key], steps_l)
            if hit is not None:
                first_failure = {"field": field, "threshold": thresholds[thresh_key], **hit}
                break

    meta_l = left.get("metadata", {})
    meta_r = right.get("metadata", {})
    cmd_l = str(meta_l.get("command_state", "unknown"))
    cmd_r = str(meta_r.get("command_state", "unknown"))
    mdp_comparable = command_state_mdp_comparable(cmd_l, cmd_r)
    command_state_report: dict[str, Any] = {
        "left": cmd_l,
        "right": cmd_r,
        "command_dependent_mdp_parity": mdp_comparable,
    }
    if not mdp_comparable:
        command_state_report["note"] = (
            "Same-state MDP compare requires PoseVelocityCommand snapshot on both sides "
            "(metadata.command_state present or loaded_present). Pure plant kinematics "
            "without command sync can show ~1.0 raw yaw/heading pseudo-diffs."
        )

    geom_l = static_l.get("geometry_fingerprint") or {}
    geom_r = static_r.get("geometry_fingerprint") or {}
    geometry_report = {
        "available": bool(geom_l.get("available") and geom_r.get("available")),
        "combined_equal": (
            geom_l.get("combined_sha256") == geom_r.get("combined_sha256")
            if geom_l.get("available") and geom_r.get("available")
            else None
        ),
        "terrain_origins_equal": (
            geom_l.get("terrain_origins_sha256") == geom_r.get("terrain_origins_sha256")
            if geom_l.get("available") and geom_r.get("available")
            else None
        ),
        "left_ngeom": geom_l.get("ngeom"),
        "right_ngeom": geom_r.get("ngeom"),
        "left_nmesh": geom_l.get("nmesh"),
        "right_nmesh": geom_r.get("nmesh"),
    }

    return {
        "steps_compared": len(steps_l),
        "thresholds": thresholds,
        "per_step": per_step_report,
        "first_consecutive_two_step_exceedance": first_failure,
        "command_state": command_state_report,
        "geometry": geometry_report,
        "passed": first_failure is None,
    }


def instinctmj_train_task_source(root: Path = DEFAULT_INSTINCTMJ_ROOT) -> Path:
    return root / "src/instinct_mj/tasks/parkour/config/g1/__init__.py"


def read_instinctmj_train_registration(source: Path | None = None) -> dict[str, Any]:
    """Parse InstinctMJ parkour registration without importing the engine."""
    path = source or instinctmj_train_task_source()
    if not path.is_file():
        raise FileNotFoundError(f"InstinctMJ parkour registration not found: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    registrations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_instinct_task":
            task_id = None
            play_false = False
            for kw in node.keywords:
                if kw.arg == "task_id" and isinstance(kw.value, ast.Constant):
                    task_id = kw.value.value
                if kw.arg == "env_cfg_factory":
                    src = ast.unparse(kw.value).replace(" ", "")
                    play_false = "play=False" in src
            registrations.append({"task_id": task_id, "play_false_in_env_cfg_factory": play_false})
    train = next(
        (
            item
            for item in registrations
            if item.get("task_id") == INSTINCTMJ_TASK_ID and item.get("play_false_in_env_cfg_factory")
        ),
        None,
    )
    if train is None:
        raise ValueError(f"no train registration for {INSTINCTMJ_TASK_ID!r} with play=False in {path}")
    return {"task_id": train["task_id"], "play_false_in_env_cfg_factory": True, "source": str(path)}


def module_top_level_engine_imports(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    banned = ("mjlab", "instinct_mj", "instinctlab")
    hits: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in banned:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in banned:
                hits.append(node.module)
    return hits


def output_schema_keys() -> frozenset[str]:
    return frozenset({"metadata", "static", "steps"})


def policy_eval_schema_keys() -> frozenset[str]:
    return frozenset({"metadata", "static", "eval"})


def policy_eval_eval_keys() -> frozenset[str]:
    return frozenset({"episodes", "summary", "terrain_mapping", "verify_camera"})


def policy_eval_summary_keys() -> frozenset[str]:
    return frozenset(
        {
            "completed_episodes",
            "warmup_steps",
            "control_steps",
            "num_envs",
            "total_env_steps",
            "episodes_still_running_at_horizon",
            "episode_length_stat_scope",
            "mean_episode_length",
            "median_episode_length",
            "completed_episode_mean_length",
            "completed_episode_median_length",
            "termination_rate_per_1000_env_steps",
            "termination_counts",
            "termination_rates_per_1000_env_steps",
            "primary_reason_counts",
            "root_height_count",
            "root_height_rate_per_1000_env_steps",
            "root_height_margin",
            "per_terrain",
            "unresolved_terrain_episodes",
        }
    )


CAMERA_SEMANTICS_NATIVE = "native"
CAMERA_SEMANTICS_INSTINCTMJ = "instinctmj_geom_groups"
CAMERA_SEMANTICS_CHOICES = (CAMERA_SEMANTICS_NATIVE, CAMERA_SEMANTICS_INSTINCTMJ)


def instinctmj_reference_camera_geom_groups() -> tuple[int, ...]:
    """Geom groups InstinctMJ parkour camera inherits from mjlab ``RayCastSensorCfg``."""
    reader_path = Path(__file__).resolve().parents[1] / "tests" / "reference_mjlab_parkour.py"
    spec = importlib.util.spec_from_file_location("_instinctmj_parkour_reference", reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load InstinctMJ reference reader from {reader_path}")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    if not reader.available():
        raise FileNotFoundError("InstinctMJ reference tree is not checked out")
    groups = reader.camera_include_geom_groups()
    if groups is None:
        raise RuntimeError("InstinctMJ parkour camera would include all geom groups, not (0, 1, 2)")
    return groups


def geom_groups_camera_mask(mj_model, groups: tuple[int, ...], device: str):
    """All geoms whose MuJoCo group is in ``groups`` — InstinctMJ stock raycast semantics."""
    from instinctlab.engines.mjlab.camera import geom_groups_camera_mask as _mask

    return _mask(mj_model, groups, device)


def _production_camera_filter_name(sensor) -> str | None:
    post = getattr(sensor, "_apply_min_distance_hop", None)
    if post is not None and getattr(post, "__name__", "") == "_apply_min_distance_hop":
        return "geom_groups_min_distance_hop"
    legacy = getattr(sensor, "_apply_min_distance_no_hop", None)
    if legacy is not None and getattr(legacy, "__name__", "") == "_apply_min_distance_no_hop":
        return "geom_groups_reject_no_hop"
    body_hop = getattr(sensor, "_filter_and_continue", None)
    if body_hop is not None and getattr(body_hop, "__name__", "") == "_filter_and_continue":
        return "body_mesh_mask_with_hop"
    grouped_hop = getattr(sensor, "_apply_hit_filter_and_continue", None)
    if grouped_hop is not None and bool(getattr(sensor, "_needs_filter_continue", False)):
        return "geom_groups_min_distance_hop"
    return None


def _production_hop_metadata(sensor) -> dict[str, Any]:
    from instinctlab.engines.mjlab.camera import pinhole_camera_hop_params

    name = _production_camera_filter_name(sensor)
    hop = pinhole_camera_hop_params()
    cfg = getattr(sensor, "cfg", None)
    out: dict[str, Any] = {
        "camera_filter": name or "geom_groups_kernel_only",
        "hop_triggers": list(hop.get("hop_triggers", ())),
        "hop_epsilon_m": hop.get("hop_epsilon_m"),
    }
    if name == "geom_groups_min_distance_hop":
        out["hop_max"] = getattr(
            sensor,
            "_hop_max",
            getattr(sensor, "_mesh_filter_max_hops", hop["hop_max"]),
        )
        out["hop_epsilon_m"] = getattr(sensor, "_mesh_filter_epsilon", out["hop_epsilon_m"])
    elif name == "body_mesh_mask_with_hop":
        out["hop_max"] = 6
    else:
        out["hop_max"] = 0
    if cfg is not None and hasattr(cfg, "min_distance"):
        out["min_distance_m"] = float(cfg.min_distance)
    return out


def native_camera_already_instinctmj_aligned(sensor) -> bool:
    """True when production already matches InstinctMJ group mask + min_distance hop."""
    cfg = getattr(sensor, "cfg", None)
    groups = getattr(cfg, "include_geom_groups", None) if cfg is not None else None
    if groups is None:
        return False
    try:
        reference = instinctmj_reference_camera_geom_groups()
    except (FileNotFoundError, RuntimeError, LookupError):
        return False
    return tuple(groups) == reference and _production_camera_filter_name(sensor) == "geom_groups_min_distance_hop"


def camera_semantics_metadata(sensor, semantics: str) -> dict[str, Any]:
    """Runtime dump proving which camera filter is active."""
    base = _collect_camera_runtime_from_sensor(sensor) if sensor is not None else {"available": False}
    base["semantics"] = semantics
    base["instinctmj_reference_groups"] = list(instinctmj_reference_camera_geom_groups())
    base["native_already_aligned"] = native_camera_already_instinctmj_aligned(sensor) if sensor is not None else False
    if sensor is not None:
        base.update(_production_hop_metadata(sensor))
    if semantics == CAMERA_SEMANTICS_INSTINCTMJ and base.get("native_already_aligned"):
        base["alias_note"] = "instinctmj_geom_groups is a no-op alias; production already matches InstinctMJ"
    return base


def _collect_camera_runtime_from_sensor(sensor) -> dict[str, Any]:
    cfg = getattr(sensor, "cfg", None)
    info: dict[str, Any] = {"available": True, "sensor_name": getattr(cfg, "name", CAMERA_NAME)}
    if cfg is not None:
        for attr in (
            "include_geom_groups",
            "origin_offset",
            "origin_offset_rot",
            "image_height",
            "image_width",
            "min_distance",
            "image_plane_max",
        ):
            if hasattr(cfg, attr):
                value = getattr(cfg, attr)
                info[f"cfg_{attr}"] = list(value) if isinstance(value, tuple) else value
    hop_meta = _production_hop_metadata(sensor)
    info.update(hop_meta)
    mask = getattr(sensor, "_allowed_geom_mask", None)
    if mask is not None:
        import torch

        count = int(mask.sum().item()) if isinstance(mask, torch.Tensor) else int(mask.sum())
        info["allowed_geom_count"] = count
    elif hop_meta.get("camera_filter") == "geom_groups_min_distance_hop":
        groups = getattr(cfg, "include_geom_groups", None) if cfg is not None else None
        info["include_geom_groups"] = list(groups) if groups is not None else None
    return info


class _CameraSemanticsPatch:
    """Restore native camera hop/mask after a probe arm finishes."""

    def __init__(self, sensor, semantics: str):
        self.sensor = sensor
        self.semantics = semantics
        self._orig_mask = None
        self._orig_filter = None
        self._orig_post = None
        self._already_aligned = False

    def apply(self, mj_model, device: str) -> None:
        if self.semantics == CAMERA_SEMANTICS_NATIVE:
            return
        if self.semantics != CAMERA_SEMANTICS_INSTINCTMJ:
            raise ValueError(f"unknown camera semantics {self.semantics!r}")
        if native_camera_already_instinctmj_aligned(self.sensor):
            setattr(self.sensor, "_probe_camera_semantics", self.semantics)
            self._already_aligned = True
            return
        raise RuntimeError(
            f"camera semantics {self.semantics!r} requested but production is not InstinctMJ-aligned; "
            "refusing legacy body-mask patch"
        )

    def restore(self) -> None:
        if getattr(self, "_already_aligned", False):
            if hasattr(self.sensor, "_probe_camera_semantics"):
                delattr(self.sensor, "_probe_camera_semantics")
            return
        if self._orig_mask is None and self._orig_filter is None and self._orig_post is None:
            return
        if self._orig_mask is not None:
            self.sensor._allowed_geom_mask = self._orig_mask
        if self._orig_filter is not None:
            self.sensor._filter_and_continue = self._orig_filter
        if hasattr(self.sensor, "_probe_camera_semantics"):
            delattr(self.sensor, "_probe_camera_semantics")

    def metadata(self) -> dict[str, Any]:
        return camera_semantics_metadata(self.sensor, self.semantics)


def apply_camera_semantics(env, semantics: str) -> _CameraSemanticsPatch:
    sensor = env.scene.sensors.get(CAMERA_NAME)
    if sensor is None:
        raise RuntimeError(f"no camera sensor {CAMERA_NAME!r} on env")
    patch = _CameraSemanticsPatch(sensor, semantics)
    patch.apply(env.sim.mj_model, str(env.device))
    return patch


def curriculum_column_indices(proportions: list[float], num_cols: int) -> list[int]:
    """Isaac Lab curriculum assignment: column ``j`` is the first type with cdf > j/n + 0.001."""
    import numpy as np

    weights = np.asarray(proportions, dtype=np.float64)
    if weights.size == 0:
        raise RuntimeError("curriculum column assignment needs at least one sub-terrain proportion.")
    weights = weights / weights.sum()
    cumulative = np.cumsum(weights)
    indices: list[int] = []
    for index in range(num_cols):
        matches = np.where(index / num_cols + 0.001 < cumulative)[0]
        if matches.size == 0:
            raise RuntimeError(
                f"curriculum column {index} of {num_cols} matched no sub-terrain. "
                f"Normalized proportions: {weights.tolist()}."
            )
        indices.append(int(np.min(matches)))
    return indices


def _actual_column_count(terrain) -> int | None:
    patches = getattr(terrain, "flat_patches", None)
    if patches and "target" in patches:
        return int(patches["target"].shape[1])
    origins = getattr(terrain, "terrain_origins", None)
    if origins is not None:
        return int(origins.shape[1])
    return None


def _sub_terrain_kind(cfg: object) -> str:
    return type(cfg).__name__


def resolve_terrain_name_mapping(terrain) -> dict[str, Any]:
    """Dump column-id → sub-terrain name from the live grid. Never invent names."""
    if terrain is None:
        return {"available": False, "reason": "scene has no terrain object"}
    cfg = getattr(terrain, "cfg", None)
    generator = getattr(cfg, "terrain_generator", None) if cfg is not None else None
    if generator is None:
        return {"available": False, "reason": "terrain.cfg.terrain_generator missing; refusing bare column ids"}
    names = list(getattr(generator, "sub_terrains", {}) or {})
    if not names:
        return {"available": False, "reason": "terrain_generator.sub_terrains is empty"}
    n_cols = _actual_column_count(terrain)
    if n_cols is None:
        return {"available": False, "reason": "cannot read column count (no flat_patches['target'] or terrain_origins)"}
    curriculum = bool(getattr(generator, "curriculum", False))
    if not curriculum:
        return {
            "available": False,
            "reason": "non-curriculum grid mixes types inside a column; names are unresolvable",
            "num_cols": n_cols,
            "sub_terrain_names": names,
        }
    sub = generator.sub_terrains
    proportions = [float(sub[name].proportion) for name in names]
    kinds = [_sub_terrain_kind(sub[name]) for name in names]
    if n_cols == len(names):
        allocation = "one_column_per_type"
        column_to_name = list(names)
        column_to_kind = list(kinds)
    else:
        allocation = "isaac_cumulative_proportion"
        try:
            indices = curriculum_column_indices(proportions, n_cols)
        except RuntimeError as exc:
            return {"available": False, "reason": str(exc), "num_cols": n_cols, "sub_terrain_names": names}
        column_to_name = [names[index] for index in indices]
        column_to_kind = [kinds[index] for index in indices]
    return {
        "available": True,
        "allocation": allocation,
        "num_cols": n_cols,
        "declared_num_cols": getattr(generator, "num_cols", None),
        "sub_terrain_names": names,
        "sub_terrain_kinds": kinds,
        "proportions": proportions,
        "column_to_name": column_to_name,
        "column_to_kind": column_to_kind,
        "columns": [
            {"column": index, "name": column_to_name[index], "kind": column_to_kind[index]} for index in range(n_cols)
        ],
    }


def _scene_terrain(env):
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    terrain = getattr(scene, "terrain", None)
    if terrain is not None:
        return terrain
    if hasattr(scene, "__getitem__"):
        try:
            return scene["terrain"]
        except Exception:
            return None
    return None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median_or_none(values: list[float]) -> float | None:
    import statistics

    if not values:
        return None
    return float(statistics.median(values))


def summarize_policy_eval(
    episodes: list[dict[str, Any]],
    *,
    control_steps: int,
    num_envs: int,
    warmup_steps: int,
) -> dict[str, Any]:
    """Episode/fall aggregates. Per-terrain keys are names, never bare column ids."""
    counted = [event for event in episodes if int(event.get("control_step", -1)) >= warmup_steps]
    lengths = [int(event["episode_length"]) for event in counted]
    term_counts: dict[str, int] = {}
    primary_counts: dict[str, int] = {}
    root_height = 0
    margins = [float(event["root_height_margin"]) for event in counted if event.get("root_height_margin") is not None]
    fall_margins = [
        float(event["root_height_margin"])
        for event in counted
        if event.get("termination_reasons", {}).get("root_height") and event.get("root_height_margin") is not None
    ]
    per_terrain_acc: dict[str, dict[str, Any]] = {}
    unresolved = 0
    for event in counted:
        reasons = event.get("termination_reasons", {})
        for name, fired in reasons.items():
            if fired:
                term_counts[name] = term_counts.get(name, 0) + 1
        if reasons.get("root_height"):
            root_height += 1
        primary = str(event.get("primary_reason") or "unknown")
        primary_counts[primary] = primary_counts.get(primary, 0) + 1
        terrain_name = event.get("terrain_name")
        if not terrain_name:
            unresolved += 1
            terrain_name = "<unresolved>"
        bucket = per_terrain_acc.setdefault(
            terrain_name, {"lengths": [], "margins": [], "fall_margins": [], "root_height": 0}
        )
        bucket["lengths"].append(int(event["episode_length"]))
        if event.get("root_height_margin") is not None:
            bucket["margins"].append(float(event["root_height_margin"]))
        if reasons.get("root_height"):
            bucket["root_height"] += 1
            if event.get("root_height_margin") is not None:
                bucket["fall_margins"].append(float(event["root_height_margin"]))
    env_steps = max(control_steps, 1) * max(num_envs, 1)
    completed_mean_length = _mean_or_none([float(x) for x in lengths])
    completed_median_length = _median_or_none([float(x) for x in lengths])
    per_terrain = {
        name: {
            "completed_episodes": len(bucket["lengths"]),
            "mean_episode_length": _mean_or_none(bucket["lengths"]),
            "median_episode_length": _median_or_none(bucket["lengths"]),
            "root_height_count": bucket["root_height"],
            "mean_root_height_margin": _mean_or_none(bucket["margins"]),
            "mean_root_height_margin_at_fall": _mean_or_none(bucket["fall_margins"]),
        }
        for name, bucket in per_terrain_acc.items()
    }
    return {
        "completed_episodes": len(counted),
        "warmup_steps": warmup_steps,
        "control_steps": control_steps,
        "num_envs": num_envs,
        "total_env_steps": env_steps,
        "episodes_still_running_at_horizon": num_envs,
        "episode_length_stat_scope": (
            "completed episodes only; right-censored episodes at the evaluation horizon are excluded, "
            "and episodes crossing warmup may include pre-measurement steps"
        ),
        # Backward-compatible names. Consumers should prefer the explicit aliases below and
        # termination_rate_per_1000_env_steps for fixed-horizon factory/policy comparisons.
        "mean_episode_length": completed_mean_length,
        "median_episode_length": completed_median_length,
        "completed_episode_mean_length": completed_mean_length,
        "completed_episode_median_length": completed_median_length,
        "termination_rate_per_1000_env_steps": len(counted) * 1000.0 / env_steps,
        "termination_counts": term_counts,
        "termination_rates_per_1000_env_steps": {
            name: count * 1000.0 / env_steps for name, count in term_counts.items()
        },
        "primary_reason_counts": primary_counts,
        "root_height_count": root_height,
        "root_height_rate_per_1000_env_steps": root_height * 1000.0 / env_steps,
        "root_height_margin": {
            "mean": _mean_or_none(margins),
            "median": _median_or_none(margins),
            "mean_at_root_height_term": _mean_or_none(fall_margins),
            "n_at_root_height_term": len(fall_margins),
        },
        "per_terrain": per_terrain,
        "unresolved_terrain_episodes": unresolved,
    }


def terrain_name_shares(per_terrain: dict[str, Any]) -> dict[str, float]:
    usable = {
        name: int(row.get("completed_episodes") or 0)
        for name, row in per_terrain.items()
        if name and name != "<unresolved>"
    }
    total = sum(usable.values())
    if total <= 0:
        return {}
    return {name: count / total for name, count in usable.items()}


def reweight_mean_length_to_mix(per_terrain: dict[str, Any], target_shares: dict[str, float]) -> dict[str, Any]:
    """Reweight this factory's per-name mean lengths onto another factory's name mix."""
    shared = [
        name
        for name in target_shares
        if name in per_terrain and per_terrain[name].get("mean_episode_length") is not None
    ]
    if not shared:
        return {"available": False, "reason": "no shared named terrains with a mean length"}
    mass = sum(target_shares[name] for name in shared)
    if mass <= 0:
        return {"available": False, "reason": "target mix has zero mass on shared names"}
    value = 0.0
    weights: dict[str, float] = {}
    for name in shared:
        weight = target_shares[name] / mass
        weights[name] = weight
        value += weight * float(per_terrain[name]["mean_episode_length"])
    dropped = sorted(set(target_shares) - set(shared))
    return {
        "available": True,
        "value": value,
        "shared_names": shared,
        "weights": weights,
        "dropped_target_names": dropped,
    }


def analyze_policy_eval_2x2(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Factory effect (same ckpt, swap factory) vs policy effect (same factory, swap ckpt)."""
    rows = []
    for arm in arms:
        meta = arm.get("metadata", {})
        summary = arm.get("eval", {}).get("summary", {})
        mapping = arm.get("eval", {}).get("terrain_mapping", {})
        rows.append(
            {
                "side": meta.get("side"),
                "checkpoint": meta.get("checkpoint"),
                "seed": meta.get("seed"),
                "summary": summary,
                "terrain_mapping": mapping,
            }
        )
    factory_pairs = []
    policy_pairs = []
    for left in rows:
        for right in rows:
            if left is right:
                continue
            if (
                left["checkpoint"] == right["checkpoint"]
                and left["seed"] == right["seed"]
                and left["side"] != right["side"]
            ):
                if left["side"] == "ours" and right["side"] == "instinctmj":
                    factory_pairs.append((left, right))
            if (
                left["side"] == right["side"]
                and left["seed"] == right["seed"]
                and left["checkpoint"] != right["checkpoint"]
            ):
                if (left["checkpoint"] or "") < (right["checkpoint"] or ""):
                    policy_pairs.append((left, right))

    def _delta(ours_like: dict[str, Any], other: dict[str, Any], *, label: str) -> dict[str, Any]:
        s0, s1 = ours_like["summary"], other["summary"]
        len0, len1 = s0.get("mean_episode_length"), s1.get("mean_episode_length")
        term0 = s0.get("termination_rate_per_1000_env_steps")
        term1 = s1.get("termination_rate_per_1000_env_steps")
        rh0, rh1 = s0.get("root_height_rate_per_1000_env_steps"), s1.get("root_height_rate_per_1000_env_steps")
        shares1 = terrain_name_shares(s1.get("per_terrain") or {})
        shares0 = terrain_name_shares(s0.get("per_terrain") or {})
        re_ours = reweight_mean_length_to_mix(s0.get("per_terrain") or {}, shares1)
        re_ref = reweight_mean_length_to_mix(s1.get("per_terrain") or {}, shares0)
        return {
            "kind": label,
            "seed": ours_like["seed"],
            "left_side": ours_like["side"],
            "right_side": other["side"],
            "left_checkpoint": ours_like["checkpoint"],
            "right_checkpoint": other["checkpoint"],
            "mean_len_left": len0,
            "mean_len_right": len1,
            "mean_len_delta": None if len0 is None or len1 is None else float(len0) - float(len1),
            "mean_len_scope": "completed episodes only; not censoring-safe",
            "termination_rate_left": term0,
            "termination_rate_right": term1,
            "termination_rate_delta": None if term0 is None or term1 is None else float(term0) - float(term1),
            "root_height_rate_left": rh0,
            "root_height_rate_right": rh1,
            "root_height_rate_delta": None if rh0 is None or rh1 is None else float(rh0) - float(rh1),
            "completed_left": s0.get("completed_episodes"),
            "completed_right": s1.get("completed_episodes"),
            "reweighted_left_len_onto_right_mix": re_ours,
            "reweighted_right_len_onto_left_mix": re_ref,
            "left_terrain_allocation": (ours_like.get("terrain_mapping") or {}).get("allocation"),
            "right_terrain_allocation": (other.get("terrain_mapping") or {}).get("allocation"),
        }

    return {
        "n_arms": len(rows),
        "factory_effect_same_ckpt": [_delta(left, right, label="factory") for left, right in factory_pairs],
        "policy_effect_same_factory": [_delta(left, right, label="policy") for left, right in policy_pairs],
    }


def classify_episode_termination(
    *,
    termination_reasons: dict[str, bool],
    truncated: bool,
) -> str:
    """Primary reason label for one finished episode."""
    if truncated and not any(termination_reasons.values()):
        return "time_out"
    for name, fired in termination_reasons.items():
        if fired and name != "time_out":
            return name
    if termination_reasons.get("time_out"):
        return "time_out"
    if truncated:
        return "time_out"
    return "unknown"


def snapshot_pre_reset_episodes(
    env,
    env_ids,
    *,
    control_step: int,
    terrain_mapping: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Capture terminal kinematics/reasons at ``_reset_idx`` — before buffers are cleared."""
    import torch

    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=env.device)
    if ids.numel() == 0:
        return []
    if ids.ndim == 0:
        ids = ids.unsqueeze(0)
    term_mgr = env.termination_manager
    active = term_mgr.active_terms if isinstance(term_mgr.active_terms, list) else term_mgr.active_terms()
    robot = env.scene["robot"] if hasattr(env.scene, "__getitem__") else env.scene.robot
    terrain = _scene_terrain(env)
    column_to_name = None
    if terrain_mapping and terrain_mapping.get("available"):
        column_to_name = list(terrain_mapping.get("column_to_name") or [])
    events: list[dict[str, Any]] = []
    for idx in ids.tolist():
        reasons = {name: bool(term_mgr.get_term(name)[idx].item()) for name in active}
        root_z = float(robot.data.root_link_pos_w[idx, 2].item())
        origin_z = float(env.scene.env_origins[idx, 2].item())
        truncated = (
            bool(env.reset_time_outs[idx].item())
            if hasattr(env, "reset_time_outs")
            else bool(term_mgr.time_outs[idx].item())
        )
        type_id = None
        level = None
        terrain_name = None
        if terrain is not None:
            types = getattr(terrain, "terrain_types", None)
            levels = getattr(terrain, "terrain_levels", None)
            if types is not None:
                type_id = int(types[idx].item())
            if levels is not None:
                level = int(levels[idx].item())
            if type_id is not None and column_to_name is not None and 0 <= type_id < len(column_to_name):
                terrain_name = column_to_name[type_id]
        events.append(
            {
                "env_index": int(idx),
                "control_step": int(control_step),
                "episode_length": int(env.episode_length_buf[idx].item()),
                "terminated": (
                    bool(env.reset_terminated[idx].item())
                    if hasattr(env, "reset_terminated")
                    else bool(term_mgr.terminated[idx].item())
                ),
                "truncated": truncated,
                "termination_reasons": reasons,
                "primary_reason": classify_episode_termination(termination_reasons=reasons, truncated=truncated),
                "root_height_margin": root_height_margin(root_z, origin_z),
                "root_z": root_z,
                "origin_z": origin_z,
                "terrain_type_id": type_id,
                "terrain_level": level,
                "terrain_name": terrain_name,
            }
        )
    return events


class EpisodeEvalRecorder:
    """Hooks ``_reset_idx`` so done/reason are read before auto_reset clears them."""

    def __init__(self, terrain_mapping: dict[str, Any] | None = None) -> None:
        self.episodes: list[dict[str, Any]] = []
        self.terrain_mapping = terrain_mapping
        self._control_step = -1
        self._orig_reset_idx = None
        self._env = None

    def bind(self, env, *, control_step: int) -> None:
        self._control_step = control_step
        target = getattr(env, "unwrapped", env)
        if self._env is not target:
            self._env = target
            if self._orig_reset_idx is None:
                self._orig_reset_idx = target._reset_idx

                def wrapped_reset_idx(env_ids):
                    self.episodes.extend(
                        snapshot_pre_reset_episodes(
                            target,
                            env_ids,
                            control_step=self._control_step,
                            terrain_mapping=self.terrain_mapping,
                        )
                    )
                    return self._orig_reset_idx(env_ids)

                target._reset_idx = wrapped_reset_idx

    def unbind(self) -> None:
        if self._env is not None and self._orig_reset_idx is not None:
            self._env._reset_idx = self._orig_reset_idx
        self._env = None
        self._orig_reset_idx = None


# ---------------------------------------------------------------------------
# Runtime (lazy engine imports)
# ---------------------------------------------------------------------------


def _repo_commit(repo: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def instinctmj_python_hint(current: str | None = None) -> str:
    """Tell the caller to use InstinctMJ's training venv, not this interpreter."""
    exe = current or sys.executable
    return (
        f"InstinctMJ side must use {DEFAULT_INSTINCTMJ_PYTHON} "
        "(the interpreter behind instinct-train) with CUDA_VISIBLE_DEVICES=<n> "
        f"and --device cuda:0. Current executable is {exe}. "
        "Do not pip-install missing modules into this env. Example:\n"
        f"  CUDA_VISIBLE_DEVICES=0 {DEFAULT_INSTINCTMJ_PYTHON} {Path(__file__).resolve()} "
        "run --side instinctmj --mode dump --device cuda:0 --num-envs 2 --steps 0 "
        "--out /tmp/mj_dump.json"
    )


def _ensure_instinctmj_root(root: Path) -> Path:
    src = root / "src"
    if not src.is_dir():
        raise RuntimeError(
            f"InstinctMJ reference tree not found at {root}. Set INSTINCTMJ_ROOT; refusing to fall back to ours."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _silence_observation_noise(env_cfg: object) -> None:
    groups = env_cfg.observations
    items = groups.values() if isinstance(groups, dict) else vars(groups).values()
    for group in items:
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def _disable_terminations(env_cfg: object) -> None:
    """Null every termination term so nothing can fire.

    This is the plant-compare switch: without it, ``auto_reset`` (mjlab default True)
    replaces a dying env in-place, and a post-step ``root_z``/``qpos`` sample is the
    *next* spawn while ``termination_manager.terminated`` still describes the previous
    body. The probe always sets ``auto_reset=False`` as well, so flag and kinematics
    stay on the same control step even if a term is left enabled.
    """
    terms = env_cfg.terminations
    if isinstance(terms, dict):
        for key in list(terms.keys()):
            terms[key] = None
        return
    for key in list(vars(terms)):
        setattr(terms, key, None)


def _prepare_probe_env_cfg(env_cfg: object, *, disable_terminations: bool, friction_fixed: bool) -> None:
    if hasattr(env_cfg, "auto_reset"):
        env_cfg.auto_reset = False
    if disable_terminations:
        _disable_terminations(env_cfg)
    if friction_fixed:
        _disable_friction_randomization(env_cfg)


def _disable_friction_randomization(env_cfg: object) -> None:
    events = env_cfg.events
    if isinstance(events, dict):
        if "physics_material" in events:
            events["physics_material"] = None
        return
    if hasattr(events, "physics_material"):
        events.physics_material = None


def _fix_actuator_lag(env_cfg: object, lag: int | None) -> None:
    """Set every builtin actuator to one deterministic lag before the env is built."""
    if lag is None:
        return
    robot_cfg = env_cfg.scene.entities["robot"]
    for actuator in robot_cfg.articulation.actuators:
        if hasattr(actuator, "delay_min_lag"):
            actuator.delay_min_lag = int(lag)
            actuator.delay_max_lag = int(lag)


def _build_ours(
    *,
    num_envs: int,
    device: str,
    seed: int,
    disable_obs_noise: bool,
    disable_terminations: bool,
    friction_fixed: bool,
    terrain_num_cols: int | None = None,
    actuator_lag: int | None = None,
):
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    spec = parkour_target_g1()
    if terrain_num_cols is not None:
        spec.sim.profiles.setdefault("mjlab", {})["num_cols"] = int(terrain_num_cols)
    MjlabAdapter.bootstrap(argparse.Namespace(device=device))
    compiled = MjlabAdapter().compile(spec, num_envs=num_envs, device=device)
    compiled.env_cfg.seed = seed
    if disable_obs_noise:
        _silence_observation_noise(compiled.env_cfg)
    _fix_actuator_lag(compiled.env_cfg, actuator_lag)
    _prepare_probe_env_cfg(compiled.env_cfg, disable_terminations=disable_terminations, friction_fixed=friction_fixed)
    env = compiled.make_env()
    return env, compiled, spec


def _build_instinctmj(
    *,
    num_envs: int,
    device: str,
    seed: int,
    disable_obs_noise: bool,
    disable_terminations: bool,
    friction_fixed: bool,
    root: Path,
    terrain_num_cols: int | None = None,
    actuator_lag: int | None = None,
):
    _ensure_instinctmj_root(root)
    try:
        import instinct_mj.tasks  # noqa: F401
        from instinct_mj.envs import InstinctRlEnv
        from instinct_mj.tasks.registry import load_env_cfg
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{exc}\n{instinctmj_python_hint()}") from exc

    env_cfg = load_env_cfg(INSTINCTMJ_TASK_ID, play=False)
    if env_cfg is None:
        raise SystemExit(f"{INSTINCTMJ_TASK_ID} train factory returned None; play=False is required")
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    if terrain_num_cols is not None and env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_cols = int(terrain_num_cols)
    if disable_obs_noise:
        _silence_observation_noise(env_cfg)
    _fix_actuator_lag(env_cfg, actuator_lag)
    _prepare_probe_env_cfg(env_cfg, disable_terminations=disable_terminations, friction_fixed=friction_fixed)
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    if device not in {"cuda:0", "cpu"}:
        print(
            "[WARN] InstinctMJ training uses CUDA_VISIBLE_DEVICES=<n> and --device cuda:0. "
            f"You passed {device}; Warp often then raycasts on cuda:0 and faults. "
            f"Prefer: CUDA_VISIBLE_DEVICES=0 {DEFAULT_INSTINCTMJ_PYTHON} ... --device cuda:0",
            flush=True,
        )
    env = InstinctRlEnv(cfg=env_cfg, device=device)
    return env, env_cfg


def _action_term(env):
    return env.action_manager.get_term("joint_pos")


def _action_target_names(env) -> list[str]:
    term = _action_term(env)
    return list(getattr(term, "target_names", getattr(term, "_joint_names", [])))


def _joint_names(env) -> list[str]:
    return list(env.scene["robot"].joint_names)


def _native_indices_for_names(env, names: list[str]):
    """Column indices into ``robot.data.joint_*`` for ``names`` (name order, not argsort)."""
    import torch

    native = _joint_names(env)
    missing = set(names) - set(native)
    if missing:
        raise RuntimeError(f"robot missing joints from state npz: {sorted(missing)}")
    return torch.tensor([native.index(name) for name in names], device=env.device, dtype=torch.long)


def _action_buffers(action_term) -> tuple[Any, Any]:
    """Return (raw, processed) tensors. mjlab exposes ``raw_action`` but only ``_processed_actions``."""
    raw = getattr(action_term, "raw_action", None)
    if raw is None:
        raw = getattr(action_term, "_raw_actions", None)
    processed = getattr(action_term, "_processed_actions", None)
    if processed is None:
        processed = getattr(action_term, "processed_actions", None)
    return raw, processed


def _command_term(env):
    return env.command_manager.get_term(COMMAND_NAME)


def _capture_state(env, joint_order: list[str], *, include_command: bool = True) -> dict[str, Any]:
    robot = env.scene["robot"]
    data = robot.data
    native_ids = _native_indices_for_names(env, joint_order)
    qpos = data.joint_pos[:, native_ids].detach().cpu().numpy()
    qvel = data.joint_vel[:, native_ids].detach().cpu().numpy()
    root_pos = (data.root_link_pos_w - env.scene.env_origins).detach().cpu().numpy()
    root_quat = data.root_link_quat_w.detach().cpu().numpy()
    root_lin = data.root_link_lin_vel_w.detach().cpu().numpy()
    root_ang = data.root_link_ang_vel_w.detach().cpu().numpy()
    state: dict[str, Any] = {
        "joint_names": list(joint_order),
        "action_target_names": _action_target_names(env),
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin,
        "root_ang_vel": root_ang,
        "joint_pos": qpos,
        "joint_vel": qvel,
    }
    if include_command:
        try:
            state["command_state"] = capture_command_state(_command_term(env))
        except (AttributeError, KeyError):
            state["command_state"] = None
    return state


def _refresh_sensors_and_obs(env) -> None:
    """After a write: kinematics, rays, then one obs compute so depth matches the written pose."""
    import torch

    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(env.sim, "sense"):
        env.sim.sense()
    env_ids = torch.arange(env.num_envs, device=env.device)
    if hasattr(env, "observation_manager"):
        env.observation_manager.reset(env_ids)
        env.obs_buf = env.observation_manager.compute(update_history=True)


def _camera_observation_terms(env) -> list[Any]:
    """Return the live depth observation callables without importing either project."""
    manager = getattr(env, "observation_manager", None)
    cfgs = getattr(manager, "_group_obs_term_cfgs", {}) if manager is not None else {}
    found: list[Any] = []
    seen: set[int] = set()
    for group_cfgs in cfgs.values():
        for cfg in group_cfgs:
            term = getattr(cfg, "func", None)
            if term is None or id(term) in seen:
                continue
            if hasattr(term, "_delay") or hasattr(term, "_num_delayed_frames"):
                found.append(term)
                seen.add(id(term))
    return found


def _prime_camera_history_at_current_state(env, *, reference_ray_range: bool = False) -> dict[str, Any]:
    """Clear both camera implementations and prime exactly one frame at the current pose."""
    import torch

    sensor = env.scene.sensors.get(CAMERA_NAME)
    if sensor is None:
        raise RuntimeError(f"controlled camera probe needs sensor {CAMERA_NAME!r}")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    range_report: dict[str, Any] | None = None
    cfg = getattr(sensor, "cfg", None)
    image_plane_max = getattr(cfg, "image_plane_max", None) if cfg is not None else None
    if reference_ray_range and image_plane_max is not None:
        original = float(cfg.max_distance)
        cfg.max_distance = float(image_plane_max)
        range_report = {"original_m": original, "controlled_m": float(cfg.max_distance)}
    sensor.reset(env_ids)
    terms = _camera_observation_terms(env)
    term_report: list[dict[str, Any]] = []
    for term in terms:
        if hasattr(term, "clear_history"):
            term.clear_history(env_ids)
        if hasattr(term, "reset"):
            term.reset(env_ids)
        delay_name = "_delay" if hasattr(term, "_delay") else "_num_delayed_frames"
        delay = getattr(term, delay_name)
        delay[env_ids] = 0
        term_report.append({"type": type(term).__name__, "delay_field": delay_name, "delay": 0})

    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(env.sim, "sense"):
        env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return {
        "enabled": True,
        "sensor_type": type(sensor).__name__,
        "history_phase": "sensor reset, one sense, one observation compute",
        "terms": term_report,
        "ray_range": range_report,
    }


def _apply_state(env, state: dict[str, Any]) -> dict[str, Any]:
    import torch

    stored_names = list(state["joint_names"])
    native = _joint_names(env)
    if set(stored_names) != set(native):
        raise ValueError(f"joint name sets differ: stored={stored_names!r} native={native!r}")
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    root_pos = torch.as_tensor(state["root_pos"], device=env.device, dtype=torch.float32)
    root_quat = torch.as_tensor(state["root_quat"], device=env.device, dtype=torch.float32)
    root_lin = torch.as_tensor(state["root_lin_vel"], device=env.device, dtype=torch.float32)
    root_ang = torch.as_tensor(state["root_ang_vel"], device=env.device, dtype=torch.float32)
    joint_pos = torch.as_tensor(
        permute_by_names(state["joint_pos"], stored_names, native), device=env.device, dtype=torch.float32
    )
    joint_vel = torch.as_tensor(
        permute_by_names(state["joint_vel"], stored_names, native), device=env.device, dtype=torch.float32
    )
    pose = torch.cat([root_pos + env.scene.env_origins, root_quat], dim=-1)
    velocity = torch.cat([root_lin, root_ang], dim=-1)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    command_report: dict[str, Any] = {"applied": [], "missing_in_snapshot": list(COMMAND_FIELD_NAMES), "schema": None}
    if state.get("command_state"):
        try:
            command_report = apply_command_state(_command_term(env), state["command_state"])
        except (AttributeError, KeyError):
            command_report = {
                "applied": [],
                "missing_on_term": list(COMMAND_FIELD_NAMES),
                "missing_in_snapshot": [],
                "schema": state["command_state"].get("schema"),
                "error": f"command term {COMMAND_NAME!r} unavailable",
            }
    _refresh_sensors_and_obs(env)
    return command_report


def _write_state_npz(path: Path, state: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "joint_names": np.array(state["joint_names"], dtype=object),
        "action_target_names": np.array(state.get("action_target_names", []), dtype=object),
        "root_pos": state["root_pos"],
        "root_quat": state["root_quat"],
        "root_lin_vel": state["root_lin_vel"],
        "root_ang_vel": state["root_ang_vel"],
        "joint_pos": state["joint_pos"],
        "joint_vel": state["joint_vel"],
    }
    command_state = state.get("command_state")
    if _command_snapshot_has_fields(command_state):
        arrays["command_state_schema"] = np.array(command_state.get("schema", COMMAND_STATE_SCHEMA))
        arrays["command_name"] = np.array(command_state.get("command_name", COMMAND_NAME))
        for name, value in command_state["fields"].items():
            arrays[command_npz_key(name)] = value
    np.savez(path, **arrays)


def _load_state_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=True) as archive:
        loaded: dict[str, Any] = {
            "joint_names": [str(x) for x in archive["joint_names"].tolist()],
            "action_target_names": [str(x) for x in archive.get("action_target_names", []).tolist()],
            "root_pos": archive["root_pos"],
            "root_quat": archive["root_quat"],
            "root_lin_vel": archive["root_lin_vel"],
            "root_ang_vel": archive["root_ang_vel"],
            "joint_pos": archive["joint_pos"],
            "joint_vel": archive["joint_vel"],
        }
        schema = archive.get("command_state_schema")
        fields: dict[str, Any] = {}
        prefix_len = len(COMMAND_STATE_PREFIX)
        for key in archive.files:
            if key.startswith(COMMAND_STATE_PREFIX):
                fields[key[prefix_len:]] = archive[key]
        if schema is None or not fields:
            loaded["command_state"] = None
            return loaded
        loaded["command_state"] = {
            "schema": str(np.asarray(schema).item()),
            "command_name": str(np.asarray(archive.get("command_name", COMMAND_NAME)).item()),
            "fields": fields,
            "missing_on_term": [],
        }
        return loaded


def _load_action_npz(path: Path, target_names: list[str]) -> Any:
    import numpy as np

    with np.load(path, allow_pickle=True) as archive:
        names = [str(x) for x in archive["action_target_names"].tolist()]
        align_names_or_fail(names, target_names, label="action_target")
        return np.asarray(archive["actions"])


def dump_state_output_path(*, out: Path, state_npz: Path | None, incoming_exists: bool) -> Path:
    """Write captured state next to --out when --state-npz is an existing input."""
    if incoming_exists:
        return out.with_suffix(".state.npz")
    return state_npz if state_npz is not None else out.with_suffix(".state.npz")


def _to_numpy(value: Any):
    """Host copy of a torch / Warp / numpy array. Warp ``nacon`` is an array, not an int."""
    import numpy as np

    if value is None:
        return None
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        try:
            out = numpy_fn()
            return out if isinstance(out, np.ndarray) else np.asarray(out)
        except TypeError:
            pass
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def _first_int(value: Any) -> int | None:
    arr = _to_numpy(value)
    if arr is None or getattr(arr, "size", 0) == 0:
        return None
    return int(arr.reshape(-1)[0])


def _max_int(value: Any) -> int | None:
    arr = _to_numpy(value)
    if arr is None or getattr(arr, "size", 0) == 0:
        return None
    return int(arr.max())


def _mjlab_contact_snapshot(env) -> dict[str, Any]:
    """Read nacon/nefc without importing instinctlab (InstinctMJ venv has none)."""
    raw = getattr(env, "unwrapped", env)
    sim = getattr(raw, "sim", None)
    wp_data = getattr(sim, "wp_data", None)
    if wp_data is None:
        return {"available": False, "reason": "sim.wp_data unavailable"}
    cfg = getattr(sim, "cfg", None)
    return {
        "available": True,
        "nacon": _first_int(getattr(wp_data, "nacon", None)),
        "nefc_max": _max_int(getattr(wp_data, "nefc", None)),
        "nconmax": getattr(cfg, "nconmax", None) if cfg is not None else None,
        "njmax": getattr(cfg, "njmax", None) if cfg is not None else None,
    }


def _collect_camera_runtime(env) -> dict[str, Any]:
    sensor = env.scene.sensors.get(CAMERA_NAME)
    if sensor is None:
        return {"available": False, "reason": f"no sensor {CAMERA_NAME!r}"}
    semantics = getattr(sensor, "_probe_camera_semantics", CAMERA_SEMANTICS_NATIVE)
    return camera_semantics_metadata(sensor, semantics)


def _model_geometry_fingerprint(env) -> dict[str, Any]:
    """Hash camera-visible MuJoCo geometry so a same-mesh claim is auditable."""
    import numpy as np

    model = getattr(env.sim, "mj_model", None)
    if model is None:
        return {"available": False, "reason": "sim.mj_model unavailable"}
    fields = (
        "mesh_vert",
        "mesh_face",
        "mesh_vertadr",
        "mesh_vertnum",
        "mesh_faceadr",
        "mesh_facenum",
        "hfield_data",
        "hfield_size",
        "geom_type",
        "geom_dataid",
        "geom_pos",
        "geom_quat",
        "geom_group",
    )
    per_field: dict[str, Any] = {}
    combined = hashlib.sha256()
    for name in fields:
        value = getattr(model, name, None)
        if value is None:
            continue
        arr = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        per_field[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": digest}
        combined.update(name.encode("utf-8"))
        combined.update(str(arr.dtype).encode("ascii"))
        combined.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        combined.update(arr.tobytes())
    terrain = _scene_terrain(env)
    origins = _to_numpy(getattr(terrain, "terrain_origins", None)) if terrain is not None else None
    origins_digest = None
    if origins is not None:
        origins_arr = np.ascontiguousarray(origins)
        origins_digest = hashlib.sha256(origins_arr.tobytes()).hexdigest()
    return {
        "available": True,
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "combined_sha256": combined.hexdigest(),
        "terrain_origins_sha256": origins_digest,
        "fields": per_field,
    }


def _collect_static(env, *, side: str, task_id: str, source_path: str | None) -> dict[str, Any]:
    import numpy as np

    robot = env.scene["robot"]
    data = robot.data
    action_term = _action_term(env)
    soft = data.soft_joint_pos_limits.detach().cpu().numpy()
    actuators: list[dict[str, Any]] = []
    for act in robot.actuators:
        entry = {
            "target_names": list(getattr(act, "target_names", [])),
            "stiffness": _tolist(getattr(act, "stiffness", None)),
            "damping": _tolist(getattr(act, "damping", None)),
            "armature": _tolist(getattr(act, "armature", None)),
            "effort_limit": _tolist(getattr(act, "effort_limit", None)),
        }
        actuators.append(entry)
    sim = env.sim
    mj_model = getattr(sim, "mj_model", None)
    opt: dict[str, Any] | None = None
    if mj_model is not None:
        opt = {
            "timestep": float(mj_model.opt.timestep),
            "gravity": np.asarray(mj_model.opt.gravity, dtype=float).tolist(),
            "iterations": int(mj_model.opt.iterations),
            "ls_iterations": int(mj_model.opt.ls_iterations),
            "ccd_iterations": int(getattr(mj_model.opt, "ccd_iterations", 0)),
        }
    sim_cfg = getattr(sim, "cfg", None)
    return {
        "joint_names": _joint_names(env),
        "action_target_names": _action_target_names(env),
        "action_scale": _tolist(getattr(action_term, "scale", None)),
        "default_joint_pos": data.default_joint_pos.detach().cpu().numpy().tolist(),
        "soft_joint_pos_limits_shape": list(soft.shape),
        "soft_joint_pos_limits": soft.tolist(),
        "actuators": actuators,
        "mujoco_opt": opt,
        "nconmax": getattr(sim_cfg, "nconmax", None) if sim_cfg is not None else None,
        "njmax": getattr(sim_cfg, "njmax", None) if sim_cfg is not None else None,
        "camera": _collect_camera_runtime(env),
        "geometry_fingerprint": _model_geometry_fingerprint(env),
        "side": side,
        "task_id": task_id,
        "source_path": source_path,
    }


def _tolist(value) -> Any:
    if value is None:
        return None
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return float(value)


def _raw_depth_tensor(env):
    """Sensor plane depth, same wash on both sides. Does not import instinctlab."""
    sensor = env.scene.sensors.get(CAMERA_NAME)
    if sensor is None:
        return None
    return _depth_raw_from_sensor(sensor)


def _depth_raw_from_sensor(sensor):
    """``distance_to_image_plane`` as (env, H, W, 1); miss / past far plane → +inf."""
    import torch

    data = getattr(sensor, "data", None)
    output = getattr(data, "output", None)
    if not isinstance(output, dict) or "distance_to_image_plane" not in output:
        return None
    image = output["distance_to_image_plane"]
    if image.ndim == 3:
        image = image.unsqueeze(-1)
    cfg = getattr(sensor, "cfg", None)
    far = getattr(cfg, "image_plane_max", None) if cfg is not None else None
    if far is None and cfg is not None:
        far = getattr(cfg, "max_distance", None)
    needs_inf = ~torch.isfinite(image)
    # InstinctMJ clips misses and out-of-range rays to exactly max_distance, so
    # equality must be canonicalized to +inf as well for a semantic raw compare.
    too_far = image >= float(far) if far is not None else torch.zeros_like(image, dtype=torch.bool)
    if not bool(needs_inf.any()) and not bool(too_far.any()):
        return image
    cleaned = image.clone()
    cleaned[needs_inf | too_far] = float("inf")
    return cleaned


def _processed_depth_tensor(env):
    """Policy ``depth_image`` from the cached obs pack. Never ``compute(update_history=True)``."""
    buf = getattr(env, "obs_buf", None)
    if not buf:
        buf = env.observation_manager.compute(update_history=False)
    policy = buf.get("policy") if isinstance(buf, dict) else None
    if isinstance(policy, dict) and "depth_image" in policy:
        return policy["depth_image"]
    return None


def _depth_payload(raw, processed) -> dict[str, Any]:
    return {
        "raw": _torch_summary(raw) if raw is not None else {"available": False, "reason": f"no {CAMERA_NAME}"},
        "processed": (
            _torch_summary(processed)
            if processed is not None
            else {"available": False, "reason": "policy depth_image missing from obs_buf"}
        ),
    }


def _camera_debug_tensors(env) -> dict[str, Any]:
    """Best-effort common camera internals for ray-level parity diagnosis."""
    sensor = env.scene.sensors.get(CAMERA_NAME)
    if sensor is None:
        return {}
    data = getattr(sensor, "data", None)
    raycast_data = getattr(sensor, "raycast_data", None)
    candidates = {
        "ray_origins": getattr(sensor, "_cached_world_origins", None),
        "ray_directions": getattr(sensor, "_cached_world_rays", None),
        "camera_pos": getattr(sensor, "_cam_pos", getattr(data, "pos_w", None)),
        "camera_quat": getattr(sensor, "_cam_quat", getattr(data, "quat_w_world", None)),
        "hit_positions": getattr(sensor, "_hit_pos_w", getattr(raycast_data, "positions", None)),
        "ray_distances": getattr(sensor, "_distances", getattr(raycast_data, "distances", None)),
    }
    return {name: value for name, value in candidates.items() if value is not None}


def _height_scanner_hits(env, name: str) -> Any | None:
    sensor = env.scene.sensors.get(name)
    if sensor is None:
        return None
    data = getattr(sensor, "data", None)
    hits = getattr(data, "ray_hits_w", None)
    if hits is None:
        hits = getattr(data, "hit_pos_w", None)
    return hits


def _torch_summary(tensor) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return {"available": False, "reason": f"not a tensor: {type(tensor)!r}"}
    arr = tensor.detach().float().cpu().numpy()
    return tensor_summary(arr)


def _policy_obs_summary(env, wrapper=None) -> dict[str, Any]:
    import torch

    if wrapper is not None:
        obs, _extras = wrapper.get_observations()
        return _torch_summary(obs)
    obs_pack = env.observation_manager.compute()
    policy = obs_pack.get("policy")
    if isinstance(policy, dict):
        parts = {key: _torch_summary(val) for key, val in policy.items()}
        return {"kind": "dict", "terms": parts}
    if isinstance(policy, torch.Tensor):
        return {"kind": "tensor", "summary": _torch_summary(policy)}
    return {"available": False, "reason": f"unexpected policy obs type {type(policy)!r}"}


def _collect_step_record(
    env,
    *,
    phase: str,
    step_index: int,
    action_term,
    store_arrays: dict[str, Any],
    depth_raw=None,
    depth_processed=None,
) -> dict[str, Any]:
    robot = env.scene["robot"]
    data = robot.data
    names = _joint_names(env)
    native_ids = _native_indices_for_names(env, names)
    qpos = data.joint_pos[:, native_ids].detach().cpu().numpy()
    qvel = data.joint_vel[:, native_ids].detach().cpu().numpy()
    qfrc = data.qfrc_actuator[:, native_ids].detach().cpu().numpy()
    root = data.root_link_pos_w.detach().cpu().numpy()
    root_quat = data.root_link_quat_w.detach().cpu().numpy()
    root_vel = data.root_link_lin_vel_w.detach().cpu().numpy()
    origins_z = env.scene.env_origins[:, 2].detach().cpu().numpy()
    margins = [
        root_height_margin(float(root[env_idx, 2]), float(origins_z[env_idx])) for env_idx in range(env.num_envs)
    ]
    term_mgr = env.termination_manager
    active = term_mgr.active_terms if isinstance(term_mgr.active_terms, list) else term_mgr.active_terms()
    term_flags = {name: bool(term_mgr._term_dones[name].any().item()) for name in active}
    raw, processed = _action_buffers(action_term)
    timeouts = (
        term_mgr.time_outs
        if hasattr(term_mgr, "time_outs") and not callable(term_mgr.time_outs)
        else term_mgr.time_outs()
    )
    truncated_any = bool(timeouts.any().item())
    if depth_raw is None:
        depth_raw = _raw_depth_tensor(env)
    if depth_processed is None:
        depth_processed = _processed_depth_tensor(env)
    record: dict[str, Any] = {
        "phase": phase,
        "step_index": step_index,
        "root_link_pos_w": root.tolist(),
        "root_link_quat_w": root_quat.tolist(),
        "root_link_lin_vel_w": root_vel.tolist(),
        "env_origins_z": origins_z.tolist(),
        "root_height_margin": margins,
        "termination_any": bool(term_mgr.terminated.any().item()),
        "truncation_any": truncated_any,
        "auto_reset": bool(getattr(env.cfg, "auto_reset", True)),
        "termination_flags": term_flags,
        "contact": _mjlab_contact_snapshot(env),
        "depth": _depth_payload(depth_raw, depth_processed),
        "raw_action_summary": (
            _torch_summary(raw) if raw is not None else {"available": False, "reason": "no raw_action buffer"}
        ),
        "processed_action_summary": (
            _torch_summary(processed)
            if processed is not None
            else {"available": False, "reason": "no _processed_actions buffer"}
        ),
    }
    prefix = f"step_{step_index}"
    store_arrays[f"{prefix}_qpos"] = qpos
    store_arrays[f"{prefix}_qvel"] = qvel
    store_arrays[f"{prefix}_qfrc_actuator"] = qfrc
    store_arrays[f"{prefix}_root_link_pos_w"] = root
    store_arrays[f"{prefix}_env_origins_z"] = origins_z
    if raw is not None:
        store_arrays[f"{prefix}_raw_action"] = raw.detach().cpu().numpy()
    if processed is not None:
        store_arrays[f"{prefix}_processed_action"] = processed.detach().cpu().numpy()
    if depth_raw is not None:
        store_arrays[f"{prefix}_depth_raw"] = depth_raw.detach().float().cpu().numpy()
    if depth_processed is not None:
        store_arrays[f"{prefix}_depth_processed"] = depth_processed.detach().float().cpu().numpy()
    for name, value in _camera_debug_tensors(env).items():
        array = _to_numpy(value)
        if array is not None:
            store_arrays[f"{prefix}_camera_{name}"] = array
    for scanner_name in ("left_height_scanner", "right_height_scanner"):
        hits = _height_scanner_hits(env, scanner_name)
        array = _to_numpy(hits)
        if array is not None:
            store_arrays[f"{prefix}_{scanner_name}_hits"] = array
    record["qpos_summary"] = tensor_summary(qpos)
    return record


def _inference_action(policy, obs):
    """Policy output for env.step: no grad, no autograd graph into mjlab delay buffers."""
    import torch

    with torch.inference_mode():
        action = policy(obs)
    if isinstance(action, torch.Tensor):
        return action.detach()
    return action


def _validate_run_args(args: argparse.Namespace) -> None:
    if getattr(args, "camera_reference_ray_range", False) and not getattr(args, "camera_controlled", False):
        raise SystemExit("--camera-reference-ray-range requires --camera-controlled")
    if args.mode == "policy_eval":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --mode policy_eval")
        if args.camera_semantics not in CAMERA_SEMANTICS_CHOICES:
            raise SystemExit(f"--camera-semantics must be one of {CAMERA_SEMANTICS_CHOICES}")
        if args.side != "ours" and args.camera_semantics != CAMERA_SEMANTICS_NATIVE:
            raise SystemExit("--camera-semantics override is only supported on --side ours")
        return
    if args.mode == "live_policy" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --mode live_policy")
    if args.mode not in {"live_policy", "policy_eval"} and args.checkpoint is not None:
        raise SystemExit(f"--checkpoint is not used with --mode {args.mode}")
    if args.out is None:
        raise SystemExit("--out is required")
    if args.mode == "dump" and args.action_npz is not None:
        raise SystemExit("--action-npz is not used with --mode dump")
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if getattr(args, "camera_semantics", CAMERA_SEMANTICS_NATIVE) != CAMERA_SEMANTICS_NATIVE and args.side != "ours":
        raise SystemExit("--camera-semantics override is only supported on --side ours")


def _build_ours_eval(
    *,
    num_envs: int,
    device: str,
    seed: int,
):
    """Training cfg with obs noise off and friction DR off; terminations and auto_reset stay on."""
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    spec = parkour_target_g1()
    MjlabAdapter.bootstrap(argparse.Namespace(device=device))
    compiled = MjlabAdapter().compile(spec, num_envs=num_envs, device=device)
    compiled.env_cfg.seed = seed
    _silence_observation_noise(compiled.env_cfg)
    _disable_friction_randomization(compiled.env_cfg)
    if hasattr(compiled.env_cfg, "auto_reset"):
        compiled.env_cfg.auto_reset = True
    env = compiled.make_env()
    return env, compiled, spec


def _build_instinctmj_eval(
    *,
    num_envs: int,
    device: str,
    seed: int,
    root: Path,
):
    """InstinctMJ *training* factory (play=False). Terminations and auto_reset stay on."""
    _ensure_instinctmj_root(root)
    try:
        import instinct_mj.tasks  # noqa: F401
        from instinct_mj.envs import InstinctRlEnv
        from instinct_mj.tasks.registry import load_env_cfg
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{exc}\n{instinctmj_python_hint()}") from exc

    env_cfg = load_env_cfg(INSTINCTMJ_TASK_ID, play=False)
    if env_cfg is None:
        raise SystemExit(f"{INSTINCTMJ_TASK_ID} train factory returned None; play=False is required")
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    _silence_observation_noise(env_cfg)
    _disable_friction_randomization(env_cfg)
    if hasattr(env_cfg, "auto_reset"):
        env_cfg.auto_reset = True
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    if device not in {"cuda:0", "cpu"}:
        print(
            "[WARN] InstinctMJ training uses CUDA_VISIBLE_DEVICES=<n> and --device cuda:0. "
            f"You passed {device}; Warp often then raycasts on cuda:0 and faults. "
            f"Prefer: CUDA_VISIBLE_DEVICES=0 {DEFAULT_INSTINCTMJ_PYTHON} ... --device cuda:0",
            flush=True,
        )
    env = InstinctRlEnv(cfg=env_cfg, device=device)
    return env, env_cfg


def run_policy_eval(args: argparse.Namespace) -> int:
    _validate_run_args(args)
    if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
        sys.path.pop(0)

    import numpy as np

    from instinct_rl.runners import OnPolicyRunner

    side = args.side
    instinctmj_root = Path(os.environ.get("INSTINCTMJ_ROOT", DEFAULT_INSTINCTMJ_ROOT))
    camera_patch = None
    compiled = None
    if side == "ours":
        from instinctlab.utils.wrappers.instinct_rl.mjlab_vecenv_wrapper import MjlabVecEnvWrapper

        env, compiled, _spec = _build_ours_eval(num_envs=args.num_envs, device=args.device, seed=args.seed)
        camera_patch = apply_camera_semantics(env, args.camera_semantics)
        camera_meta = camera_patch.metadata()
        agent_cfg = compiled.agent_cfg
        wrapper = MjlabVecEnvWrapper(
            env,
            policy_group=getattr(agent_cfg, "policy_observation_group", "policy"),
            critic_group=getattr(agent_cfg, "critic_observation_group", "critic"),
        )
        task_id = OURS_TASK_ID
        source_path = Path(__file__).resolve().parents[1]
    elif side == "instinctmj":
        read_instinctmj_train_registration(instinctmj_train_task_source(instinctmj_root))
        from instinct_mj.rl import InstinctRlVecEnvWrapper
        from instinct_mj.tasks.registry import load_instinct_rl_cfg

        env, _env_cfg = _build_instinctmj_eval(
            num_envs=args.num_envs, device=args.device, seed=args.seed, root=instinctmj_root
        )
        camera_meta = _collect_camera_runtime(env)
        camera_meta["semantics"] = CAMERA_SEMANTICS_NATIVE
        agent_cfg = load_instinct_rl_cfg(INSTINCTMJ_TASK_ID)
        wrapper = InstinctRlVecEnvWrapper(
            env,
            policy_group=agent_cfg.policy_observation_group,
            critic_group=agent_cfg.critic_observation_group,
        )
        task_id = INSTINCTMJ_TASK_ID
        source_path = instinctmj_root
    else:
        raise SystemExit(f"unknown side {side!r}")

    agent_cfg.device = args.device
    obs_format = wrapper.get_obs_format()
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=args.device)
    try:
        runner.load(str(args.checkpoint))
    except Exception as exc:
        if camera_patch is not None:
            camera_patch.restore()
        env.close()
        raise SystemExit(f"checkpoint load failed for {args.checkpoint}: {exc}") from exc
    policy = runner.get_inference_policy(device=args.device)
    checkpoint_loaded = True

    terrain_mapping = resolve_terrain_name_mapping(_scene_terrain(getattr(env, "unwrapped", env)))
    recorder = EpisodeEvalRecorder(terrain_mapping=terrain_mapping)
    static = _collect_static(env, side=side, task_id=task_id, source_path=str(source_path))
    static["camera"] = camera_meta
    static["terrain_mapping"] = terrain_mapping

    post_reset_depth_native = None
    post_reset_depth_processed = None
    if args.verify_camera:
        _refresh_sensors_and_obs(env)
        post_reset_depth_native = _raw_depth_tensor(env)
        post_reset_depth_processed = _processed_depth_tensor(env)

    for step in range(args.steps):
        recorder.bind(env, control_step=step)
        obs, _ = wrapper.get_observations()
        action = _inference_action(policy, obs)
        wrapper.step(action)

    episodes = recorder.episodes
    recorder.unbind()
    summary = summarize_policy_eval(
        episodes,
        control_steps=args.steps,
        num_envs=args.num_envs,
        warmup_steps=args.eval_warmup_steps,
    )

    payload: dict[str, Any] = {
        "metadata": {
            "side": side,
            "mode": "policy_eval",
            "camera_semantics": args.camera_semantics if side == "ours" else CAMERA_SEMANTICS_NATIVE,
            "seed": args.seed,
            "steps": args.steps,
            "eval_warmup_steps": args.eval_warmup_steps,
            "num_envs": args.num_envs,
            "task_id": task_id,
            "commit": _repo_commit(source_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_loaded": checkpoint_loaded,
            "obs_format": {
                group: {key: list(shape) for key, shape in terms.items()} for group, terms in obs_format.items()
            },
            "disable_obs_noise": True,
            "friction_fixed": True,
            "disable_terminations": False,
            "auto_reset": bool(getattr(env.cfg, "auto_reset", True)),
            "camera_runtime": camera_meta,
        },
        "static": static,
        "eval": {
            "episodes": episodes,
            "summary": summary,
            "terrain_mapping": terrain_mapping,
            "verify_camera": {
                "post_reset_depth_raw": (
                    _torch_summary(post_reset_depth_native) if post_reset_depth_native is not None else None
                ),
                "post_reset_depth_processed": (
                    _torch_summary(post_reset_depth_processed) if post_reset_depth_processed is not None else None
                ),
            },
        },
    }
    companion = args.out.with_suffix(".npz")
    np.savez(
        companion,
        episode_length=np.array([event["episode_length"] for event in episodes], dtype=np.int32),
        root_height_margin=np.array([event["root_height_margin"] for event in episodes], dtype=np.float64),
        primary_reason=np.array([event["primary_reason"] for event in episodes], dtype=object),
        terrain_name=np.array([event.get("terrain_name") or "" for event in episodes], dtype=object),
    )
    payload["companion_npz"] = str(companion)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"wrote {args.out}: side={side} {summary['completed_episodes']} episodes after warmup, "
        f"all_term={summary['termination_rate_per_1000_env_steps']:.3f}/1000 env-steps, "
        f"root_height={summary['root_height_count']} "
        f"({summary['root_height_rate_per_1000_env_steps']:.3f}/1000 env-steps), "
        f"completed_only_mean_len={summary['completed_episode_mean_length']} "
        f"terrain={terrain_mapping.get('allocation') or terrain_mapping.get('reason')}",
        flush=True,
    )
    if camera_patch is not None:
        camera_patch.restore()
    env.close()
    return 0


def run_probe(args: argparse.Namespace) -> int:
    _validate_run_args(args)
    if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
        sys.path.pop(0)

    import numpy as np
    import torch

    side = args.side
    instinctmj_root = Path(os.environ.get("INSTINCTMJ_ROOT", DEFAULT_INSTINCTMJ_ROOT))
    source_path: str | None = None
    task_id = OURS_TASK_ID if side == "ours" else INSTINCTMJ_TASK_ID
    spec = None
    compiled = None
    env_cfg = None

    if side == "ours":
        env, compiled, spec = _build_ours(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            disable_obs_noise=args.disable_obs_noise,
            disable_terminations=args.disable_terminations,
            friction_fixed=args.friction_fixed,
            terrain_num_cols=args.terrain_num_cols,
            actuator_lag=args.actuator_lag,
        )
        source_path = str(Path(__file__).resolve().parents[1])
    elif side == "instinctmj":
        read_instinctmj_train_registration(instinctmj_train_task_source(instinctmj_root))
        env, env_cfg = _build_instinctmj(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            disable_obs_noise=args.disable_obs_noise,
            disable_terminations=args.disable_terminations,
            friction_fixed=args.friction_fixed,
            root=instinctmj_root,
            terrain_num_cols=args.terrain_num_cols,
            actuator_lag=args.actuator_lag,
        )
        source_path = str(instinctmj_root)
    else:
        raise SystemExit(f"unknown side {side!r}")

    camera_patch = None
    if side == "ours" and getattr(args, "camera_semantics", CAMERA_SEMANTICS_NATIVE) != CAMERA_SEMANTICS_NATIVE:
        camera_patch = apply_camera_semantics(env, args.camera_semantics)

    wrapper = None
    policy = None
    env.reset()
    if args.mode == "live_policy":
        from instinct_rl.runners import OnPolicyRunner

        if side == "ours":
            from instinctlab.engines.mjlab import MjlabAdapter

            agent_cfg = compiled.agent_cfg
            policy_group = getattr(agent_cfg, "policy_observation_group", "policy")
            critic_group = getattr(agent_cfg, "critic_observation_group", "critic")
            from instinctlab.utils.wrappers.instinct_rl.mjlab_vecenv_wrapper import MjlabVecEnvWrapper

            wrapper = MjlabVecEnvWrapper(env, policy_group=policy_group, critic_group=critic_group)
        else:
            from instinct_mj.rl import InstinctRlVecEnvWrapper
            from instinct_mj.tasks.registry import load_instinct_rl_cfg

            agent_cfg = load_instinct_rl_cfg(INSTINCTMJ_TASK_ID)
            wrapper = InstinctRlVecEnvWrapper(
                env,
                policy_group=agent_cfg.policy_observation_group,
                critic_group=agent_cfg.critic_observation_group,
            )
        agent_cfg.device = args.device
        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)

    incoming_state = args.state_npz if args.state_npz is not None and args.state_npz.is_file() else None
    include_command = not getattr(args, "no_command_state", False)
    loaded_command_present = False
    command_apply_report: dict[str, Any] | None = None
    if incoming_state is not None:
        loaded = _load_state_npz(incoming_state)
        loaded_command_present = _command_snapshot_has_fields(loaded.get("command_state"))
        command_apply_report = _apply_state(env, loaded)

    camera_control_report: dict[str, Any] | None = None
    if args.camera_controlled:
        camera_control_report = _prime_camera_history_at_current_state(
            env, reference_ray_range=args.camera_reference_ray_range
        )

    action_term = _action_term(env)
    joint_order = _joint_names(env)
    static = _collect_static(env, side=side, task_id=task_id, source_path=source_path)
    if camera_patch is not None:
        static["camera"] = camera_patch.metadata()
    arrays: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []

    payload_state_npz: str | None = None
    captured_command_state: dict[str, Any] | None = None
    if args.mode == "dump":
        state = _capture_state(env, joint_order, include_command=include_command)
        captured_command_state = state.get("command_state")
        state_path = dump_state_output_path(
            out=args.out, state_npz=args.state_npz, incoming_exists=incoming_state is not None
        )
        _write_state_npz(state_path, state)
        payload_state_npz = str(state_path)
        steps.append(
            _collect_step_record(env, phase="post_reset", step_index=-1, action_term=action_term, store_arrays=arrays)
        )
    else:
        steps.append(
            _collect_step_record(env, phase="post_reset", step_index=-1, action_term=action_term, store_arrays=arrays)
        )
        actions_seq: np.ndarray | None = None
        if args.mode == "frozen_action":
            if args.action_npz is not None:
                actions_seq = _load_action_npz(args.action_npz, _action_target_names(env))
            else:
                actions_seq = None
        for step in range(args.steps):
            driving_raw = _raw_depth_tensor(env)
            driving_processed = _processed_depth_tensor(env)
            extra: dict[str, Any] = {}
            if args.mode == "live_policy":
                obs, _ = wrapper.get_observations()
                extra["policy_obs_summary"] = _torch_summary(obs)
                action = _inference_action(policy, obs)
                extra["policy_raw_action_summary"] = _torch_summary(action)
            elif args.mode == "frozen_action":
                if actions_seq is None:
                    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                else:
                    if actions_seq.ndim == 2:
                        row = actions_seq
                    else:
                        row = actions_seq[min(step, actions_seq.shape[0] - 1)]
                    action = torch.as_tensor(row, device=env.device, dtype=torch.float32)
                    if action.ndim == 1:
                        action = action.unsqueeze(0).expand(env.num_envs, -1)
            else:
                raise RuntimeError(args.mode)
            if wrapper is not None:
                wrapper.step(action)
            else:
                env.step(action)
            step_record = _collect_step_record(
                env,
                phase="post_control_step",
                step_index=step,
                action_term=action_term,
                store_arrays=arrays,
                depth_raw=driving_raw,
                depth_processed=driving_processed,
            )
            step_record.update(extra)
            steps.append(step_record)
            timeouts = (
                env.termination_manager.time_outs
                if hasattr(env.termination_manager, "time_outs") and not callable(env.termination_manager.time_outs)
                else env.termination_manager.time_outs()
            )
            if not args.disable_terminations and bool(env.termination_manager.terminated.any() or timeouts.any()):
                step_record["note"] = (
                    "episode ended; later steps not executed. "
                    "auto_reset is False so qpos/root are the terminal state, not a respawn."
                )
                break

    commit = _repo_commit(Path(source_path)) if source_path else None
    cmd_status = command_state_status(
        captured=include_command and _command_snapshot_has_fields(captured_command_state),
        loaded=incoming_state is not None,
        loaded_present=loaded_command_present,
    )
    metadata: dict[str, Any] = {
        "side": side,
        "mode": args.mode,
        "seed": args.seed,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "task_id": task_id,
        "commit": commit,
        "source_path": source_path,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "disable_obs_noise": args.disable_obs_noise,
        "disable_terminations": args.disable_terminations,
        "friction_fixed": args.friction_fixed,
        "state_npz": payload_state_npz,
        "state_npz_loaded": str(incoming_state) if incoming_state is not None else None,
        "camera_semantics": getattr(args, "camera_semantics", CAMERA_SEMANTICS_NATIVE),
        "command_state": cmd_status,
        "command_state_schema": COMMAND_STATE_SCHEMA if include_command else None,
        "command_name": COMMAND_NAME,
        "terrain_num_cols_requested": args.terrain_num_cols,
        "camera_controlled": bool(args.camera_controlled),
        "camera_reference_ray_range": bool(args.camera_reference_ray_range),
        "actuator_lag": args.actuator_lag,
    }
    if captured_command_state is not None:
        metadata["command_state_fields"] = sorted(captured_command_state.get("fields", {}).keys())
        metadata["command_state_missing_on_capture"] = list(captured_command_state.get("missing_on_term", []))
    if command_apply_report is not None:
        metadata["command_state_apply"] = command_apply_report
    if camera_control_report is not None:
        metadata["camera_control"] = camera_control_report
    payload: dict[str, Any] = {
        "metadata": metadata,
        "static": static,
        "steps": steps,
    }
    companion = args.out.with_suffix(".npz")
    if arrays:
        np.savez(companion, **arrays)
        payload["companion_npz"] = str(companion)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(steps)} step records, companion={companion.name})", flush=True)
    if camera_patch is not None:
        camera_patch.restore()
    env.close()
    return 0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="collect one rollout (default)")
    run.add_argument("--side", required=True, choices=("ours", "instinctmj"))
    run.add_argument("--mode", required=True, choices=("dump", "frozen_action", "live_policy", "policy_eval"))
    run.add_argument("--checkpoint", type=Path, default=None)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--num-envs", type=int, default=2)
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--state-npz", type=Path, default=None)
    run.add_argument("--action-npz", type=Path, default=None)
    run.add_argument(
        "--no-command-state",
        action="store_true",
        help=(
            "Dump only robot kinematics into state.npz (no PoseVelocityCommand snapshot). "
            "Compare will mark command_dependent_mdp_parity=false."
        ),
    )
    run.add_argument("--disable-obs-noise", action="store_true")
    run.add_argument(
        "--disable-terminations",
        action="store_true",
        help=(
            "Null every termination term so episodes cannot end. The probe also forces "
            "auto_reset=False, so a post-step root/qpos sample is the same control step "
            "as termination_manager flags (not a respawn)."
        ),
    )
    run.add_argument(
        "--friction-fixed",
        action="store_true",
        help=(
            "Disable startup physics_material DR. Does not write a numeric friction; both sides keep the MJCF default."
        ),
    )
    run.add_argument(
        "--terrain-num-cols",
        type=int,
        default=None,
        help="Force the declared terrain width. Use 10 for InstinctMJ's one-column-per-type curriculum mesh.",
    )
    run.add_argument(
        "--camera-controlled",
        action="store_true",
        help=(
            "After state injection, reset camera/history, force visual delay to zero, then perform exactly "
            "one sense and one observation compute."
        ),
    )
    run.add_argument(
        "--camera-reference-ray-range",
        action="store_true",
        help=(
            "Controlled diagnostic: force the ray-cast max distance to image_plane_max. Production MJLab "
            "now already uses this InstinctMJ-compatible range, so this normally records a no-op."
        ),
    )
    run.add_argument(
        "--actuator-lag",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Controlled plant probe: force every builtin actuator to this deterministic physics-step lag.",
    )
    run.add_argument(
        "--camera-semantics",
        choices=CAMERA_SEMANTICS_CHOICES,
        default=CAMERA_SEMANTICS_NATIVE,
        help=(
            "Ours-side camera semantics for causal A/B. "
            "native=production geom_groups_min_distance_hop; "
            "instinctmj_geom_groups=alias verifying native already matches InstinctMJ."
        ),
    )
    run.add_argument(
        "--eval-warmup-steps",
        type=int,
        default=0,
        help="policy_eval: exclude episodes ending before this control step from summary.",
    )
    run.add_argument(
        "--verify-camera",
        action="store_true",
        help="policy_eval: record post-reset depth digest proving camera semantics are live.",
    )

    compare = sub.add_parser("compare", help="diff two probe JSON outputs")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--qpos-tol", type=float, default=DEFAULT_THRESHOLDS["qpos"])
    compare.add_argument("--root-z-tol", type=float, default=DEFAULT_THRESHOLDS["root_z"])
    compare.add_argument("--qfrc-tol", type=float, default=DEFAULT_THRESHOLDS["qfrc"])
    compare.add_argument("--action-tol", type=float, default=DEFAULT_THRESHOLDS["action"])
    compare.add_argument("--depth-tol", type=float, default=DEFAULT_THRESHOLDS["depth"])

    analyze = sub.add_parser("analyze-eval", help="factory vs policy effects from policy_eval JSON arms")
    analyze.add_argument("arms", type=Path, nargs="+")

    # Allow omitting the ``run`` subcommand for backward-compatible ergonomics.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"run", "compare", "analyze-eval", "-h", "--help"}:
        argv = ["run", *argv]
    ns = parser.parse_args(argv)
    if ns.command is None:
        parser.error("the following arguments are required: command (run or compare)")
    return ns


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.command == "analyze-eval":
        arms = [json.loads(path.read_text(encoding="utf-8")) for path in args.arms]
        report = analyze_policy_eval_2x2(arms)
        print(json.dumps(report, indent=1))
        return 0
    if args.command == "compare":
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
        report = compare_rollout_payloads(
            left,
            right,
            thresholds={
                "qpos": args.qpos_tol,
                "root_z": args.root_z_tol,
                "qfrc": args.qfrc_tol,
                "action": args.action_tol,
                "depth": args.depth_tol,
            },
        )
        print(json.dumps(report, indent=1))
        return 0 if report["passed"] else 1
    if args.mode == "policy_eval":
        return run_policy_eval(args)
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
