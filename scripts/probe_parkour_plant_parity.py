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

InstinctMJ **must** use ``/root/InstinctMJ/.venv/bin/python`` (the ``instinct-train``
interpreter) and the training GPU convention ``CUDA_VISIBLE_DEVICES=<n> --device cuda:0``.
The current process's site-packages are the wrong stack; a missing ``coacd`` there is
not a missing project dependency. Passing ``--device cuda:1`` inside that venv is a
Warp device mismatch, not a missing package.

Engine packages are imported only inside ``run`` (never at module import time), so
``--help`` and offline tests stay engine-free.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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

    return {
        "steps_compared": len(steps_l),
        "thresholds": thresholds,
        "per_step": per_step_report,
        "first_consecutive_two_step_exceedance": first_failure,
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


def _build_ours(
    *,
    num_envs: int,
    device: str,
    seed: int,
    disable_obs_noise: bool,
    disable_terminations: bool,
    friction_fixed: bool,
):
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    spec = parkour_target_g1()
    MjlabAdapter.bootstrap(argparse.Namespace(device=device))
    compiled = MjlabAdapter().compile(spec, num_envs=num_envs, device=device)
    compiled.env_cfg.seed = seed
    if disable_obs_noise:
        _silence_observation_noise(compiled.env_cfg)
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
    if disable_obs_noise:
        _silence_observation_noise(env_cfg)
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


def _capture_state(env, joint_order: list[str]) -> dict[str, Any]:
    robot = env.scene["robot"]
    data = robot.data
    native_ids = _native_indices_for_names(env, joint_order)
    qpos = data.joint_pos[:, native_ids].detach().cpu().numpy()
    qvel = data.joint_vel[:, native_ids].detach().cpu().numpy()
    root_pos = (data.root_link_pos_w - env.scene.env_origins).detach().cpu().numpy()
    root_quat = data.root_link_quat_w.detach().cpu().numpy()
    root_lin = data.root_link_lin_vel_w.detach().cpu().numpy()
    root_ang = data.root_link_ang_vel_w.detach().cpu().numpy()
    return {
        "joint_names": list(joint_order),
        "action_target_names": _action_target_names(env),
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin,
        "root_ang_vel": root_ang,
        "joint_pos": qpos,
        "joint_vel": qvel,
    }


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


def _apply_state(env, state: dict[str, Any]) -> None:
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
    _refresh_sensors_and_obs(env)


def _write_state_npz(path: Path, state: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        joint_names=np.array(state["joint_names"], dtype=object),
        action_target_names=np.array(state.get("action_target_names", []), dtype=object),
        root_pos=state["root_pos"],
        root_quat=state["root_quat"],
        root_lin_vel=state["root_lin_vel"],
        root_ang_vel=state["root_ang_vel"],
        joint_pos=state["joint_pos"],
        joint_vel=state["joint_vel"],
    )


def _load_state_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=True) as archive:
        return {
            "joint_names": [str(x) for x in archive["joint_names"].tolist()],
            "action_target_names": [str(x) for x in archive.get("action_target_names", []).tolist()],
            "root_pos": archive["root_pos"],
            "root_quat": archive["root_quat"],
            "root_lin_vel": archive["root_lin_vel"],
            "root_ang_vel": archive["root_ang_vel"],
            "joint_pos": archive["joint_pos"],
            "joint_vel": archive["joint_vel"],
        }


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
    cfg = getattr(sensor, "cfg", None)
    info: dict[str, Any] = {"available": True, "sensor_name": CAMERA_NAME}
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
    mask = getattr(sensor, "_allowed_geom_mask", None)
    if mask is not None:
        import torch

        info["allowed_geom_count"] = int(mask.sum().item()) if isinstance(mask, torch.Tensor) else int(mask.sum())
        info["camera_filter"] = "body_mesh_mask_with_hop"
        info["hop_max"] = 6
    else:
        groups = getattr(cfg, "include_geom_groups", None) if cfg is not None else None
        info["camera_filter"] = "geom_groups"
        info["include_geom_groups"] = list(groups) if groups is not None else None
        info["mesh_prim_paths"] = list(getattr(cfg, "mesh_prim_paths", []) or [])
    return info


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
    too_far = image > float(far) if far is not None else torch.zeros_like(image, dtype=torch.bool)
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
    if args.mode == "live_policy" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --mode live_policy")
    if args.mode != "live_policy" and args.checkpoint is not None:
        raise SystemExit(f"--checkpoint is not used with --mode {args.mode}")
    if args.out is None:
        raise SystemExit("--out is required")
    if args.mode == "dump" and args.action_npz is not None:
        raise SystemExit("--action-npz is not used with --mode dump")
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")


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
        )
        source_path = str(instinctmj_root)
    else:
        raise SystemExit(f"unknown side {side!r}")

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
    if incoming_state is not None:
        _apply_state(env, _load_state_npz(incoming_state))

    action_term = _action_term(env)
    joint_order = _joint_names(env)
    static = _collect_static(env, side=side, task_id=task_id, source_path=source_path)
    arrays: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []

    payload_state_npz: str | None = None
    if args.mode == "dump":
        state = _capture_state(env, joint_order)
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
    payload: dict[str, Any] = {
        "metadata": {
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
        },
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
    env.close()
    return 0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="collect one rollout (default)")
    run.add_argument("--side", required=True, choices=("ours", "instinctmj"))
    run.add_argument("--mode", required=True, choices=("dump", "frozen_action", "live_policy"))
    run.add_argument("--checkpoint", type=Path, default=None)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--num-envs", type=int, default=2)
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--state-npz", type=Path, default=None)
    run.add_argument("--action-npz", type=Path, default=None)
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

    compare = sub.add_parser("compare", help="diff two probe JSON outputs")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--qpos-tol", type=float, default=DEFAULT_THRESHOLDS["qpos"])
    compare.add_argument("--root-z-tol", type=float, default=DEFAULT_THRESHOLDS["root_z"])
    compare.add_argument("--qfrc-tol", type=float, default=DEFAULT_THRESHOLDS["qfrc"])
    compare.add_argument("--action-tol", type=float, default=DEFAULT_THRESHOLDS["action"])
    compare.add_argument("--depth-tol", type=float, default=DEFAULT_THRESHOLDS["depth"])

    # Allow omitting the ``run`` subcommand for backward-compatible ergonomics.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"run", "compare", "-h", "--help"}:
        argv = ["run", *argv]
    ns = parser.parse_args(argv)
    if ns.command is None:
        parser.error("the following arguments are required: command (run or compare)")
    return ns


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
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
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
