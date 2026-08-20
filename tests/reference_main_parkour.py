"""Reading main's parkour target AMP config without importing it.

The legacy Isaac task is still on ``main``; this module asks git for the source and reads literals
off the syntax tree. Anything that needs evaluation — entity configs, function objects — is named,
not half-interpreted, so a fact reported here is a fact the file states.

G1-specific overrides from ``ShoeConfigMixin`` and ``G1ParkourRoughEnvCfg`` are merged into the
effective tables, because training never ran the base ``ParkourEnvCfg`` alone.
"""

from __future__ import annotations

import ast
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

PARKOUR_CFG = "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
G1_CFG = "source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py"
AGENT_CFG = "source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py"

# ShoeConfigMixin.apply_shoe_config() writes these over the base parkour_env_cfg values.
G1_SHOE_HEIGHT_OFFSET = 0.058
G1_SHOE_VOLUME_Z = (-0.063, -0.023)
G1_DELAYED_ACTUATORS = "beyondmimic_g1_29dof_delayed_actuators"
G1_SHOE_URDF_SUFFIX = "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
G1_SPAWN_Z = 0.9
G1_MERGE_FIXED_JOINTS = True


def _git_show(path: str) -> str:
    shown = subprocess.run(("git", "show", f"main:{path}"), cwd=REPO, capture_output=True, text=True)
    if shown.returncode != 0:
        raise FileNotFoundError(f"{path} is not on main: {shown.stderr.strip()}")
    return shown.stdout


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        if isinstance(node, ast.Dict):
            return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values, strict=True) if k is not None}
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(_literal(element) for element in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _literal(node.operand)
            return -inner if isinstance(inner, (int, float)) else f"<-{inner}>"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sqrt":
            arg = _literal(node.args[0])
            if isinstance(arg, (int, float)):
                return arg**0.5
            return f"sqrt({arg})"
        if isinstance(node, ast.Attribute):
            return ast.unparse(node)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return f"<call {ast.unparse(node.func).rsplit('.', 1)[-1]}>"
        return f"<{type(node).__name__}>"


def _kwargs(call: ast.Call) -> dict[str, Any]:
    return {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg is not None}


def _func_name(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "func":
            return ast.unparse(kw.value).rsplit(".", 1)[-1]
    return ""


def _class_assignments(module: ast.Module, class_name: str) -> dict[str, ast.Call]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out: dict[str, ast.Call] = {}
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and isinstance(item.value, ast.Call)
                ):
                    out[item.targets[0].id] = item.value
            return out
    raise LookupError(f"no class {class_name!r} in module")


def _obs_group(module: ast.Module, group_class: str) -> dict[str, ast.Call]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg":
            for inner in node.body:
                if isinstance(inner, ast.ClassDef) and inner.name == group_class:
                    return {
                        item.targets[0].id: item.value
                        for item in inner.body
                        if isinstance(item, ast.Assign)
                        and len(item.targets) == 1
                        and isinstance(item.targets[0], ast.Name)
                        and isinstance(item.value, ast.Call)
                    }
    raise LookupError(f"no observation group {group_class!r}")


@lru_cache(maxsize=1)
def _parkour_module() -> ast.Module:
    return ast.parse(_git_show(PARKOUR_CFG))


def reward_names() -> frozenset[str]:
    return frozenset(_class_assignments(_parkour_module(), "G1Rewards"))


def reward_weights() -> dict[str, float]:
    return {
        name: _kwargs(call).get("weight") for name, call in _class_assignments(_parkour_module(), "G1Rewards").items()
    }


def reward_functions() -> dict[str, str]:
    return {name: _func_name(call) for name, call in _class_assignments(_parkour_module(), "G1Rewards").items()}


def reward_params() -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for name, call in _class_assignments(_parkour_module(), "G1Rewards").items():
        raw = dict(_kwargs(call).get("params") or {})
        if name == "feet_at_plane":
            raw["height_offset"] = G1_SHOE_HEIGHT_OFFSET
        params[name] = raw
    return params


def termination_names() -> frozenset[str]:
    return frozenset(_class_assignments(_parkour_module(), "TerminationsCfg"))


def event_names() -> frozenset[str]:
    return frozenset(_class_assignments(_parkour_module(), "EventCfg"))


_OBS_GROUPS = {
    "policy": "PolicyCfg",
    "critic": "CriticCfg",
    "amp_policy": "AmpPolicyStateObsCfg",
    "amp_reference": "AmpReferenceStateObsCfg",
}

# Shipped dataset that main's yaml filter and our clip path both resolve to.
SHIPPED_MOTION_YAML = Path("/root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run.yaml")
SHIPPED_MOTION_NPZ = Path(
    "/root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
)
MAIN_TRAIN = "scripts/instinct_rl/train.py"
MAIN_WRAPPER = "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py"
ISAAC_OBS_TERM_CFG = Path("/root/IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py")


def observation_order(group: str) -> tuple[str, ...]:
    return tuple(_obs_group(_parkour_module(), _OBS_GROUPS[group]).keys())


def observation_scales(group: str = "policy") -> dict[str, float | None]:
    scales: dict[str, float | None] = {}
    for name, call in _obs_group(_parkour_module(), _OBS_GROUPS[group]).items():
        scales[name] = _kwargs(call).get("scale")
    return scales


def observation_noise(group: str = "policy") -> dict[str, tuple[float, float] | None]:
    """Uniform noise bounds, or None when the term declares no noise / noise=None."""
    bounds: dict[str, tuple[float, float] | None] = {}
    for name, call in _obs_group(_parkour_module(), _OBS_GROUPS[group]).items():
        bounds[name] = None
        for kw in call.keywords:
            if kw.arg != "noise":
                continue
            if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                bounds[name] = None
            elif isinstance(kw.value, ast.Call):
                noise = _kwargs(kw.value)
                if "n_min" in noise and "n_max" in noise:
                    bounds[name] = (noise["n_min"], noise["n_max"])
    return bounds


def observation_history(group: str = "policy") -> dict[str, int]:
    history: dict[str, int] = {}
    for name, call in _obs_group(_parkour_module(), _OBS_GROUPS[group]).items():
        history[name] = _kwargs(call).get("history_length", 0)
    return history


def observation_clip(group: str = "policy") -> dict[str, Any]:
    return {name: _kwargs(call).get("clip") for name, call in _obs_group(_parkour_module(), _OBS_GROUPS[group]).items()}


def observation_flatten_history(group: str = "policy") -> dict[str, bool | None]:
    """True when main sets flatten_history_dim; None when it relies on Isaac Lab's default."""
    out: dict[str, bool | None] = {}
    for name, call in _obs_group(_parkour_module(), _OBS_GROUPS[group]).items():
        out[name] = _kwargs(call).get("flatten_history_dim")
    return out


def observation_functions(group: str = "policy") -> dict[str, str]:
    return {name: _func_name(call) for name, call in _obs_group(_parkour_module(), _OBS_GROUPS[group]).items()}


def _term_asset_cfg_kwargs(call: ast.Call) -> dict[str, Any] | None:
    """SceneEntityCfg keywords inside a term's params, or None if the term names no asset."""
    for kw in call.keywords:
        if kw.arg != "params" or not isinstance(kw.value, ast.Dict):
            continue
        for key, value in zip(kw.value.keys, kw.value.values, strict=True):
            if key is not None and _literal(key) == "asset_cfg" and isinstance(value, ast.Call):
                return _kwargs(value)
    return None


def observation_joint_names(group: str, term: str) -> tuple[str, ...] | None:
    """Named joints on a term's asset_cfg, or None when main left the selector implicit."""
    call = _obs_group(_parkour_module(), _OBS_GROUPS[group])[term]
    cfg = _term_asset_cfg_kwargs(call)
    if cfg is None:
        return None
    names = cfg.get("joint_names")
    if names is None:
        return None
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def observation_preserve_order(group: str, term: str) -> bool | None:
    call = _obs_group(_parkour_module(), _OBS_GROUPS[group])[term]
    cfg = _term_asset_cfg_kwargs(call)
    if cfg is None:
        return None
    return cfg.get("preserve_order")


def action_kwargs() -> dict[str, Any]:
    return _kwargs(_class_assignments(_parkour_module(), "ActionsCfg")["joint_pos"])


def delayed_depth_params() -> dict[str, Any]:
    """Params of policy depth_image on main (sensor-side history + delayed_visualizable_image)."""
    call = _obs_group(_parkour_module(), "PolicyCfg")["depth_image"]
    return dict(_kwargs(call).get("params") or {})


def camera_ray_alignment() -> str | None:
    """Literal on SceneCfg.camera, which both engines' grouped cameras ignore at runtime."""
    for node in _parkour_module().body:
        if isinstance(node, ast.ClassDef) and node.name == "SceneCfg":
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "camera"
                    and isinstance(item.value, ast.Call)
                ):
                    return _kwargs(item.value).get("ray_alignment")
    raise LookupError("SceneCfg.camera missing on main")


def motion_reference_source() -> dict[str, Any]:
    """Fields of the MotionReferenceManagerCfg assigned in the G1 AMP file."""
    g1 = ast.parse(_git_show(G1_CFG))
    source: dict[str, Any] = {}
    for node in g1.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "motion_reference_cfg" for t in node.targets
        ):
            if isinstance(node.value, ast.Call):
                source.update(_kwargs(node.value))
        if isinstance(node, ast.ClassDef) and node.name == "AmassMotionCfg":
            fields: dict[str, Any] = {}
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    fields[item.targets[0].id] = ast.unparse(item.value)
            source["amass"] = fields
    return source


def train_script_facts() -> dict[str, Any]:
    """What main's Isaac-only trainer actually writes, read off the syntax tree."""
    tree = ast.parse(_git_show(MAIN_TRAIN))
    src = ast.unparse(tree)
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = ast.unparse(node.func) if isinstance(node.func, ast.Attribute) else ""
        if not func.endswith("add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if flag in {"--num_envs", "--seed", "--max_iterations"}:
            for kw in node.keywords:
                if kw.arg == "default":
                    defaults[str(flag)] = _literal(kw.value)
    return {
        "num_envs_default": defaults.get("--num_envs"),
        "seed_default": defaults.get("--seed"),
        "max_iterations_default": defaults.get("--max_iterations"),
        "sets_env_seed_from_agent": "env_cfg.seed = agent_cfg.seed" in src,
        "calls_runner_load": "runner.load(" in src,
        "sets_tf32": "allow_tf32 = True" in src,
        "init_at_random_ep_len": "init_at_random_ep_len" in src,
        "wrapper": "InstinctRlVecEnvWrapper" in src,
    }


def isaac_observation_term_flatten_history_default() -> bool:
    """Isaac Lab's ObservationTermCfg default, read from the installed source."""
    text = ISAAC_OBS_TERM_CFG.read_text()
    return "flatten_history_dim: bool = True" in text


def wrapper_sets_missing_step_dict() -> bool:
    """Whether the working-tree wrapper fills infos['step'] when Isaac never wrote it."""
    return 'setdefault("step", {})' in Path(REPO, MAIN_WRAPPER).read_text()


def main_wrapper_sets_missing_step_dict() -> bool:
    return 'setdefault("step", {})' not in _git_show(MAIN_WRAPPER)


def command_params() -> dict[str, Any]:
    for node in _parkour_module().body:
        if isinstance(node, ast.ClassDef) and node.name == "CommandsCfg":
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "base_velocity"
                ):
                    return _kwargs(item.value)
    raise LookupError("base_velocity command missing on main")


def _constant(node: ast.AST) -> Any:
    """Evaluate a literal, or an arithmetic expression over literals such as ``10 * 2**15``."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(node, ast.BinOp):
        left, right = _constant(node.left), _constant(node.right)
        for op, apply in ((ast.Pow, lambda a, b: a**b), (ast.Mult, lambda a, b: a * b)):
            if isinstance(node.op, op):
                return apply(left, right)
    raise ValueError(f"not a constant expression: {ast.unparse(node)}")


SIM_PARAM_TARGETS = {
    "decimation": "self.decimation",
    "episode_length_s": "self.episode_length_s",
    "physics_dt": "self.sim.dt",
    "gpu_max_rigid_patch_count": "self.sim.physx.gpu_max_rigid_patch_count",
    "gpu_collision_stack_size": "self.sim.physx.gpu_collision_stack_size",
}


def sim_params() -> dict[str, Any]:
    """Values assigned in ``ParkourEnvCfg.__post_init__`` on main.

    Read off the assignment nodes rather than by matching substrings in the unparsed source.
    ``ast.unparse`` normalises ``2**29`` to ``2 ** 29``, so the substring probes this used to run
    missed every power of two and returned ``None`` for it. ``gpu_collision_stack_size`` was one:
    its drift row then reported main's value as the string ``"None"`` and still passed, because the
    row only has to differ from ours. A parser that answers "absent" when it cannot parse is worse
    than one that raises, so unknown expressions now raise instead.
    """
    for node in _parkour_module().body:
        if isinstance(node, ast.ClassDef) and node.name == "ParkourEnvCfg":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                    assigned = {
                        ast.unparse(target): _constant(stmt.value)
                        for stmt in ast.walk(item)
                        if isinstance(stmt, ast.Assign)
                        for target in stmt.targets
                        if ast.unparse(target) in SIM_PARAM_TARGETS.values()
                    }
                    return {name: assigned.get(target) for name, target in SIM_PARAM_TARGETS.items()}
            raise LookupError("ParkourEnvCfg.__post_init__ missing")
    raise LookupError("ParkourEnvCfg missing on main")


def g1_robot_overrides() -> dict[str, Any]:
    """What G1ParkourEnvCfg adds on top of the shared parkour file."""
    g1 = ast.parse(_git_show(G1_CFG))
    overrides: dict[str, Any] = {
        "spawn_z": G1_SPAWN_Z,
        "merge_fixed_joints": G1_MERGE_FIXED_JOINTS,
        "actuators": G1_DELAYED_ACTUATORS,
        "shoe_urdf": G1_SHOE_URDF_SUFFIX,
        "volume_z_min": G1_SHOE_VOLUME_Z[0],
        "volume_z_max": G1_SHOE_VOLUME_Z[1],
        "feet_at_plane_height_offset": G1_SHOE_HEIGHT_OFFSET,
    }
    for node in g1.body:
        if isinstance(node, ast.ClassDef) and node.name == "G1ParkourRoughEnvCfg":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                    src = ast.unparse(item)
                    overrides["uses_delayed_actuators"] = G1_DELAYED_ACTUATORS in src
    return overrides


def uses_instinct_rl_env() -> bool:
    shown = subprocess.run(
        ("git", "show", "main:source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py"),
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return "InstinctRlEnv" in shown.stdout


def uses_multi_reward_cfg() -> bool:
    return "MultiRewardCfg" in _git_show(PARKOUR_CFG) and "rewards: G1Rewards" in _git_show(PARKOUR_CFG)
