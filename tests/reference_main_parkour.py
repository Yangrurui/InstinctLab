"""Reading main's parkour target AMP config without importing it.

The legacy Isaac task is still on ``main``; this module asks git for the source and reads literals
off the syntax tree. Anything that needs evaluation — entity configs, function objects — is named,
not half-interpreted, so a fact reported here is a fact the file states.

G1-specific overrides from ``ShoeConfigMixin`` and ``G1ParkourRoughEnvCfg`` are merged into the
effective tables, because training never ran the base ``ParkourEnvCfg`` alone.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

PARKOUR_CFG = "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
G1_CFG = "source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py"
G1_INIT = "source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py"
AGENT_CFG = "source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py"
ASSETS_CFG = "source/instinctlab/instinctlab/assets/unitree_g1.py"

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


MAIN_PARKOUR_REWARDS = "source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py"
MAIN_POSE_VELOCITY = "source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py"
ISAAC_MDP_REWARDS = Path("/root/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py")
ISAAC_MDP_OBSERVATIONS = Path("/root/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py")

_ROOT_VELOCITY_SPELLINGS = ("root_lin_vel_b", "root_link_lin_vel_b", "root_com_lin_vel_b")


def _velocity_spelling_in(source: str, qualname: str) -> str:
    """Which root linear-velocity attribute the named function or method reads.

    Ambiguity raises rather than picking one. A term that reads two spellings, or none, is not a
    fact this table can record, and returning a default would put a wrong value into a drift row
    that nobody would ever see fail.
    """
    *owner, name = qualname.split(".")
    tree = ast.parse(source)
    scope: ast.AST = tree
    if owner:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == owner[0]:
                scope = node
                break
        else:
            raise LookupError(f"no class {owner[0]!r} in the reference source")
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            body = ast.unparse(node)
            hits = {spelling for spelling in _ROOT_VELOCITY_SPELLINGS if f".{spelling}" in body}
            if len(hits) != 1:
                raise LookupError(f"{qualname} reads {sorted(hits) or 'no'} root velocity spelling")
            return hits.pop()
    raise LookupError(f"no function {qualname!r} in the reference source")


def velocity_frame_spellings() -> dict[str, str]:
    """The root-velocity attribute each of main's velocity-frame call sites reads.

    ``track_lin_vel_xy_exp`` resolves into Isaac Lab rather than main, because main takes it from
    ``isaaclab.envs.mdp``; the other two are main's own files. All three are read, not transcribed,
    so the KNOWN_DRIFTS rows fail if main's frame ever changes.
    """
    return {
        "track_lin_vel_xy_exp": _velocity_spelling_in(ISAAC_MDP_REWARDS.read_text(), "track_lin_vel_xy_exp"),
        "dont_wait": _velocity_spelling_in(_git_show(MAIN_PARKOUR_REWARDS), "dont_wait"),
        "command_metrics": _velocity_spelling_in(_git_show(MAIN_POSE_VELOCITY), "_update_metrics"),
        "base_lin_vel": _velocity_spelling_in(ISAAC_MDP_OBSERVATIONS.read_text(), "base_lin_vel"),
    }


MAIN_VOLUME_POINTS = "source/instinctlab/instinctlab/sensors/volume_points/volume_points.py"


def volume_points_point_velocity() -> dict[str, bool]:
    """How main builds a volume point's velocity, read off ``_refresh_volume_points``.

    It takes the body velocity from PhysX ``get_velocities()`` -- a centre-of-mass quantity -- and
    then adds ``ω × (p − pos_w)``, where ``pos_w`` is the *link origin* from ``get_transforms()``.
    Mixing the two leaves every point off by ``ω × (origin − com)``. Recorded as two separate
    facts so the drift row fails if either half moves, rather than on a single substring that
    could match for the wrong reason.
    """
    tree = ast.parse(_git_show(MAIN_VOLUME_POINTS))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_refresh_volume_points":
            body = ast.unparse(node)
            return {
                "velocity_from_physx_com": "get_velocities()" in body,
                "lever_from_link_origin": "points_pos_w - self._data.pos_w" in body.replace("\n", " "),
            }
    raise LookupError("main's volume points sensor has no _refresh_volume_points")


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


def registered_env_cfg_class() -> str:
    """The env cfg class the trained task id actually points at."""
    tree = ast.parse(_git_show(G1_INIT))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _func_name_of(node) == "register"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        task_id = kwargs.get("id")
        if not (isinstance(task_id, ast.Constant) and task_id.value == "Instinct-Parkour-Target-Amp-G1-v0"):
            continue
        entry = kwargs.get("kwargs")
        if not isinstance(entry, ast.Dict):
            break
        for key, value in zip(entry.keys, entry.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "env_cfg_entry_point":
                return _literal_parts(value).rsplit(":", 1)[-1]
    raise LookupError("Instinct-Parkour-Target-Amp-G1-v0 does not name an env_cfg_entry_point on main")


def _literal_parts(node: ast.AST) -> str:
    """The constant text of a string node; f-string holes contribute nothing.

    main writes the entry point as ``f"{task_entry}.…:G1ParkourEnvCfg"``, and only
    the part after the colon is needed.
    """
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value for part in node.values if isinstance(part, ast.Constant))
    raise TypeError(f"cannot read a string out of {type(node).__name__}")


def _func_name_of(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")


def _class_named(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(f"{name} is not defined on main in {G1_CFG}")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _executed_statements(module: ast.Module, class_name: str) -> list[ast.stmt]:
    """``__post_init__`` flattened in execution order, following super() and mixin calls.

    Only the two call shapes main uses are expanded. Anything else raises rather
    than being skipped, because a silently unexpanded call is how a later
    override goes unnoticed -- which is the exact bug this reader exists to catch.
    """
    cls = _class_named(module, class_name)
    post_init = _method(cls, "__post_init__")
    if post_init is None:
        base_names = [b.id for b in cls.bases if isinstance(b, ast.Name)]
        if not base_names:
            raise LookupError(f"{class_name} has neither __post_init__ nor a resolvable base")
        return _executed_statements(module, base_names[0])

    out: list[ast.stmt] = []
    for stmt in post_init.body:
        call = stmt.value if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) else None
        if call is None:
            out.append(stmt)
            continue
        name = _func_name_of(call)
        if name == "__post_init__" and isinstance(call.func, ast.Attribute):
            bases = [b.id for b in cls.bases if isinstance(b, ast.Name)]
            if not bases:
                raise LookupError(f"{class_name} calls super().__post_init__ but names no base")
            # A base from another module (ParkourEnvCfg) is not followed. Safe only
            # because a later in-module statement replaces self.scene.robot outright;
            # effective_robot_actuators() raises if that replacement is not found.
            if any(isinstance(n, ast.ClassDef) and n.name == bases[0] for n in module.body):
                out.extend(_executed_statements(module, bases[0]))
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            if call.func.value.id != "self":
                out.append(stmt)
                continue
            owner = next(
                (c for c in module.body if isinstance(c, ast.ClassDef) and _method(c, name) is not None),
                None,
            )
            if owner is None:
                raise LookupError(f"{class_name}.__post_init__ calls self.{name}(), which is not in {G1_CFG}")
            method = _method(owner, name)
            assert method is not None
            out.extend(method.body)
        else:
            out.append(stmt)
    return out


def _module_symbol_root(module: ast.Module, name: str) -> str:
    """Follow ``X = copy.deepcopy(Y)`` / ``X = Y`` at module level back to its origin."""
    seen: set[str] = set()
    current = name
    while current not in seen:
        seen.add(current)
        source = None
        for node in module.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name) and target.id == current):
                continue
            value = node.value
            if isinstance(value, ast.Call) and _func_name_of(value) == "deepcopy" and value.args:
                source = value.args[0]
            else:
                source = value
        if isinstance(source, ast.Name):
            current = source.id
            continue
        return current
    return current


def effective_robot_actuators() -> dict[str, Any]:
    """The actuator table the *registered* task ends up with, not the one it declares.

    ``G1ParkourEnvCfg`` assigns delayed actuators onto ``self.scene.robot`` and then
    calls ``apply_shoe_config()``, which replaces ``self.scene.robot`` wholesale with
    a module-level copy taken before that assignment. Asking whether the delayed
    table is *mentioned* in the file therefore answers the wrong question.
    """
    module = ast.parse(_git_show(G1_CFG))
    statements = _executed_statements(module, registered_env_cfg_class())

    robot_symbol: str | None = None
    actuator_table: str | None = None
    for stmt in statements:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = ast.unparse(stmt.targets[0])
        if target == "self.scene.robot":
            value = stmt.value
            base = value.func.value if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) else value
            robot_symbol = base.id if isinstance(base, ast.Name) else ast.unparse(base)
            actuator_table = None  # a wholesale replacement drops the table set before it
        elif target == "self.scene.robot.actuators" and isinstance(stmt.value, ast.Name):
            actuator_table = stmt.value.id

    if actuator_table is None:
        if robot_symbol is None:
            raise LookupError("main's registered parkour cfg never assigns self.scene.robot")
        root = _module_symbol_root(module, robot_symbol)
        actuator_table = _articulation_actuators(root)

    return {
        "table": actuator_table,
        "delayed": actuator_table == G1_DELAYED_ACTUATORS,
        "robot_symbol": robot_symbol,
        "declared_in_base": G1_DELAYED_ACTUATORS in ast.unparse(module),
    }


def actuator_joint_velocity_limits(joint_names: Sequence[str]) -> dict[str, float]:
    """Per-joint velocity limit from the actuator table the registered task actually runs.

    main never spells ``velocity_limit``; it sets ``velocity_limit_sim`` and lets Isaac Lab
    fall back (``ActuatorBase.__init__``: ``velocity_limit = _parse_joint_parameter(
    cfg.velocity_limit, self.velocity_limit_sim)``), which is what ends up in
    ``soft_joint_vel_limits`` and therefore in main's ``dof_vel_limits`` penalty. We hard-code
    the same numbers as a literal tuple on our term instead of reading them off the robot, so
    the two only agree as long as somebody keeps them agreeing -- hence this reader.

    Groups may give one scalar for every joint they match or a regex→value mapping.
    """
    table = effective_robot_actuators()["table"]
    assets = ast.parse(_git_show(ASSETS_CFG))
    node = next(
        (
            n
            for n in assets.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == table
            and isinstance(n.value, ast.Dict)
        ),
        None,
    )
    if node is None:
        raise LookupError(f"main's {ASSETS_CFG} has no dict literal named {table!r}")

    limits: dict[str, float] = {}
    for group in node.value.values:
        if not isinstance(group, ast.Call):
            raise LookupError(f"{table} holds a non-call entry: {ast.unparse(group)}")
        kwargs = _kwargs(group)
        patterns = kwargs.get("joint_names_expr")
        spec = kwargs.get("velocity_limit_sim")
        if patterns is None or spec is None:
            raise LookupError(f"{table} entry {ast.unparse(group.func)} lacks joint_names_expr/velocity_limit_sim")
        matched = [j for p in patterns for j in joint_names if re.fullmatch(p, j)]
        if not matched:
            raise LookupError(f"{table} entry matches no joint of the {len(joint_names)} given: {patterns}")
        for joint in matched:
            if isinstance(spec, dict):
                hits = [v for p, v in spec.items() if re.fullmatch(p, joint)]
                if len(hits) != 1:
                    raise LookupError(f"{joint!r} matches {len(hits)} velocity_limit_sim patterns in {table}")
                limits[joint] = float(hits[0])
            else:
                limits[joint] = float(spec)
    missing = [j for j in joint_names if j not in limits]
    if missing:
        raise LookupError(f"{table} leaves {missing} without a velocity limit")
    return limits


def _articulation_actuators(cfg_symbol: str) -> str:
    """The ``actuators=`` table named by an ArticulationCfg in main's asset module."""
    assets = ast.parse(_git_show(ASSETS_CFG))
    for node in assets.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == cfg_symbol):
            continue
        if not isinstance(node.value, ast.Call):
            break
        for kw in node.value.keywords:
            if kw.arg == "actuators" and isinstance(kw.value, ast.Name):
                return kw.value.id
    raise LookupError(f"{cfg_symbol} does not name an actuators table in main's {ASSETS_CFG}")


def g1_robot_overrides() -> dict[str, Any]:
    """What G1ParkourEnvCfg adds on top of the shared parkour file."""
    effective = effective_robot_actuators()
    return {
        "spawn_z": G1_SPAWN_Z,
        "merge_fixed_joints": G1_MERGE_FIXED_JOINTS,
        "actuators": effective["table"],
        "shoe_urdf": G1_SHOE_URDF_SUFFIX,
        "volume_z_min": G1_SHOE_VOLUME_Z[0],
        "volume_z_max": G1_SHOE_VOLUME_Z[1],
        "feet_at_plane_height_offset": G1_SHOE_HEIGHT_OFFSET,
        "uses_delayed_actuators": effective["delayed"],
        "declares_delayed_actuators": effective["declared_in_base"],
    }


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
