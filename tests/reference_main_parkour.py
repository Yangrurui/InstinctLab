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


def observation_order(group: str) -> tuple[str, ...]:
    mapping = {
        "policy": "PolicyCfg",
        "critic": "CriticCfg",
        "amp_policy": "AmpPolicyStateObsCfg",
        "amp_reference": "AmpReferenceStateObsCfg",
    }
    return tuple(_obs_group(_parkour_module(), mapping[group]).keys())


def observation_scales(group: str = "policy") -> dict[str, float | None]:
    mapping = {
        "policy": "PolicyCfg",
        "critic": "CriticCfg",
        "amp_policy": "AmpPolicyStateObsCfg",
        "amp_reference": "AmpReferenceStateObsCfg",
    }
    scales: dict[str, float | None] = {}
    for name, call in _obs_group(_parkour_module(), mapping[group]).items():
        scales[name] = _kwargs(call).get("scale")
    return scales


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


def sim_params() -> dict[str, Any]:
    """Literals set in ParkourEnvCfg.__post_init__ on main."""
    for node in _parkour_module().body:
        if isinstance(node, ast.ClassDef) and node.name == "ParkourEnvCfg":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                    src = ast.unparse(item)
                    break
            else:
                raise LookupError("ParkourEnvCfg.__post_init__ missing")
            return {
                "decimation": 4 if "decimation = 4" in src else None,
                "episode_length_s": 20.0 if "episode_length_s = 20.0" in src else None,
                "physics_dt": 0.005 if "sim.dt = 0.005" in src else None,
                "gpu_max_rigid_patch_count": 10 * 2**15 if "gpu_max_rigid_patch_count = 10 * 2**15" in src else None,
                "gpu_collision_stack_size": 2**29 if "gpu_collision_stack_size = 2**29" in src else None,
            }
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
