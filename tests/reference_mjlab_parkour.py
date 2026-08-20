"""Reading InstinctMJ's parkour AMP config without importing it.

InstinctMJ is not a dependency. The factory in ``g1_parkour_target_amp_cfg.py`` is a module of
literals plus a ``shoe=True`` post-process; training never ran ``shoe=False``. Facts here are
what the file states, with the shoe overrides applied to the effective tables.
"""

from __future__ import annotations

import ast
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

REFERENCE = Path("/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py")
AGENT = Path("/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py")
ASSET = Path("/root/InstinctMJ/src/instinct_mj/assets/unitree_g1.py")
TRAIN = Path("/root/InstinctMJ/src/instinct_mj/scripts/instinct_rl/train.py")
SHIPPED_MOTION_YAML = Path("/root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run.yaml")
SHIPPED_MOTION_NPZ = Path(
    "/root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
)

FACTORY = "instinct_g1_parkour_amp_env_cfg"
SHOE_HEIGHT_OFFSET = 0.058
SHOE_VOLUME_Z = (-0.063, -0.023)
SHOE_XML_SUFFIX = "g1_29dof_torsoBase_popsicle_with_shoe.xml"
SPAWN_Z = 0.9
DELAY_MAX_LAG = 2
NCONMAX = 128
CCD_ITERATIONS = 128
SCANNER_MAX_DISTANCE = 10.0
SCANNER_ORIGIN_OFFSET = (0.0, 0.0, 0.0)


def available() -> bool:
    return REFERENCE.is_file()


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
    return ast.unparse(call.func).rsplit(".", 1)[-1]


@lru_cache(maxsize=1)
def _module() -> ast.Module:
    return ast.parse(REFERENCE.read_text())


def _factory() -> ast.FunctionDef:
    for node in _module().body:
        if isinstance(node, ast.FunctionDef) and node.name == FACTORY:
            return node
    raise LookupError(f"{REFERENCE.name} has no {FACTORY!r}")


def _assigned_dict(name: str) -> dict[str, ast.Call]:
    """A ``name = {k: Call(...)}`` inside the factory."""
    for statement in _factory().body:
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
            and isinstance(statement.value, ast.Dict)
        ):
            return {
                key.value: value
                for key, value in zip(statement.value.keys, statement.value.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(value, ast.Call)
            }
    raise LookupError(f"{FACTORY} has no dict named {name!r}")


def _cfg_attr_dict(attr: str) -> dict[str, ast.Call]:
    """A ``cfg.<attr> = {...}`` assignment. Rewards nest one more ``{"rewards": {...}}``."""
    for statement in _factory().body:
        if not isinstance(statement, ast.Assign):
            continue
        target = statement.targets[0]
        if ast.unparse(target) != f"cfg.{attr}" or not isinstance(statement.value, ast.Dict):
            continue
        node = statement.value
        if attr == "rewards":
            node = node.values[0]
            if not isinstance(node, ast.Dict):
                raise LookupError("cfg.rewards is not {group: {terms}}")
        return {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Call)
        }
    raise LookupError(f"{FACTORY} does not assign cfg.{attr}")


def _term_kwargs(nodes: dict[str, ast.Call]) -> dict[str, dict[str, Any]]:
    return {name: _kwargs(call) for name, call in nodes.items()}


def reward_names() -> tuple[str, ...]:
    return tuple(_cfg_attr_dict("rewards"))


def reward_weights() -> dict[str, float]:
    return {name: kwargs["weight"] for name, kwargs in _term_kwargs(_cfg_attr_dict("rewards")).items()}


def reward_functions() -> dict[str, str]:
    return {name: _func_name(call) for name, call in _cfg_attr_dict("rewards").items()}


def reward_params() -> dict[str, dict[str, Any]]:
    params = {
        name: dict(kwargs.get("params") or {}) for name, kwargs in _term_kwargs(_cfg_attr_dict("rewards")).items()
    }
    params["feet_at_plane"]["height_offset"] = SHOE_HEIGHT_OFFSET
    return params


def termination_names() -> tuple[str, ...]:
    return tuple(_cfg_attr_dict("terminations"))


def event_names() -> tuple[str, ...]:
    return tuple(_cfg_attr_dict("events"))


def event_params() -> dict[str, dict[str, Any]]:
    return {name: dict(kwargs.get("params") or {}) for name, kwargs in _term_kwargs(_cfg_attr_dict("events")).items()}


def observation_order(group: str) -> tuple[str, ...]:
    names = {
        "policy": "policy_terms",
        "critic": "critic_terms",
        "amp_policy": "amp_policy_terms",
        "amp_reference": "amp_reference_terms",
    }
    return tuple(_assigned_dict(names[group]))


def observation_noise(group: str = "policy") -> dict[str, tuple[float, float]]:
    names = {"policy": "policy_terms", "critic": "critic_terms"}
    bounds: dict[str, tuple[float, float]] = {}
    for name, call in _assigned_dict(names[group]).items():
        for kw in call.keywords:
            if kw.arg == "noise" and isinstance(kw.value, ast.Call):
                noise = _kwargs(kw.value)
                if "n_min" in noise and "n_max" in noise:
                    bounds[name] = (noise["n_min"], noise["n_max"])
    return bounds


def observation_scales(group: str = "policy") -> dict[str, float | None]:
    names = {
        "policy": "policy_terms",
        "critic": "critic_terms",
        "amp_policy": "amp_policy_terms",
        "amp_reference": "amp_reference_terms",
    }
    return {name: _kwargs(call).get("scale") for name, call in _assigned_dict(names[group]).items()}


def command_params() -> dict[str, Any]:
    commands = _cfg_attr_dict("commands")
    kwargs = _kwargs(commands["base_velocity"])
    ranges = kwargs.get("ranges")
    if isinstance(ranges, dict):
        kwargs["ranges"] = ranges
    return kwargs


def sim_overrides() -> dict[str, Any]:
    found: dict[str, Any] = {}
    for statement in _factory().body:
        if not isinstance(statement, ast.Assign):
            continue
        target = ast.unparse(statement.targets[0])
        mapping = {
            "cfg.episode_length_s": "episode_length_s",
            "cfg.sim.nconmax": "nconmax",
            "cfg.sim.njmax": "njmax",
            "cfg.sim.contact_sensor_maxmatch": "contact_sensor_maxmatch",
            "cfg.sim.mujoco.iterations": "iterations",
            "cfg.sim.mujoco.ls_iterations": "ls_iterations",
            "cfg.sim.mujoco.ccd_iterations": "ccd_iterations",
            "cfg.scene.env_spacing": "env_spacing",
            "cfg.scene.entities['robot'].init_state.pos": "init_pos",
        }
        key = mapping.get(target) or mapping.get(target.replace('"', "'"))
        if key is not None:
            found[key] = _literal(statement.value)
    return found


def sensor_cfgs() -> dict[str, dict[str, Any]]:
    """Contact / ray / volume sensors the factory constructs, by name."""
    wanted = {
        "ForceThresholdContactSensorCfg",
        "ContactSensorCfg",
        "VolumePointsCfg",
        "RayCastSensorCfg",
        "NoisyGroupedRayCasterCameraCfg",
    }
    result: dict[str, dict[str, Any]] = {}
    for call in ast.walk(_factory()):
        if not isinstance(call, ast.Call):
            continue
        cls = ast.unparse(call.func).rsplit(".", 1)[-1]
        if cls not in wanted:
            continue
        kwargs = _kwargs(call)
        name = kwargs.get("name")
        if isinstance(name, str):
            result[name] = {"cfg_class": cls, **kwargs}
    return result


def shoe_effective() -> dict[str, Any]:
    """The numbers training uses after ``if shoe:`` (default True)."""
    return {
        "height_offset": SHOE_HEIGHT_OFFSET,
        "volume_z": SHOE_VOLUME_Z,
        "xml_suffix": SHOE_XML_SUFFIX,
        "spawn_z": SPAWN_Z,
    }


def delayed_actuator_lags() -> tuple[int, int]:
    """``(min_lag, max_lag)`` on InstinctMJ's beyondmimic delayed parkour actuators."""
    tree = ast.parse(ASSET.read_text())
    mins, maxs = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id.startswith("BEYONDMIMIC_G1_29DOF_DELAYED_") for t in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        kwargs = _kwargs(node.value)
        if "delay_min_lag" in kwargs:
            mins.add(kwargs["delay_min_lag"])
        if "delay_max_lag" in kwargs:
            maxs.add(kwargs["delay_max_lag"])
    if mins != {0} or maxs != {DELAY_MAX_LAG}:
        raise AssertionError(f"InstinctMJ delayed actuators are {mins}/{maxs}, not {{0}}/{{{DELAY_MAX_LAG}}}")
    return (0, DELAY_MAX_LAG)


def motion_source() -> dict[str, Any]:
    """AMASS directory + yaml filter, not a single npz."""
    found: dict[str, Any] = {}
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_PARKOUR_DATASET_DIR" for t in node.targets
        ):
            found["dataset_dir"] = ast.unparse(node.value)
        if isinstance(node, ast.ClassDef) and node.name == "AmassMotionCfg":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "filtered_motion_selection_filepath":
                        found["filter"] = ast.unparse(item.value) if item.value is not None else None
                    if item.target.id == "motion_start_from_middle_range":
                        found["motion_start_from_middle_range"] = _literal(item.value)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("MotionReferenceManagerCfg"):
            found.update(_kwargs(node))
    return found


def motion_filter_files() -> tuple[str, ...]:
    """Resolve InstinctMJ's yaml filter on disk when the shipped dataset is present."""
    import yaml

    source = motion_source()
    filter_expr = source.get("filter") or ""
    if "parkour_motion_without_run.yaml" not in filter_expr:
        raise LookupError("parkour yaml filter path not found in reference config")
    if SHIPPED_MOTION_YAML.is_file():
        yaml_path = SHIPPED_MOTION_YAML
    else:
        yaml_path = Path(
            os.path.expanduser("~/Xyk/Datasets/data&model/parkour_motion_reference/parkour_motion_without_run.yaml")
        )
    if not yaml_path.is_file():
        raise FileNotFoundError(f"parkour motion yaml not found at {yaml_path}")
    with yaml_path.open() as handle:
        data = yaml.safe_load(handle)
    files = tuple(data.get("selected_files") or ())
    return files


TERRAIN = Path("/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/parkour_env_cfg.py")
TERRAIN_SYMBOL = "ROUGH_TERRAINS_CFG"


@lru_cache(maxsize=1)
def terrain_recipe() -> dict[str, Any]:
    """InstinctMJ's ``ROUGH_TERRAINS_CFG``: grid constants plus per-sub-terrain kwargs.

    Read rather than transcribed. The test this feeds is named "matches InstinctMJ" and used to
    assert hand-copied literals, so it would have stayed green through any change on their side --
    the failure mode is a docstring that claims a parity nobody is checking.

    The play variant is deliberately not followed: ``ROUGH_TERRAINS_CFG_PLAY`` mutates ``num_rows``
    and ``num_cols`` after the copy, and training uses the un-mutated one.
    """
    module = ast.parse(TERRAIN.read_text())
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(isinstance(t, ast.Name) and t.id == TERRAIN_SYMBOL for t in node.targets):
            continue
        recipe = _kwargs(node.value)
        subs: dict[str, dict[str, Any]] = {}
        for keyword in node.value.keywords:
            if keyword.arg != "sub_terrains" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Call):
                    subs[key.value] = _kwargs(value)
        recipe["sub_terrains"] = subs
        return recipe
    raise LookupError(f"{TERRAIN.name} has no {TERRAIN_SYMBOL!r} assignment")


def terrain_importer() -> dict[str, Any]:
    """The ``cfg.scene.terrain = InstinctTerrainImporterCfg(...)`` the factory installs."""
    for statement in _factory().body:
        if (
            isinstance(statement, ast.Assign)
            and ast.unparse(statement.targets[0]) == "cfg.scene.terrain"
            and isinstance(statement.value, ast.Call)
        ):
            return _kwargs(statement.value)
    raise LookupError(f"{FACTORY} does not assign cfg.scene.terrain")


def train_script_calls_configure_torch_backends() -> bool:
    """InstinctMJ's runner calls mjlab's torch backend helper before learning."""
    if not TRAIN.is_file():
        return False
    tree = ast.parse(TRAIN.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("configure_torch_backends"):
            return True
    return False


def agent_fields() -> dict[str, Any]:
    """Literal defaults on InstinctMJ's parkour runner / algo / encoder classes."""
    tree = ast.parse(AGENT.read_text())
    fields: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            target = None
            value = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target, value = item.target.id, item.value
            elif isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
                target, value = item.targets[0].id, item.value
            if target is None or value is None:
                continue
            fields[f"{node.name}.{target}"] = _literal(value)
    return fields
