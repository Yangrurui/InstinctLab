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


def delayed_actuator_lag_groups() -> frozenset[frozenset[str]]:
    """Which joints draw the *same* actuation lag, keyed by ``delay_update_period``.

    mjlab fuses actuators whose delay settings match into one ``DelayBuffer``, so the period
    constant is what decides the joint-space correlation of the lag, not the actuator split.
    InstinctMJ gives its two leg groups one shared constant and everything else its own, which
    is a choice and not an accident -- the file defines ``_..._PERIOD_LEGS`` and then four
    siblings at ``+1..+4``. Returned as sets of joint-name patterns so the comparison is about
    the partition, not about group names.
    """
    tree = ast.parse(ASSET.read_text())
    by_period: dict[Any, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(isinstance(t, ast.Name) and t.id.startswith("BEYONDMIMIC_G1_29DOF_DELAYED_") for t in node.targets):
            continue
        kwargs = _kwargs(node.value)
        period, names = kwargs.get("delay_update_period"), kwargs.get("target_names_expr")
        if period is None or names is None:
            raise AssertionError(f"{ast.unparse(node.targets[0])} has no delay period / target names")
        by_period.setdefault(str(period), set()).update(names)
    if not by_period:
        raise AssertionError("no BEYONDMIMIC delayed actuator cfgs found in InstinctMJ's asset module")
    return frozenset(frozenset(group) for group in by_period.values())


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


_INSTINCTMJ_CAMERA = Path("/root/InstinctMJ/src/instinct_mj/sensors/noisy_camera/noisy_grouped_raycaster_camera.py")
_INSTINCTMJ_NOISY_MIXIN = Path("/root/InstinctMJ/src/instinct_mj/sensors/noisy_camera/noisy_camera.py")
_INSTINCTMJ_ASYNC_BUFFER = Path("/root/InstinctMJ/src/instinct_mj/utils/buffers/async_circular_buffer.py")
_INSTINCTMJ_DEPTH_OBS = Path("/root/InstinctMJ/src/instinct_mj/envs/mdp/observations/exteroception.py")
_MJLAB_CIRCULAR_BUFFER = Path("/root/mjlab/src/mjlab/utils/buffers/circular_buffer.py")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"InstinctMJ/mjlab source missing; refusing to guess: {path}")
    return path


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return item
    return None


def _calls_name(fn: ast.FunctionDef, suffix: str) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(suffix):
            return True
    return False


def _assigns_zero_to(fn: ast.FunctionDef, attr: str) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if attr in ast.unparse(target):
                value = _literal(node.value)
                if value in (0, 0.0):
                    return True
    return False


def depth_history_reset() -> dict[str, Any]:
    """How InstinctMJ clears camera history on env reset. Reads their sources; missing files raise.

    History lives on the sensor ``AsyncCircularBuffer``, not on ``delayed_visualizable_image``.
    That obs term only redraws delay. The buffer ``reset`` (inherited from mjlab
    ``CircularBuffer`` unless InstinctMJ overrides it) zeros the selected envs.
    """
    camera_src = _require_file(_INSTINCTMJ_CAMERA)
    mixin_src = _require_file(_INSTINCTMJ_NOISY_MIXIN)
    async_src = _require_file(_INSTINCTMJ_ASYNC_BUFFER)
    obs_src = _require_file(_INSTINCTMJ_DEPTH_OBS)
    camera = ast.parse(camera_src.read_text())
    mixin = ast.parse(mixin_src.read_text())
    async_buf = ast.parse(async_src.read_text())
    obs = ast.parse(obs_src.read_text())

    camera_reset = _class_method(camera, "NoisyGroupedRayCasterCamera", "reset")
    if camera_reset is None:
        raise LookupError(f"{camera_src} has no NoisyGroupedRayCasterCamera.reset")
    mixin_reset = _class_method(mixin, "NoisyCameraMixin", "reset_history_buffers")
    if mixin_reset is None:
        raise LookupError(f"{mixin_src} has no NoisyCameraMixin.reset_history_buffers")
    obs_reset = _class_method(obs, "delayed_visualizable_image", "reset")
    if obs_reset is None:
        raise LookupError(f"{obs_src} has no delayed_visualizable_image.reset")

    async_defines_reset = _class_method(async_buf, "AsyncCircularBuffer", "reset") is not None
    if async_defines_reset:
        buffer_reset = _class_method(async_buf, "AsyncCircularBuffer", "reset")
        buffer_src = async_src
    else:
        buffer_src = _require_file(_MJLAB_CIRCULAR_BUFFER)
        buffer_reset = _class_method(ast.parse(buffer_src.read_text()), "CircularBuffer", "reset")
        if buffer_reset is None:
            raise LookupError(f"{buffer_src} has no CircularBuffer.reset (AsyncCircularBuffer inherits it)")

    obs_reset_src = ast.unparse(obs_reset)
    return {
        "camera_source": str(camera_src),
        "camera_reset_calls_reset_history_buffers": _calls_name(camera_reset, "reset_history_buffers"),
        "history_buffers_reset_calls_buffer_reset": _calls_name(mixin_reset, "reset"),
        "async_buffer_defines_reset": async_defines_reset,
        "buffer_reset_source": str(buffer_src),
        "buffer_reset_zeros_buffer": _assigns_zero_to(buffer_reset, "_buffer"),
        "buffer_reset_zeros_num_pushes": _assigns_zero_to(buffer_reset, "_num_pushes"),
        "obs_term_reset_clears_history": "_history" in obs_reset_src or "_buffer" in obs_reset_src,
        "obs_term_reset_resamples_delay": "_num_delayed_frames" in obs_reset_src,
        "history_owner": "sensor_AsyncCircularBuffer",
    }
