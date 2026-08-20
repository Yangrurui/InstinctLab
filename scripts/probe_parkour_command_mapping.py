"""Runtime probe: parkour pose-velocity box binding vs built terrain columns.

Does not change task / command / terrain defaults. Constructs the training factory
(``play=False`` on InstinctMJ) and reads the live command term.

    python scripts/probe_parkour_command_mapping.py run --side ours \\
        --num-envs 256 --seed 42 --device cuda:2 --out /tmp/cmd_ours_s42.json
    CUDA_VISIBLE_DEVICES=0 /root/InstinctMJ/.venv/bin/python \\
        scripts/probe_parkour_command_mapping.py run --side instinctmj \\
        --num-envs 256 --seed 42 --device cuda:0 --out /tmp/cmd_mj_s42.json
    python scripts/probe_parkour_command_mapping.py compare \\
        /tmp/cmd_ours_s42.json /tmp/cmd_mj_s42.json

Engine packages are imported only inside ``run`` (never at module import time).
Compare keys by sub-terrain **name**, never by bare column / type id.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

OURS_TASK_ID = "Instinct-Parkour-Target-G1"
INSTINCTMJ_TASK_ID = "Instinct-Parkour-Target-Amp-G1-v0"
DEFAULT_INSTINCTMJ_ROOT = Path("/root/InstinctMJ")
DEFAULT_INSTINCTMJ_PYTHON = DEFAULT_INSTINCTMJ_ROOT / ".venv/bin/python"
COMMAND_NAME = "base_velocity"
DEFAULT_RESAMPLE_ROUNDS = 64
OFFICIAL_NUM_ENVS = 256
RANGE_ATOL = 1e-5

# Fields read from the live PoseVelocity command. Fail if missing — do not guess aliases.
REQUIRED_COMMAND_FIELDS = (
    "lin_vel_x_range",
    "vel_command_b",
    "max_command_b",
    "is_standing_env",
    "pos_command_w",
    "random_velocity_indices",
)


# ---------------------------------------------------------------------------
# Pure helpers (no engine imports)
# ---------------------------------------------------------------------------


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
    return {
        "task_id": train["task_id"],
        "play_false_in_env_cfg_factory": True,
        "source": str(path),
    }


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


def instinctmj_python_hint(current: str | None = None) -> str:
    exe = current or sys.executable
    return (
        f"InstinctMJ side must use {DEFAULT_INSTINCTMJ_PYTHON} "
        "(the interpreter behind instinct-train) with CUDA_VISIBLE_DEVICES=<n> "
        f"and --device cuda:0. Current executable is {exe}."
    )


def curriculum_column_indices(proportions: list[float], num_cols: int) -> list[int]:
    """Isaac / InstinctMJ declared-column formula: first type with cdf > j/n + 0.001."""
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


def built_column_names(names: list[str], proportions: list[float], built_num_cols: int) -> list[str]:
    """Name each **built** column. Declared ``num_cols`` is never used here."""
    if built_num_cols == len(names):
        return list(names)
    return [names[index] for index in curriculum_column_indices(proportions, built_num_cols)]


def name_mapping_from_lists(
    *,
    names: list[str],
    proportions: list[float],
    built_num_cols: int,
    declared_num_cols: int,
) -> dict[str, Any]:
    if built_num_cols <= 0:
        raise ValueError("built_num_cols must be positive")
    if built_num_cols == len(names):
        allocation = "one_column_per_type"
        column_to_name = list(names)
    else:
        allocation = "isaac_cumulative_proportion"
        column_to_name = built_column_names(names, proportions, built_num_cols)
    return {
        "kind": "built_grid",
        "allocation": allocation,
        "rule": (
            "type_id indexes the built grid. Names are one column per sub-terrain when "
            "built_cols == n_types, else Isaac j/n+0.001 on the built width. "
            "Declared num_cols is never used to name columns."
        ),
        "built_num_cols": int(built_num_cols),
        "declared_num_cols": int(declared_num_cols),
        "sub_terrain_names": list(names),
        "proportions": [float(p) for p in proportions],
        "column_to_name": list(column_to_name),
    }


def type_id_to_name(type_id: int, mapping: dict[str, Any]) -> str:
    names = mapping["column_to_name"]
    if not (0 <= int(type_id) < len(names)):
        raise KeyError(f"type_id {type_id} is outside built columns {len(names)}; refusing to invent a name")
    return str(names[int(type_id)])


def ranges_close(left: tuple[float, float], right: tuple[float, float], atol: float = RANGE_ATOL) -> bool:
    return abs(left[0] - right[0]) <= atol and abs(left[1] - right[1]) <= atol


def unique_range(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not pairs:
        return None
    first = pairs[0]
    for item in pairs[1:]:
        if not ranges_close(first, item):
            return None
    return first


def declared_column_box_for_type_ids(
    *,
    names: list[str],
    proportions: list[float],
    declared_num_cols: int,
    velocity_ranges: dict[str, tuple[float, float]],
) -> dict[int, dict[str, Any]]:
    """Replay InstinctMJ's init loop: declared column id is matched against ``terrain_types``.

    Returns ``{declared_col_id: {source_name, lin_vel_x}}`` for each declared column that
    received a box. Built type ids that never appear as a declared column stay unbound.
    """
    sub_indices = curriculum_column_indices(proportions, declared_num_cols)
    out: dict[int, dict[str, Any]] = {}
    for name, box in velocity_ranges.items():
        if name not in names:
            raise KeyError(f"velocity box {name!r} is not in sub-terrain names {names}")
        type_index = names.index(name)
        for declared_col in (index for index, assigned in enumerate(sub_indices) if assigned == type_index):
            out[int(declared_col)] = {
                "source_name": name,
                "lin_vel_x": (float(box[0]), float(box[1])),
            }
    return out


def name_loop_box_for_type_ids(
    *,
    names: list[str],
    proportions: list[float],
    built_num_cols: int,
    velocity_ranges: dict[str, tuple[float, float]],
) -> dict[int, dict[str, Any]]:
    """Ours-style bind: apply each named box to built columns that carry that name."""
    column_to_name = built_column_names(names, proportions, built_num_cols)
    out: dict[int, dict[str, Any]] = {}
    for name, box in velocity_ranges.items():
        for column, column_name in enumerate(column_to_name):
            if column_name == name:
                out[column] = {
                    "source_name": name,
                    "lin_vel_x": (float(box[0]), float(box[1])),
                }
    return out


def classify_bind(
    *,
    name: str,
    effective: tuple[float, float] | None,
    declared: tuple[float, float] | None,
    predicted_source: str | None,
) -> str:
    if effective is None:
        return "mixed_or_empty"
    if declared is None:
        return "no_declared_box"
    if ranges_close(effective, (0.0, 0.0)) and not ranges_close(declared, (0.0, 0.0)):
        return "unbound_zero"
    value_ok = ranges_close(effective, declared)
    source_ok = predicted_source is None or predicted_source == name
    if value_ok and source_ok:
        return "aligned"
    if value_ok and not source_ok:
        return "value_ok_source_mismatch"
    return "wrong_box"


def summarize_samples(values: list[float], standing: list[bool], target_near: list[bool]) -> dict[str, Any]:
    if not values:
        return {
            "n_samples": 0,
            "min": None,
            "max": None,
            "mean": None,
            "zero_fraction": None,
            "standing_fraction": None,
            "target_near_fraction": None,
            "zero_given_not_standing_not_near": None,
        }
    n = len(values)
    zeros = sum(1 for v in values if abs(v) <= RANGE_ATOL)
    stand_n = sum(1 for flag in standing if flag)
    near_n = sum(1 for flag in target_near if flag)
    residual = [values[i] for i in range(n) if (not standing[i]) and (not target_near[i])]
    residual_zero = sum(1 for v in residual if abs(v) <= RANGE_ATOL) / len(residual) if residual else None
    return {
        "n_samples": n,
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / n),
        "zero_fraction": float(zeros / n),
        "standing_fraction": float(stand_n / n),
        "target_near_fraction": float(near_n / n),
        "zero_given_not_standing_not_near": residual_zero,
        "note": (
            "Issued vel_command_b[:,0] after real _resample + _update_command. "
            "Zeros come from (1) standing envs (~rel_standing_envs), "
            "(2) target_dist <= target_dis_threshold, "
            "(3) lin_vel_x_range / max_command clamp at 0."
        ),
    }


def aggregate_by_terrain_name(
    *,
    type_ids: list[int],
    mapping: dict[str, Any],
    live_ranges: list[tuple[float, float]],
    declared_boxes: dict[str, tuple[float, float]],
    predicted_declared: dict[int, dict[str, Any]],
    predicted_name_loop: dict[int, dict[str, Any]],
    issued: list[float],
    standing: list[bool],
    target_near: list[bool],
    resample_rounds: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    n_env = len(type_ids)
    for env_i, type_id in enumerate(type_ids):
        name = type_id_to_name(type_id, mapping)
        row = grouped.setdefault(
            name,
            {
                "name": name,
                "env_indices": [],
                "type_ids": set(),
                "live_ranges": [],
                "issued": [],
                "standing": [],
                "target_near": [],
            },
        )
        row["env_indices"].append(env_i)
        row["type_ids"].add(int(type_id))
        row["live_ranges"].append(live_ranges[env_i])
        # ``_resample_issued`` is round-major: [r0e0, r0e1, ..., r1e0, ...].
        for round_i in range(resample_rounds):
            index = round_i * n_env + env_i
            row["issued"].append(issued[index])
            row["standing"].append(standing[index])
            row["target_near"].append(target_near[index])

    out: dict[str, dict[str, Any]] = {}
    for name, row in grouped.items():
        ids = sorted(row["type_ids"])
        effective = unique_range(row["live_ranges"])
        declared = declared_boxes.get(name)
        pred_sources = {predicted_declared[i]["source_name"] for i in ids if i in predicted_declared}
        pred_source = next(iter(pred_sources)) if len(pred_sources) == 1 else None
        name_sources = {predicted_name_loop[i]["source_name"] for i in ids if i in predicted_name_loop}
        name_source = next(iter(name_sources)) if len(name_sources) == 1 else None
        out[name] = {
            "name": name,
            "env_count": len(row["env_indices"]),
            "env_fraction": len(row["env_indices"]) / n_env if n_env else 0.0,
            "type_ids": ids,
            "effective_range": list(effective) if effective is not None else None,
            "live_range_unique": sorted({(round(a, 6), round(b, 6)) for a, b in row["live_ranges"]}),
            "declared_box": list(declared) if declared is not None else None,
            "predicted_declared_col_source": pred_source,
            "predicted_declared_col_sources": sorted(pred_sources),
            "predicted_name_loop_source": name_source,
            "bind_class": classify_bind(
                name=name,
                effective=effective,
                declared=declared,
                predicted_source=pred_source,
            ),
            "name_loop_class": classify_bind(
                name=name,
                effective=effective,
                declared=declared,
                predicted_source=name_source,
            ),
            "sample": summarize_samples(row["issued"], row["standing"], row["target_near"]),
        }
    return out


def payload_schema_keys() -> frozenset[str]:
    return frozenset({"metadata", "name_mapping_source", "by_terrain_name", "misbinds"})


def terrain_row_schema_keys() -> frozenset[str]:
    return frozenset(
        {
            "name",
            "env_count",
            "env_fraction",
            "type_ids",
            "effective_range",
            "live_range_unique",
            "declared_box",
            "predicted_declared_col_source",
            "predicted_declared_col_sources",
            "predicted_name_loop_source",
            "bind_class",
            "name_loop_class",
            "sample",
        }
    )


def validate_payload_schema(payload: dict[str, Any], *, official: bool = False) -> None:
    missing = payload_schema_keys() - set(payload)
    if missing:
        raise ValueError(f"payload missing keys {sorted(missing)}")
    meta = payload["metadata"]
    for key in (
        "side",
        "seed",
        "num_envs",
        "task_id",
        "play",
        "declared_num_cols",
        "built_num_cols",
        "command_name",
    ):
        if key not in meta:
            raise ValueError(f"metadata missing {key!r}")
    mapping = payload["name_mapping_source"]
    if mapping.get("kind") != "built_grid":
        raise ValueError("name_mapping_source.kind must be 'built_grid' (no bare column pairing)")
    if "column_to_name" not in mapping:
        raise ValueError("name_mapping_source missing column_to_name")
    if official:
        if int(meta["num_envs"]) != OFFICIAL_NUM_ENVS:
            raise ValueError(f"official schema requires num_envs={OFFICIAL_NUM_ENVS}, got {meta['num_envs']}")
        if meta["side"] == "instinctmj" and meta.get("play") is not False:
            raise ValueError("official InstinctMJ payload must record play=False")
    by_name = payload["by_terrain_name"]
    if any(key.isdigit() for key in by_name):
        raise ValueError("by_terrain_name is keyed by a digit; refuse bare type-id tables")
    for name, row in by_name.items():
        missing_row = terrain_row_schema_keys() - set(row)
        if missing_row:
            raise ValueError(f"{name}: missing {sorted(missing_row)}")
        if row.get("name") != name:
            raise ValueError(f"row name {row.get('name')!r} != key {name!r}")


def collect_misbinds(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    misbinds: list[dict[str, Any]] = []
    for name, row in by_name.items():
        klass = row.get("bind_class")
        if klass in {"aligned", "no_declared_box", "mixed_or_empty"}:
            continue
        misbinds.append(
            {
                "name": name,
                "type_ids": list(row.get("type_ids") or []),
                "bind_class": klass,
                "declared_box": row.get("declared_box"),
                "effective_range": row.get("effective_range"),
                "predicted_declared_col_source": row.get("predicted_declared_col_source"),
                "predicted_declared_col_sources": row.get("predicted_declared_col_sources"),
            }
        )
    return misbinds


def compare_payloads(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Diff two probe dumps by sub-terrain name. Raises if either table is type-id keyed."""
    validate_payload_schema(left)
    validate_payload_schema(right)
    left_names = set(left["by_terrain_name"])
    right_names = set(right["by_terrain_name"])
    names = sorted(left_names | right_names)
    per_name: dict[str, Any] = {}
    range_diffs: list[str] = []
    source_diffs: list[str] = []
    for name in names:
        lrow = left["by_terrain_name"].get(name)
        rrow = right["by_terrain_name"].get(name)
        if lrow is None or rrow is None:
            per_name[name] = {"present": [lrow is not None, rrow is not None]}
            range_diffs.append(name)
            continue
        lrange = tuple(lrow["effective_range"]) if lrow["effective_range"] is not None else None
        rrange = tuple(rrow["effective_range"]) if rrow["effective_range"] is not None else None
        same_range = lrange is not None and rrange is not None and ranges_close(lrange, rrange)
        same_source = lrow.get("predicted_declared_col_source") == rrow.get("predicted_declared_col_source")
        per_name[name] = {
            "left_effective_range": lrow["effective_range"],
            "right_effective_range": rrow["effective_range"],
            "left_declared_box": lrow["declared_box"],
            "right_declared_box": rrow["declared_box"],
            "left_bind_class": lrow["bind_class"],
            "right_bind_class": rrow["bind_class"],
            "left_source": lrow.get("predicted_declared_col_source"),
            "right_source": rrow.get("predicted_declared_col_source"),
            "left_type_ids": lrow["type_ids"],
            "right_type_ids": rrow["type_ids"],
            "same_effective_range": same_range,
            "same_declared_col_source": same_source,
        }
        if not same_range:
            range_diffs.append(name)
        if lrow.get("bind_class") != rrow.get("bind_class") or not same_source:
            source_diffs.append(name)
    return {
        "left_side": left["metadata"]["side"],
        "right_side": right["metadata"]["side"],
        "names_only_left": sorted(left_names - right_names),
        "names_only_right": sorted(right_names - left_names),
        "effective_range_diffs": range_diffs,
        "source_or_class_diffs": source_diffs,
        "per_name": per_name,
        "left_misbinds": left.get("misbinds") or [],
        "right_misbinds": right.get("misbinds") or [],
    }


def unique_box_recipe(names: list[str]) -> dict[str, tuple[float, float]]:
    """Synthetic boxes with a distinct lo per name so source identity cannot hide."""
    return {name: (0.1 * (index + 1), 0.1 * (index + 1) + 0.05) for index, name in enumerate(names)}


# ---------------------------------------------------------------------------
# Runtime (engine imports stay inside these functions)
# ---------------------------------------------------------------------------


def _ensure_instinctmj_root(root: Path) -> Path:
    src = root / "src"
    if not src.is_dir():
        raise RuntimeError(
            f"InstinctMJ reference tree not found at {root}. Set INSTINCTMJ_ROOT; refusing to fall back to ours."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _as_float_pair(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _require_field(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        public = sorted(key for key in dir(obj) if not key.startswith("__"))
        raise RuntimeError(f"command term has no field {name!r}; public names={public}")
    return getattr(obj, name)


def _tensor_to_pairs(tensor) -> list[tuple[float, float]]:
    rows = tensor.detach().cpu().tolist()
    return [(float(row[0]), float(row[1])) for row in rows]


def _tensor_to_list(tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _actual_column_count(terrain) -> int:
    patches = getattr(terrain, "flat_patches", None)
    if patches and "target" in patches:
        return int(patches["target"].shape[1])
    origins = getattr(terrain, "terrain_origins", None)
    if origins is not None:
        return int(origins.shape[1])
    raise RuntimeError("cannot read built column count (no flat_patches['target'] or terrain_origins)")


def _live_name_mapping(terrain) -> dict[str, Any]:
    cfg = getattr(terrain, "cfg", None)
    generator = getattr(cfg, "terrain_generator", None) if cfg is not None else None
    if generator is None:
        raise RuntimeError("terrain.cfg.terrain_generator missing; refusing bare column ids")
    names = list(getattr(generator, "sub_terrains", {}) or {})
    if not names:
        raise RuntimeError("terrain_generator.sub_terrains is empty")
    if not bool(getattr(generator, "curriculum", False)):
        raise RuntimeError("non-curriculum grid mixes types inside a column; names are unresolvable")
    sub = generator.sub_terrains
    proportions = [float(sub[name].proportion) for name in names]
    return name_mapping_from_lists(
        names=names,
        proportions=proportions,
        built_num_cols=_actual_column_count(terrain),
        declared_num_cols=int(generator.num_cols),
    )


def _declared_boxes_from_cfg(cfg) -> dict[str, tuple[float, float]]:
    raw = getattr(cfg, "velocity_ranges", None) or {}
    out: dict[str, tuple[float, float]] = {}
    for name, box in raw.items():
        lin = box["lin_vel_x"] if isinstance(box, dict) else box
        out[str(name)] = _as_float_pair(lin)
    return out


def _build_ours(*, num_envs: int, device: str, seed: int):
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    spec = parkour_target_g1()
    MjlabAdapter.bootstrap(argparse.Namespace(device=device))
    compiled = MjlabAdapter().compile(spec, num_envs=num_envs, device=device)
    compiled.env_cfg.seed = seed
    env = compiled.make_env()
    return env, compiled


def _build_instinctmj(*, num_envs: int, device: str, seed: int, root: Path):
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
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    if device not in {"cuda:0", "cpu"}:
        print(
            "[WARN] InstinctMJ training uses CUDA_VISIBLE_DEVICES=<n> and --device cuda:0. "
            f"You passed {device}. Prefer: CUDA_VISIBLE_DEVICES=0 {DEFAULT_INSTINCTMJ_PYTHON} "
            "... --device cuda:0",
            flush=True,
        )
    env = InstinctRlEnv(cfg=env_cfg, device=device)
    return env, env_cfg


def _command_term(env, name: str):
    manager = env.command_manager
    if not hasattr(manager, "get_term"):
        raise RuntimeError("command_manager has no get_term")
    return manager.get_term(name)


def _resample_issued(term, *, rounds: int, target_dis_threshold: float) -> tuple[list[float], list[bool], list[bool]]:
    import torch

    env_ids = torch.arange(term.num_envs, device=term.device)
    issued: list[float] = []
    standing: list[bool] = []
    target_near: list[bool] = []
    for _ in range(rounds):
        term._resample(env_ids)
        term._update_command()
        vel = _require_field(term, "vel_command_b")[:, 0].detach().cpu().tolist()
        stand = _require_field(term, "is_standing_env").detach().cpu().tolist()
        pos = _require_field(term, "pos_command_w")
        robot = term.robot
        root = robot.data.root_link_pos_w[:, :3]
        dist = torch.norm(pos[:, :2] - root[:, :2], dim=1).detach().cpu().tolist()
        issued.extend(float(v) for v in vel)
        standing.extend(bool(flag) for flag in stand)
        target_near.extend(float(d) <= target_dis_threshold for d in dist)
    return issued, standing, target_near


def _repo_commit(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = root / ".git" / text.split(" ", 1)[1]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        return text
    return text


def run_probe(args: argparse.Namespace) -> int:
    if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
        sys.path.pop(0)

    import torch

    side = args.side
    instinctmj_root = Path(os.environ.get("INSTINCTMJ_ROOT", DEFAULT_INSTINCTMJ_ROOT))
    task_id = OURS_TASK_ID if side == "ours" else INSTINCTMJ_TASK_ID
    play = False

    if side == "ours":
        env, compiled = _build_ours(num_envs=args.num_envs, device=args.device, seed=args.seed)
        source_path = str(Path(__file__).resolve().parents[1])
        _ = compiled
    elif side == "instinctmj":
        read_instinctmj_train_registration(instinctmj_train_task_source(instinctmj_root))
        env, _env_cfg = _build_instinctmj(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            root=instinctmj_root,
        )
        source_path = str(instinctmj_root)
    else:
        raise SystemExit(f"unknown side {side!r}")

    env.reset()
    term = _command_term(env, args.command_name)
    for field in REQUIRED_COMMAND_FIELDS:
        _require_field(term, field)

    terrain = env.scene["terrain"]
    mapping = _live_name_mapping(terrain)
    type_ids = [int(x) for x in _tensor_to_list(terrain.terrain_types)]
    live_ranges = _tensor_to_pairs(_require_field(term, "lin_vel_x_range"))
    declared_boxes = _declared_boxes_from_cfg(term.cfg)
    names = list(mapping["sub_terrain_names"])
    proportions = list(mapping["proportions"])
    predicted_declared = declared_column_box_for_type_ids(
        names=names,
        proportions=proportions,
        declared_num_cols=int(mapping["declared_num_cols"]),
        velocity_ranges=declared_boxes,
    )
    predicted_name_loop = name_loop_box_for_type_ids(
        names=names,
        proportions=proportions,
        built_num_cols=int(mapping["built_num_cols"]),
        velocity_ranges=declared_boxes,
    )

    live_matches_declared_loop = all(
        type_id in predicted_declared and ranges_close(live_ranges[env_i], predicted_declared[type_id]["lin_vel_x"])
        for env_i, type_id in enumerate(type_ids)
        if type_id in predicted_declared
    ) and all(
        (type_id in predicted_declared) or ranges_close(live_ranges[env_i], (0.0, 0.0))
        for env_i, type_id in enumerate(type_ids)
    )
    live_matches_name_loop = all(
        type_id in predicted_name_loop and ranges_close(live_ranges[env_i], predicted_name_loop[type_id]["lin_vel_x"])
        for env_i, type_id in enumerate(type_ids)
        if type_id in predicted_name_loop
    ) and all(
        (type_id in predicted_name_loop) or ranges_close(live_ranges[env_i], (0.0, 0.0))
        for env_i, type_id in enumerate(type_ids)
    )

    rel_standing = float(getattr(term.cfg, "rel_standing_envs", 0.0))
    target_thr = float(getattr(term.cfg, "target_dis_threshold", 0.0))
    issued, standing, target_near = _resample_issued(term, rounds=args.resample_rounds, target_dis_threshold=target_thr)

    targets = _require_field(term, "pos_command_w").detach().cpu().tolist()
    by_name = aggregate_by_terrain_name(
        type_ids=type_ids,
        mapping=mapping,
        live_ranges=live_ranges,
        declared_boxes=declared_boxes,
        predicted_declared=predicted_declared,
        predicted_name_loop=predicted_name_loop,
        issued=issued,
        standing=standing,
        target_near=target_near,
        resample_rounds=args.resample_rounds,
    )
    for name, row in by_name.items():
        env_ids = [i for i, tid in enumerate(type_ids) if type_id_to_name(tid, mapping) == name]
        xs = [targets[i][0] for i in env_ids]
        ys = [targets[i][1] for i in env_ids]
        zs = [targets[i][2] for i in env_ids]
        row["reset_target"] = {
            "count": len(env_ids),
            "x_mean": float(sum(xs) / len(xs)) if xs else None,
            "y_mean": float(sum(ys) / len(ys)) if ys else None,
            "z_mean": float(sum(zs) / len(zs)) if zs else None,
        }

    payload = {
        "metadata": {
            "side": side,
            "seed": int(args.seed),
            "num_envs": int(env.num_envs),
            "task_id": task_id,
            "play": play,
            "command_name": args.command_name,
            "declared_num_cols": int(mapping["declared_num_cols"]),
            "built_num_cols": int(mapping["built_num_cols"]),
            "resample_rounds": int(args.resample_rounds),
            "rel_standing_envs": rel_standing,
            "target_dis_threshold": target_thr,
            "command_fields_used": list(REQUIRED_COMMAND_FIELDS),
            "live_matches_declared_col_loop": live_matches_declared_loop,
            "live_matches_name_loop": live_matches_name_loop,
            "column_names_attr": list(getattr(term, "_column_names", [])) or None,
            "commit": _repo_commit(Path(source_path)),
            "source_path": source_path,
            "device": args.device,
        },
        "name_mapping_source": mapping,
        "by_terrain_name": by_name,
        "misbinds": collect_misbinds(by_name),
    }
    validate_payload_schema(payload, official=int(env.num_envs) == OFFICIAL_NUM_ENVS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"wrote {args.out} side={side} built_cols={mapping['built_num_cols']} "
        f"declared={mapping['declared_num_cols']} misbinds={len(payload['misbinds'])}",
        flush=True,
    )
    for row in payload["misbinds"]:
        print(
            f"  misbind {row['name']}: class={row['bind_class']} "
            f"effective={row['effective_range']} declared={row['declared_box']} "
            f"source={row['predicted_declared_col_source']}",
            flush=True,
        )
    env.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="construct factory and dump command binding")
    run.add_argument("--side", required=True, choices=("ours", "instinctmj"))
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--num-envs", type=int, default=OFFICIAL_NUM_ENVS)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--command-name", default=COMMAND_NAME)
    run.add_argument("--resample-rounds", type=int, default=DEFAULT_RESAMPLE_ROUNDS)

    compare = sub.add_parser("compare", help="diff two dumps by terrain name")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)

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
        report = compare_payloads(left, right)
        print(json.dumps(report, indent=1))
        return 0
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
