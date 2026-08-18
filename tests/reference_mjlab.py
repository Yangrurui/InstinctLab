"""Reading InstinctMJ's locomotion config without importing it.

InstinctMJ is the mjlab-side reference implementation and, per decision D3, not a dependency: it is
not installed, and a test that imported it would either fail everywhere or pin the project to it.
Its config is a module of literals, though, so the facts that parity is about -- which terms exist,
in what order, with what weights -- can be read straight off the syntax tree.

Only what a comparison needs is extracted. Anything requiring evaluation is left alone rather than
half-interpreted, so a fact this module reports is a fact the file states.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

REFERENCE = Path("/root/InstinctMJ/src/instinct_mj/tasks/locomotion/config/g1/flat_env_cfg.py")


def available() -> bool:
    return REFERENCE.is_file()


def _literal(node: ast.AST) -> Any:
    """The node's value if it is one, else a marker naming what it was."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        # A container is worth descending into: one un-evaluable entry, typically an entity config,
        # should not turn its neighbours into markers too.
        if isinstance(node, ast.Dict):
            return {
                _literal(key): _literal(value)
                for key, value in zip(node.keys, node.values, strict=True)
                if key is not None
            }
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(_literal(element) for element in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _literal(node.operand)
            return -inner if isinstance(inner, (int, float)) else f"<-{inner}>"
        if isinstance(node, ast.Attribute):
            return f"<{ast.unparse(node)}>"
        if isinstance(node, ast.Call):
            return f"<call {ast.unparse(node.func)}>"
        return f"<{type(node).__name__}>"


def _kwargs(call: ast.Call) -> dict[str, Any]:
    return {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg is not None}


def _func_name(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "func":
            return ast.unparse(kw.value).rsplit(".", 1)[-1]
    return ""


def _module() -> ast.Module:
    return ast.parse(REFERENCE.read_text())


def _returned_dict(function: str) -> dict[str, ast.AST]:
    """The dict literal a single-return helper returns, keyed by its string keys."""
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for statement in ast.walk(node):
                if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict):
                    return {
                        key.value: value
                        for key, value in zip(statement.value.keys, statement.value.values, strict=True)
                        if isinstance(key, ast.Constant)
                    }
    raise LookupError(f"{REFERENCE.name} has no single-return helper named {function!r}.")


def _assigned_dict(function: str, name: str) -> dict[str, ast.AST]:
    """A dict literal assigned to a local variable inside a helper."""
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == name
                    and isinstance(statement.value, ast.Dict)
                ):
                    return {
                        key.value: value
                        for key, value in zip(statement.value.keys, statement.value.values, strict=True)
                        if isinstance(key, ast.Constant)
                    }
    raise LookupError(f"{REFERENCE.name} has no {name!r} inside {function!r}.")


def observation_terms(group: str = "policy") -> list[tuple[str, str]]:
    """``(term name, function name)`` for one group, in declaration order."""
    return [
        (name, _func_name(node))
        for name, node in _assigned_dict("_observations_cfg", f"{group}_terms").items()
        if isinstance(node, ast.Call)
    ]


def observation_noise(group: str = "policy") -> dict[str, tuple[float, float]]:
    """Uniform noise bounds per term, for terms that declare them."""
    bounds: dict[str, tuple[float, float]] = {}
    for name, node in _assigned_dict("_observations_cfg", f"{group}_terms").items():
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "noise" and isinstance(kw.value, ast.Call):
                noise = _kwargs(kw.value)
                if "n_min" in noise and "n_max" in noise:
                    bounds[name] = (noise["n_min"], noise["n_max"])
    return bounds


def rewards() -> dict[str, float]:
    """Reward weights by term name, in declaration order."""
    return {
        name: _kwargs(node).get("weight")
        for name, node in _returned_dict("_rewards_cfg").items()
        if isinstance(node, ast.Call)
    }


def reward_functions() -> dict[str, str]:
    return {
        name: _func_name(node) for name, node in _returned_dict("_rewards_cfg").items() if isinstance(node, ast.Call)
    }


def terminations() -> dict[str, str]:
    return {
        name: _func_name(node)
        for name, node in _returned_dict("_terminations_cfg").items()
        if isinstance(node, ast.Call)
    }


def events() -> dict[str, dict[str, Any]]:
    """Mode, interval and parameters per event term."""
    result: dict[str, dict[str, Any]] = {}
    for name, node in _returned_dict("_events_cfg").items():
        if not isinstance(node, ast.Call):
            continue
        kwargs = _kwargs(node)
        result[name] = {
            "func": _func_name(node),
            "mode": kwargs.get("mode"),
            "interval_range_s": kwargs.get("interval_range_s"),
            "params": kwargs.get("params", {}),
        }
    return result


def commands() -> dict[str, Any]:
    for name, node in _returned_dict("_commands_cfg").items():
        if isinstance(node, ast.Call):
            ranges = next(
                (_kwargs(kw.value) for kw in node.keywords if kw.arg == "ranges" and isinstance(kw.value, ast.Call)),
                {},
            )
            return {"name": name, **_kwargs(node), "ranges": ranges}
    raise LookupError("The reference declares no command.")


def timing() -> dict[str, Any]:
    """Class-level timing constants of the env config."""
    wanted = {"decimation", "episode_length_s"}
    found: dict[str, Any] = {}
    for node in ast.walk(_module()):
        if isinstance(node, ast.ClassDef) and node.name == "G1LocomotionFlatEnvCfg":
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id in wanted
                    and statement.value is not None
                ):
                    found[statement.target.id] = _literal(statement.value)
    for node in ast.walk(_module()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MujocoCfg":
            found.update({key: value for key, value in _kwargs(node).items()})
    return found


def scene_sensors() -> dict[str, dict[str, Any]]:
    """Contact sensors the reference declares, by name."""
    result: dict[str, dict[str, Any]] = {}
    for node in ast.walk(_module()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.endswith("ContactSensorCfg"):
            kwargs = _kwargs(node)
            primary = next(
                (_kwargs(kw.value) for kw in node.keywords if kw.arg == "primary" and isinstance(kw.value, ast.Call)),
                {},
            )
            result[kwargs.get("name", "?")] = {**kwargs, "primary": primary, "cfg_class": node.func.id}
    return result
