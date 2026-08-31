"""Bounded capture of mutable tensor state owned by native components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .snapshot import SnapshotError

_SKIP_ATTRIBUTES = frozenset(
    {
        "cfg",
        "_cfg",
        "env",
        "_env",
        "scene",
        "_scene",
        "sim",
        "_sim",
        "model",
        "_model",
        "entity",
        "_entity",
        "asset",
        "_asset",
        "robot",
        "_robot",
        "terrain",
        "_terrain",
        "sensor",
        "_sensor",
        "device",
        "_device",
        "clip",
        "clips",
        "inventory",
        "ref",
        "resolved",
        "_packed_clips",
        "_motion_origins",
    }
)
_SCALAR_STATE_WORDS = (
    "counter",
    "count",
    "index",
    "lag",
    "phase",
    "pointer",
    "step",
    "tick",
    "time",
    "timestamp",
)


def capture_state_tree(roots: Mapping[str, object]) -> dict[str, Any]:
    """Clone mutable tensors beneath explicitly selected native roots."""
    seen: set[int] = set()
    return {
        name: _capture(value, seen=seen, depth=0, attribute_name=name)
        for name, value in roots.items()
    }


def validate_state_tree(roots: Mapping[str, object], state: Mapping[str, Any]) -> None:
    """Validate the complete tree before any destination tensor is changed."""
    if set(state) != set(roots):
        raise SnapshotError(
            "Native state roots do not match: "
            f"got {sorted(state)}, expected {sorted(roots)}."
        )
    seen: set[int] = set()
    for name, root in roots.items():
        _validate(root, state[name], seen=seen, path=name)


def restore_state_tree(roots: Mapping[str, object], state: Mapping[str, Any]) -> None:
    """Restore a tree previously accepted by :func:`validate_state_tree`."""
    validate_state_tree(roots, state)
    seen: set[int] = set()
    for name, root in roots.items():
        _restore(root, state[name], seen=seen, path=name)


def _capture(
    value: Any,
    *,
    seen: set[int],
    depth: int,
    attribute_name: str,
) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "value": value.detach().clone()}
    if _is_torch_array(value):
        return {"kind": "tensor_proxy", "value": value.clone()}
    if isinstance(value, (bool, int, float)):
        return {"kind": "scalar", "value": value}
    if depth > 12:
        raise SnapshotError(
            f"Native component state exceeds the supported nesting depth at {attribute_name!r}."
        )
    identity = id(value)
    if identity in seen:
        return {"kind": "shared"}
    seen.add(identity)

    snapshot = getattr(value, "snapshot_state", None)
    restore = getattr(value, "restore_state", None)
    if callable(snapshot) and callable(restore):
        return {"kind": "component", "value": dict(snapshot())}
    if isinstance(value, Mapping):
        items = {
            str(key): _capture(
                item,
                seen=seen,
                depth=depth + 1,
                attribute_name=str(key),
            )
            for key, item in value.items()
            if isinstance(key, str)
        }
        return {"kind": "mapping", "items": items}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [
                _capture(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    attribute_name=str(index),
                )
                for index, item in enumerate(value)
            ],
        }
    attributes: dict[str, Any] = {}
    for name, item in vars(value).items() if hasattr(value, "__dict__") else ():
        if name in _SKIP_ATTRIBUTES:
            continue
        if _is_state_value(item, name):
            attributes[name] = _capture(
                item,
                seen=seen,
                depth=depth + 1,
                attribute_name=name,
            )
    return {"kind": "object", "attributes": attributes}


def _validate(
    target: Any,
    state: Any,
    *,
    seen: set[int],
    path: str,
) -> None:
    import torch

    if not isinstance(state, Mapping) or "kind" not in state:
        raise SnapshotError(f"Malformed native state node at {path}.")
    kind = state["kind"]
    if kind == "shared":
        return
    if kind == "scalar":
        if not isinstance(target, type(state.get("value"))):
            raise SnapshotError(f"Scalar state type changed at {path}.")
        return
    if kind in {"tensor", "tensor_proxy"}:
        source = state.get("value")
        destination = target if kind == "tensor" else getattr(target, "_tensor", None)
        if not isinstance(source, torch.Tensor) or not isinstance(
            destination, torch.Tensor
        ):
            raise SnapshotError(f"Expected tensor state at {path}.")
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise SnapshotError(
                f"Tensor state at {path} has shape/dtype "
                f"{tuple(source.shape)}/{source.dtype}, expected "
                f"{tuple(destination.shape)}/{destination.dtype}."
            )
        return
    identity = id(target)
    if identity in seen:
        return
    seen.add(identity)
    if kind == "component":
        if not callable(getattr(target, "restore_state", None)):
            raise SnapshotError(f"Stateful component at {path} has no restore_state().")
        if not isinstance(state.get("value"), Mapping):
            raise SnapshotError(f"Malformed component state at {path}.")
        return
    if kind == "mapping":
        items = state.get("items")
        if not isinstance(target, Mapping) or not isinstance(items, Mapping):
            raise SnapshotError(f"Expected mapping state at {path}.")
        if set(items) != {key for key in target if isinstance(key, str)}:
            raise SnapshotError(f"Mapping state keys changed at {path}.")
        for name, item in items.items():
            _validate(target[name], item, seen=seen, path=f"{path}.{name}")
        return
    if kind == "sequence":
        items = state.get("items")
        if not isinstance(target, (list, tuple)) or not isinstance(items, list):
            raise SnapshotError(f"Expected sequence state at {path}.")
        if len(target) != len(items):
            raise SnapshotError(f"Sequence state length changed at {path}.")
        for index, item in enumerate(items):
            _validate(target[index], item, seen=seen, path=f"{path}[{index}]")
        return
    if kind != "object":
        raise SnapshotError(f"Unknown native state node {kind!r} at {path}.")
    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        raise SnapshotError(f"Malformed object state at {path}.")
    for name, item in attributes.items():
        if not hasattr(target, name):
            raise SnapshotError(f"Native state attribute {path}.{name} is missing.")
        _validate(getattr(target, name), item, seen=seen, path=f"{path}.{name}")


def _restore(
    target: Any,
    state: Mapping[str, Any],
    *,
    seen: set[int],
    path: str,
) -> None:
    kind = state["kind"]
    if kind == "shared":
        return
    if kind in {"tensor", "tensor_proxy"}:
        destination = target if kind == "tensor" else target._tensor
        destination.copy_(state["value"].to(destination.device))
        return
    identity = id(target)
    if identity in seen:
        return
    seen.add(identity)
    if kind == "component":
        target.restore_state(state["value"])
        return
    if kind == "mapping":
        for name, item in state["items"].items():
            if item["kind"] == "scalar":
                target[name] = item["value"]
            else:
                _restore(target[name], item, seen=seen, path=f"{path}.{name}")
        return
    if kind == "sequence":
        for index, item in enumerate(state["items"]):
            if item["kind"] == "scalar":
                if isinstance(target, list):
                    target[index] = item["value"]
                elif target[index] != item["value"]:
                    raise SnapshotError(
                        f"Cannot restore changed immutable scalar at {path}[{index}]."
                    )
            else:
                _restore(target[index], item, seen=seen, path=f"{path}[{index}]")
        return
    for name, item in state["attributes"].items():
        current = getattr(target, name)
        if item["kind"] == "scalar":
            setattr(target, name, item["value"])
        else:
            _restore(current, item, seen=seen, path=f"{path}.{name}")


def _is_state_value(value: Any, name: str) -> bool:
    import torch

    if isinstance(value, torch.Tensor) or _is_torch_array(value):
        return True
    if isinstance(value, (Mapping, list, tuple)) or hasattr(value, "__dict__"):
        return True
    if isinstance(value, (bool, int, float)):
        return any(word in name.lower() for word in _SCALAR_STATE_WORDS)
    return False


def _is_torch_array(value: Any) -> bool:
    import torch

    return isinstance(getattr(value, "_tensor", None), torch.Tensor) and hasattr(
        value, "wp_array"
    )


__all__ = ["capture_state_tree", "restore_state_tree", "validate_state_tree"]
