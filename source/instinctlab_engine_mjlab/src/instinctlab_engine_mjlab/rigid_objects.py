"""MJLab lowering for portable mesh-backed rigid objects."""

from __future__ import annotations

from typing import Any


_MESH_FORMATS = frozenset({".obj", ".stl"})


def rigid_object_conformance(ref: Any) -> dict[str, Any]:
    report = ref.resource_report("mjlab")
    resource = ref.resource_path("mjlab")
    suffix = resource.suffix.lower()
    if suffix not in _MESH_FORMATS:
        raise ValueError(
            f"MJLab rigid object {ref.name!r} has unsupported resource format {suffix!r}."
        )
    report["resource_format"] = suffix
    report["native_load"] = "mujoco_mesh"
    return report


def rigid_object_cfg(ref: Any) -> Any:
    """Build a native entity config with explicit spawn and reset state."""
    resolved = ref.for_engine("mjlab")
    rigid_object_conformance(ref)
    resource = ref.resource_path("mjlab")

    import mujoco
    from mjlab.entity import EntityCfg

    def object_spec():
        native = mujoco.MjSpec()
        mesh = native.add_mesh(
            name=f"{resolved.name}_mesh",
            file=str(resource),
            scale=resolved.scale,
        )
        body = native.worldbody.add_body(
            name=resolved.name,
            mocap=resolved.kinematic,
        )
        if not resolved.kinematic:
            body.add_freejoint(name=f"{resolved.name}_freejoint")
        body.add_geom(
            name=f"{resolved.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            mass=resolved.mass,
            group=2,
            contype=1 if resolved.collision_enabled else 0,
            conaffinity=1 if resolved.collision_enabled else 0,
            friction=(resolved.friction, 0.005, 0.0001),
        )
        return native

    return EntityCfg(
        spec_fn=object_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=resolved.initial_position,
            rot=resolved.initial_quaternion_wxyz,
            lin_vel=resolved.initial_linear_velocity,
            ang_vel=resolved.initial_angular_velocity,
            joint_pos={},
            joint_vel={},
        ),
    )


__all__ = ["rigid_object_cfg", "rigid_object_conformance"]
