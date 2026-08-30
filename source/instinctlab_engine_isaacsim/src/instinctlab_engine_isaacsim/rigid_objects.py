"""Isaac lowering for portable mesh-backed rigid objects."""

from __future__ import annotations

from typing import Any


_MESH_FORMATS = frozenset({".fbx", ".obj", ".stl"})
_USD_FORMATS = frozenset({".usd", ".usda", ".usdc"})


def rigid_object_conformance(ref: Any) -> dict[str, Any]:
    report = ref.resource_report("isaacsim")
    resource = ref.resource_path("isaacsim")
    suffix = resource.suffix.lower()
    if suffix not in _MESH_FORMATS | _USD_FORMATS:
        raise ValueError(
            f"Isaac rigid object {ref.name!r} has unsupported resource format {suffix!r}."
        )
    report["resource_format"] = suffix
    report["native_load"] = "direct_usd" if suffix in _USD_FORMATS else "mesh_to_usd"
    return report


def _with_physics_material(spawn: Any, friction: float) -> Any:
    original_func = spawn.func

    def spawn_with_material(
        prim_path: str,
        cfg: Any,
        translation=None,
        orientation=None,
    ):
        from pxr import Usd, UsdPhysics, UsdShade

        import isaaclab.sim as sim_utils

        prim = original_func(
            prim_path,
            cfg.replace(func=original_func),
            translation=translation,
            orientation=orientation,
        )
        stage = sim_utils.get_current_stage()
        for root_path in sim_utils.find_matching_prim_paths(prim_path):
            material_path = f"{root_path}/physicsMaterial"
            material = sim_utils.RigidBodyMaterialCfg(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
            )
            material.func(material_path, material)
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                raise RuntimeError(f"Isaac rigid object did not spawn at {root_path!r}.")
            collision_prims = [
                child
                for child in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if child.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if not collision_prims:
                raise RuntimeError(
                    f"Isaac rigid object at {root_path!r} has no collision geometry "
                    "for its declared material binding."
                )
            material_prim = stage.GetPrimAtPath(material_path)
            binding_api = UsdShade.MaterialBindingAPI.Apply(root)
            binding_api.Bind(
                UsdShade.Material(material_prim),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        return prim

    return spawn.replace(func=spawn_with_material)


def rigid_object_cfg(ref: Any, *, prim_path: str) -> Any:
    """Build a native rigid-object config after Isaac has bootstrapped."""
    resolved = ref.for_engine("isaacsim")
    report = rigid_object_conformance(ref)
    resource = ref.resource_path("isaacsim")
    suffix = str(report["resource_format"])

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=resolved.collision_enabled
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=resolved.kinematic
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=resolved.mass)
    if suffix in _MESH_FORMATS:
        converter = MeshConverter(
            MeshConverterCfg(
                asset_path=str(resource),
                scale=resolved.scale,
                mass_props=mass_props,
                collision_props=collision_props,
                rigid_props=rigid_props,
                mesh_collision_props=(
                    sim_utils.ConvexDecompositionPropertiesCfg()
                    if resolved.collision_enabled
                    else None
                ),
            )
        )
        usd_path = converter.usd_path
        scale = None
        spawn_mass_props = None
        spawn_collision_props = None
        spawn_rigid_props = None
    else:
        usd_path = str(resource)
        scale = resolved.scale
        spawn_mass_props = mass_props
        spawn_collision_props = collision_props
        spawn_rigid_props = rigid_props

    spawn = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=scale,
        mass_props=spawn_mass_props,
        collision_props=spawn_collision_props,
        rigid_props=spawn_rigid_props,
    )
    if resolved.collision_enabled:
        spawn = _with_physics_material(spawn, resolved.friction)
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=resolved.initial_position,
            rot=resolved.initial_quaternion_wxyz,
            lin_vel=resolved.initial_linear_velocity,
            ang_vel=resolved.initial_angular_velocity,
        ),
    )


__all__ = ["rigid_object_cfg", "rigid_object_conformance"]
