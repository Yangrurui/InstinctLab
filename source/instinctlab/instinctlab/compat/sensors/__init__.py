"""Engine-neutral sensor reads.

Sensor declarations are lowered to native engine objects.  The functions exported here are the
single runtime boundary used by shared MDP terms: they normalize names, shapes, axis order and
missing-value conventions without pretending that physically different quantities are equal.
"""

from .contact import (
    air_time,
    contact_force_history,
    contact_from_force,
    contact_time,
    element_ids,
    element_names,
    forget,
    in_contact,
    sensor_engine,
    step_contact_clock,
)
from .ray import (
    camera_pose_for_alignment,
    depth_image,
    ray_hits_w,
    ray_origin_z_w,
    refuse_unhonored_ray_alignment,
)
from .volume_points import (
    cylinder_penetration_offset,
    grid3d_points,
    link_linear_velocity_from_com,
    point_velocity_from_link,
    registered_cylinder_count,
    require_volume_points_registered,
    volume_points_penetration_offset,
    volume_points_vel_w,
)

__all__ = [
    "air_time",
    "camera_pose_for_alignment",
    "contact_from_force",
    "contact_force_history",
    "contact_time",
    "depth_image",
    "element_ids",
    "element_names",
    "forget",
    "in_contact",
    "cylinder_penetration_offset",
    "grid3d_points",
    "link_linear_velocity_from_com",
    "point_velocity_from_link",
    "ray_hits_w",
    "ray_origin_z_w",
    "refuse_unhonored_ray_alignment",
    "registered_cylinder_count",
    "require_volume_points_registered",
    "sensor_engine",
    "step_contact_clock",
    "volume_points_penetration_offset",
    "volume_points_vel_w",
]
