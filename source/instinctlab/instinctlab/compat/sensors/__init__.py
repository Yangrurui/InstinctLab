"""Engine-neutral sensor reads.

Sensor declarations are lowered to native engine objects.  The functions exported here are the
single runtime boundary used by shared MDP terms: they normalize names, shapes, axis order and
missing-value conventions without pretending that physically different quantities are equal.
"""

from .contact import (
    air_time,
    contact_force_history,
    contact_time,
    element_ids,
    element_names,
    forget,
    in_contact,
    sensor_engine,
)
from .ray import depth_image, ray_hits_w
from .volume_points import (
    registered_cylinder_count,
    require_volume_points_registered,
    volume_points_penetration_offset,
    volume_points_vel_w,
)

__all__ = [
    "air_time",
    "contact_force_history",
    "contact_time",
    "depth_image",
    "element_ids",
    "element_names",
    "forget",
    "in_contact",
    "ray_hits_w",
    "registered_cylinder_count",
    "require_volume_points_registered",
    "sensor_engine",
    "volume_points_penetration_offset",
    "volume_points_vel_w",
]
