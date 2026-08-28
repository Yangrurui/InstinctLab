# PND Adam SP assets

This directory follows the same layout as `resources/unitree_g1`:

- `meshes/`: the single shared set of visual meshes;
- `urdf/adam_sp_23_dof.urdf`: the 23-DOF model (the wrist and hand joints are fixed);
- `xml/adam_sp.xml`: the 29-DOF MuJoCo model;
- `xml/adam_sp_23_dof.xml`: the 23-DOF MuJoCo model.

The simulator configurations are intentionally separate and explicit:

- `instinctlab.assets.adam_sp.mjlab` owns the MJLab robot and actuator configuration;
- `instinctlab.assets.adam_sp.isaacsim` owns the Isaac Sim robot and actuator configuration.

The current URDF fixes both wrists, so the 29-DOF model is available only through mjlab.
