# Shadowing engine semantics

The shared shadowing declaration fixes the policy-facing joint schema to the 29-DoF G1 depth-first
order. Joint position, joint velocity, action, last action and motion-reference tensors all use that
schema. Isaac's imported articulation has a different breadth-first native order, so the Isaac
adapter resolves and writes joints by name. The MJCF natural joint order already equals the shared
depth-first order, but MJLab still receives an explicit ordered selector. A checkpoint manifest
records the complete TaskSpec hash and joint list; a different ordering is rejected before load.

The final G1 configs in both reference projects replace the base robot's actuator table with the
BeyondMimic plant. The shared RobotSpec therefore carries the corresponding per-joint stiffness,
damping, armature, effort and velocity limits, plus the reference action scale
`0.25 * effort_limit / stiffness`. Neither final shadowing reference uses actuator delay.

The following differences are intentional and must not be interpreted as numerical portability:

- Isaac uses PhysX `ImplicitActuatorCfg`; MJLab uses MuJoCo `BuiltinPdActuatorCfg`. Equal declared
  gains and limits do not make the two solvers' constraint integration or saturation identical.
- MJLab groups joints with equal gains into seven actuator configs. Their config tuple follows the
  shared DFS declaration's first appearance, while policy selection remains explicitly DFS. This
  grouping does not define the order of joint-indexed MuJoCo state or `qfrc_actuator`.
- Isaac applied torque and MuJoCo `qfrc_actuator` are engine-native quantities. Rewards using them
  must match the corresponding reference implementation; cross-engine value equality is not a
  valid invariant.
