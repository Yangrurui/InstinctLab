# Unified shadowing tasks

Shadowing is declared as `TaskSpec` and compiled by the selected engine adapter.  Importing this
package does not import Isaac Lab, MJLab, Gym or either native environment configuration.

Use the generic entry points:

```bash
python scripts/train.py --engine isaacsim --task Instinct-Shadowing-WholeBody-Plane-G1-v0
python scripts/train.py --engine mjlab --task Instinct-Shadowing-WholeBody-Plane-G1-v0
python scripts/play.py --engine mjlab --task Instinct-Shadowing-WholeBody-Plane-G1-v0 --checkpoint PATH
```

The registry contains whole-body, perceptive, perceptive VAE, perceptive HOI, BeyondMimic and
Perceptive OneMotion train/play contracts. Their files follow main and InstinctMJ: each family has
its own `*_env_cfg.py`, and each G1 task keeps its concrete values in the corresponding
`config/g1/*_cfg.py`. No task-local CLI mutates configuration at import time.

Task configuration uses the same base/concrete class shape as main and InstinctMJ. A concrete task
can change one term without copying the complete reward or observation group:

```python
class G1PerceptiveVaeEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    def __init__(self) -> None:
        super().__init__(...)
        self.rewards.action_rate_l2 = self.rewards.action_rate_l2.replace(weight=-0.2)
```

The replacement belongs in the concrete task class. Do not pass an override dictionary into the
registry factory and do not modify a built `TaskSpec`.

Native contact, actuator, solver and object semantics intentionally remain in the corresponding
engine adapter; the shared task layer only declares the portable boundary.
