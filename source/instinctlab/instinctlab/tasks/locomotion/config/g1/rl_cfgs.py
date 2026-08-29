"""Complete training selection for G1 locomotion."""

from instinctlab.spec import AgentSpec

G1_LOCOMOTION_TRAINING_CFG = AgentSpec(
    runner="instinctlab.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg"
)

__all__ = ["G1_LOCOMOTION_TRAINING_CFG"]
