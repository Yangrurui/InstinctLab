"""Complete training selection for G1 locomotion."""

from instinctlab_engine.spec import AgentSpec

G1_LOCOMOTION_TRAINING_CFG = AgentSpec(
    runner="instinctlab.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg"
)
