"""SDK-free fixture stand-ins and the external actuator runtime bridge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatefulActuatorCfgBase:
    joint_names_expr: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()
    effort_limit: float = 3.0
    stiffness: float = 2.0
    damping: float = 0.1

    def __post_init__(self) -> None:
        self.joint_names_expr = tuple(self.joint_names_expr)
        self.joint_names = tuple(self.joint_names)

    def build(self, num_envs: int) -> StatefulActuator:
        return StatefulActuator(
            num_envs=num_envs,
            stiffness=self.stiffness,
            effort_limit=self.effort_limit,
        )


class StatefulActuator:
    """One-step delayed proportional effort with partial reset."""

    def __init__(self, *, num_envs: int, stiffness: float, effort_limit: float):
        self.instinctlab_model_id = "fixture.stateful.v1"
        self.transmission_type = "joint"
        self.target_names = ("joint",)
        self.target_ids = (0,)
        self.stiffness = float(stiffness)
        self.effort_limit = float(effort_limit)
        self._pending = [0.0] * num_envs
        self.applied_effort = [0.0] * num_envs

    def compute(self, commands) -> list[float]:
        if len(commands) != len(self._pending):
            raise ValueError("one command is required for every environment")
        self.applied_effort = list(self._pending)
        limit = self.effort_limit
        self._pending = [
            max(-limit, min(limit, self.stiffness * float(command)))
            for command in commands
        ]
        return list(self.applied_effort)

    def reset(self, env_ids) -> None:
        for env_id in env_ids:
            self._pending[env_id] = 0.0
            self.applied_effort[env_id] = 0.0


class RuntimeAdapter:
    def matches(self, actuator: object) -> bool:
        return (
            getattr(actuator, "instinctlab_model_id", None)
            or getattr(getattr(actuator, "cfg", None), "instinctlab_model_id", None)
        ) == "fixture.stateful.v1"

    def stiffness_groups(self, env, asset, actuator):
        del asset
        joint_ids = getattr(actuator, "target_ids", None)
        if joint_ids is None:
            joint_ids = actuator.joint_indices
        global_ctrl_ids = getattr(actuator, "global_ctrl_ids", None)
        model = getattr(getattr(env, "sim", None), "model", None)
        gain_parameters = getattr(model, "actuator_gainprm", None)
        if global_ctrl_ids is not None and gain_parameters is not None:
            position_ids = global_ctrl_ids[: len(joint_ids)]
            if gain_parameters.ndim == 3:
                stiffness = gain_parameters[:, position_ids, 0]
            else:
                stiffness = gain_parameters[position_ids, 0]
            return ((joint_ids, stiffness),)
        stiffness = getattr(actuator, "stiffness", None)
        if stiffness is None:
            stiffness = actuator.cfg.stiffness
        return ((joint_ids, stiffness),)

    def effort_limit_for_joint(
        self, env, asset, actuator: StatefulActuator, local_index: int
    ):
        import torch

        if hasattr(asset, "indexing") and hasattr(env, "sim"):
            joint_id = int(actuator.target_ids[local_index])
            global_joint_id = int(asset.indexing.joint_ids[joint_id])
            ranges = env.sim.model.jnt_actfrcrange
            if ranges.ndim == 3:
                return ranges[:, global_joint_id].abs().max(dim=-1).values
            return ranges[global_joint_id].abs().max()

        return torch.full(
            (asset.data.joint_vel.shape[0],),
            actuator.effort_limit,
            device=asset.data.joint_vel.device,
            dtype=asset.data.joint_vel.dtype,
        )


RUNTIME_ADAPTER = RuntimeAdapter()


class ImuRuntime:
    """Scalar stand-in that makes latency/history and partial reset observable."""

    def __init__(self, sensor, context) -> None:
        self.sensor = sensor
        self.context = context
        self._delay_steps = round(sensor.latency / sensor.update_period)
        self._queues = [[] for _ in range(context.num_envs)]
        self.timestamp = 0.0

    def tick(self, samples, timestamp: float) -> list[float]:
        if len(samples) != len(self._queues):
            raise ValueError("one sample is required for every environment")
        output = []
        for queue, sample in zip(self._queues, samples, strict=True):
            queue.append(float(sample))
            if len(queue) > self.sensor.history_length + self._delay_steps + 1:
                queue.pop(0)
            output.append(
                queue[-self._delay_steps - 1] if len(queue) > self._delay_steps else 0.0
            )
        self.timestamp = float(timestamp)
        return output

    def reset(self, env_ids) -> None:
        for env_id in env_ids:
            self._queues[env_id].clear()
