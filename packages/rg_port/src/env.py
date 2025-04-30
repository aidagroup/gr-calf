from packages.goalagent.env import PendulumQuanser
from packages.goalagent.envs.utils import (
    StateInitRandomSamplerSimulator,
    UniformStateInitGenerator,
    CasADi,
)
from packages.goalagent.env import RgEnv
import numpy as np
import gymnasium as gym
from packages.goalagent.utilities import make_env
from packages.goalagent.running_objective import GymPendulumRunningObjective
from regelum import Node
from packages.goalagent.system import ThreeWheeledRobotKinematic
from packages.goalagent.running_objective import QuadraticRunningObjective


class PendulumQuanserNode(Node):
    _step_size = 0.01

    def __init__(self, system: PendulumQuanser):
        super().__init__(
            inputs=["agent_1.action"],
            name="env",
            is_root=True,
            step_size=self._step_size,
        )
        simulator = StateInitRandomSamplerSimulator(
            system=system,
            state_init=UniformStateInitGenerator(
                bounds=np.array([[np.pi - np.pi / 100, np.pi + np.pi / 100], [-1, 1]]),
            ),
            time_final=10,
            max_step=self._step_size,
            action_init=np.zeros((1,)),
        )
        running_objective = GymPendulumRunningObjective()
        env = RgEnv(simulator=simulator, running_objective=running_objective)
        self.envs = gym.vector.SyncVectorEnv([make_env(env) for i in range(1)])
        obs, info = self.envs.reset()
        self.state = self.define_variable(
            "observation",
            shape=(2,),
            value=obs,
        )
        self.info = self.define_variable(
            "info",
            shape=(1,),
            value=info,
        )

    def step(self) -> None:
        action = self.resolved_inputs.find("agent_1.action")
        obs, _, _, _, info = self.envs.step(action.value)
        self.state.value = obs
        self.info.value = info


class ThreeWheeledRobotKinematicNode(Node):
    _step_size = 0.01

    def __init__(self, system: ThreeWheeledRobotKinematic):
        super().__init__(
            inputs=["agent_1.action"],
            name="env",
            is_root=True,
            step_size=self._step_size,
        )
        simulator = CasADi(
            system=system,
            state_init=np.array([[5.0, 5.0, 2 * np.pi / 3.0]]),
            time_final=5,
            max_step=self._step_size,
            action_init=np.zeros((1, 2)),
        )
        running_objective = QuadraticRunningObjective(
            weights=np.array([1.0, 10.0, 1.0, 0.0, 0.0])
        )
        env = RgEnv(simulator=simulator, running_objective=running_objective)
        self.envs = gym.vector.SyncVectorEnv([make_env(env) for i in range(1)])
        obs, info = self.envs.reset()
        self.state = self.define_variable(
            "observation",
            shape=(3,),
            value=obs,
        )
        self.info = self.define_variable(
            "info",
            shape=(1,),
            value=info,
        )

    def step(self) -> None:
        action = self.resolved_inputs.find("agent_1.action")
        obs, _, _, _, info = self.envs.step(action.value)
        self.state.value = obs
        self.info.value = info
