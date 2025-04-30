import numpy as np
from packages.goalagent.system import InvertedPendulum as Pendulum
from packages.goalagent.system import ThreeWheeledRobotKinematic
from packages.goalagent.utilities import rg
from typing import Optional
from packages.goalagent.simulator import Simulator
import gymnasium as gym
from typing import Callable


class RgEnv(gym.Env):
    def __init__(
        self,
        simulator: Simulator,
        running_objective: Callable[[np.ndarray], float],
        action_space: Optional[gym.spaces.Box] = None,
        observation_space: Optional[gym.spaces.Box] = None,
    ) -> None:
        self.simulator = simulator
        self.running_objective = running_objective
        action_bounds = np.array(simulator.system._action_bounds)
        if action_space is None:
            self.action_space = gym.spaces.Box(
                low=action_bounds[:, 0], high=action_bounds[:, 1]
            )
        else:
            self.action_space = action_space
        if observation_space is None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(simulator.system._dim_observation,)
            )
        else:
            self.observation_space = observation_space

    def step(self, u):
        self.simulator.receive_action(self.simulator.system.apply_action_bounds(u))
        costs = self.running_objective(self.state, u)
        sim_step = self.simulator.do_sim_step()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs(), -costs, False, sim_step is not None, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs(), {}

    def _get_obs(self):
        return self.simulator.system._get_observation(None, self.state, None)


class QuadraticRunningObjective:
    def __init__(self, weights: np.ndarray, biases: np.ndarray | float = 0.0) -> None:
        self.weights = weights
        self.biases = biases

    def __call__(self, state, action) -> float:
        return (np.hstack((state - self.biases, action)) ** 2 * self.weights).sum()


class PendulumQuanser(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 2

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_state, prototype=state)

        observation[0] = state[0]
        observation[1] = state[1]

        return observation


class PendulumQuanserWithGymObservation(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


class PendulumQuanserWithNormObservation(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


class Pendulum(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 1.0, "grav_const": 10.0, "length": 1.0}
    action_bounds = [[-2.0, 2.0]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


class PendulumStabilizingPolicy:
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        switch_vel_loc: float,
        pd_coeffs: np.ndarray,
        system: Pendulum,
    ):
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.switch_vel_loc = switch_vel_loc
        self.system = system

    @staticmethod
    def hard_switch(signal1: float, signal2: float, condition: bool):
        if condition:
            return signal1
        else:
            return signal2

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )
        if np.prod(observation.shape) == 3:
            cos_angle = observation[0, 0]
            sin_angle = observation[0, 1]

            angle = np.arctan2(sin_angle, cos_angle)
            angle_vel = observation[0, 2]
        elif np.prod(observation.shape) == 2:
            angle = observation[0, 0]
            angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = self.hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc
            or np.abs(angle_vel) > self.switch_vel_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ],
        )

    def buffer_current_observation_action(self, observation, action):
        """Buffers the current observation and action.

        This method is intended to fit the interface required by CALFQ.
        The controller does not utilize the buffered observation and action directly.

        Args:
        - observation: The current observed state.
        - action: The action taken.
        """

        pass

    def reset(self):
        pass


class PendulumGoalReachingFunction:

    def __init__(self, goal_threshold: float):
        self.goal_threshold = goal_threshold

    def __call__(self, observation: np.ndarray) -> bool:
        angle = observation[0, 0]
        return 1 - np.cos(angle) <= self.goal_threshold


class ThreeWheeledRobotKinematicStabilizingPolicy:
    """Scenario for non-inertial three-wheeled robot composed of three PID scenarios."""

    def __init__(self, K):
        """Initialize an instance of scenario.

        Args:
            K: gain of scenario
        """
        # super().__init__()
        self.K = K

    def get_action(self, observation):
        x = observation[0, 0]
        y = observation[0, 1]
        angle = observation[0, 2]

        angle_cond = np.arctan2(y, x)

        if not np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            omega = (
                -self.K
                * np.sign(angle - angle_cond)
                * rg.sqrt(rg.abs(angle - angle_cond))
            )
            v = 0
        elif not np.allclose((x, y), (0, 0), atol=1e-03) and np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            omega = 0
            v = -self.K * rg.sqrt(rg.norm_2(rg.hstack([x, y])))
        elif np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, 0, atol=1e-03
        ):
            omega = -self.K * np.sign(angle) * rg.sqrt(rg.abs(angle))
            v = 0
        else:
            omega = 0
            v = 0

        return rg.force_row(rg.hstack([v, omega]))

    def reset(self): ...
