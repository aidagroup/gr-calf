from packages.goalagent.calfq import AgentCALFQ
from packages.goalagent.env import PendulumQuanser
from packages.goalagent.env import PendulumStabilizingPolicy
from packages.goalagent.running_objective import GymPendulumRunningObjective
from packages.goalagent.running_objective import QuadraticRunningObjective
from packages.goalagent.env import PendulumGoalReachingFunction
from regelum import Node
import numpy as np
from packages.goalagent.utilities import rg
from packages.goalagent.system import ThreeWheeledRobotKinematic
from packages.goalagent.env import ThreeWheeledRobotKinematicStabilizingPolicy


class AgentCALFQNode(Node):
    def __init__(self, system: PendulumQuanser):
        super().__init__(inputs=["env_1.observation"], name="agent")
        nominal_policy = PendulumStabilizingPolicy(
            gain=0.03,
            action_min=-0.1,
            action_max=0.1,
            switch_loc=np.cos(np.pi / 10),
            switch_vel_loc=0.2,
            pd_coeffs=np.array([0.6, 0.2]),
            system=system,
        )
        running_objective = GymPendulumRunningObjective()
        goal_reaching_func = PendulumGoalReachingFunction(goal_threshold=0.4)
        self.agent = AgentCALFQ(
            nominal_policy=nominal_policy,
            system=system,
            critic_struct="quad-mix",
            critic_weights_init=np.array(
                [7196.45, 323.51, 34839.243, -97235.899, 13453.018]
            ),
            critic_learn_rate=0.001,
            critic_num_grad_steps=1,
            buffer_size=10,
            actor_opt_method="SLSQP",
            actor_opt_options={"maxiter": 40, "disp": False},
            use_grad_descent=True,
            use_decay_constraint=False,
            use_kappa_constraint=False,
            check_persistence_of_excitation=True,
            critic_weight_change_penalty_coeff=0.0,
            goal_reaching_func=goal_reaching_func,
            relax_probability_max=0.999,
            relax_probability_min=0.01,
            running_objective=running_objective,
            action_sampling_period=0.01,
        )
        self.action = self.define_variable(
            "action",
            shape=(1,),
            value=rg.zeros((1, 1)),
        )
        self.agent.reset(rg.zeros((1, 2)), global_step=0)

    def step(self) -> None:
        observation = self.resolved_inputs.find("env_1.observation")
        action = self.agent.get_action(observation.value)
        self.action.value = action


class AgentCALFQNodeRobot(Node):
    def __init__(self, system: ThreeWheeledRobotKinematic):
        super().__init__(inputs=["env_1.observation"], name="agent")
        nominal_policy = ThreeWheeledRobotKinematicStabilizingPolicy(K=3)
        running_objective = QuadraticRunningObjective(
            weights=np.array([1.0, 10.0, 1.0, 0.0, 0.0])
        )
        self.agent = AgentCALFQ(
            nominal_policy=nominal_policy,
            system=system,
            critic_struct="quad-mix",
            critic_weights_init=np.array(
                [
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ]
            ),
            critic_learn_rate=0.001,
            critic_num_grad_steps=1,
            buffer_size=10,
            actor_opt_method="SLSQP",
            actor_opt_options={"maxiter": 40, "disp": False},
            use_grad_descent=True,
            use_decay_constraint=False,
            use_kappa_constraint=False,
            check_persistence_of_excitation=True,
            critic_weight_change_penalty_coeff=0.0,
            goal_reaching_func=lambda _: False,
            relax_probability_stabilize_global_step=5000,
            relax_probability_max=0.49,
            relax_probability_min=0.01,
            running_objective=running_objective,
            action_sampling_period=0.01,
        )
        self.action = self.define_variable(
            "action",
            shape=(2,),
            value=rg.zeros((1, 2)),
        )
        self.agent.reset(rg.zeros((1, 3)), global_step=0)

    def step(self) -> None:
        observation = self.resolved_inputs.find("env_1.observation")
        action = self.agent.get_action(observation.value)
        self.action.value = action
