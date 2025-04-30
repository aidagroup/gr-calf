from packages.goalagent.calfq import AgentCALFQ
import gymnasium as gym
from packages.goalagent.utilities import save_source_code, save_episodic_data, make_env
from packages.goalagent import repo_root
import os
import mlflow
from packages.goalagent.running_objective import GymPendulumRunningObjective
import numpy as np
from packages.goalagent.env import (
    PendulumQuanser,
    PendulumStabilizingPolicy,
    PendulumGoalReachingFunction,
)
from packages.goalagent.envs.utils import (
    StateInitRandomSamplerSimulator,
    UniformStateInitGenerator,
)
from packages.goalagent.env import RgEnv

config_path = repo_root / "presets" / "td3_sac_ours"
config_name = os.path.basename(__file__)[: -len(".py")]
TOTAL_TIMESTEPS = 50000


def main():
    save_source_code()
    system = PendulumQuanser()
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
    calfq_agent = AgentCALFQ(
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
    simulator = StateInitRandomSamplerSimulator(
        system=system,
        state_init=UniformStateInitGenerator(
            bounds=np.array([[np.pi - np.pi / 100, np.pi + np.pi / 100], [-1, 1]]),
        ),
        time_final=10,
        max_step=0.01,
    )
    env = RgEnv(simulator=simulator, running_objective=running_objective)

    envs = gym.vector.SyncVectorEnv([make_env(env) for i in range(1)])

    obs, info = envs.reset()
    calfq_agent.reset(obs, global_step=0)
    mlflow.log_metric("charts/relax_probability", calfq_agent.relax_probability, step=0)

    episodic_observations = []
    episodic_actions = []

    for global_step in range(TOTAL_TIMESTEPS):
        action = calfq_agent.get_action(obs)
        episodic_observations.append(obs)
        episodic_actions.append(action)
        obs, _, _, _, infos = envs.step(action)

        if "final_info" in infos:
            calfq_agent.reset(obs, global_step=global_step)
            mlflow.log_metric(
                "charts/relax_probability",
                calfq_agent.relax_probability,
                step=global_step,
            )
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    save_episodic_data(
                        info,
                        global_step,
                        episodic_observations,
                        episodic_actions,
                        ["angle", "angle_vel"],
                        ["torque"],
                    )
                    episodic_observations.clear()
                    episodic_actions.clear()


if __name__ == "__main__":
    main()
