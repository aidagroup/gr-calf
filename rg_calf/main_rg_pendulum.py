from packages.rg_port import AgentCALFQNode, PendulumQuanserNode
from packages.goalagent.env import PendulumQuanser
from packages.goalagent.utilities import save_episodic_data
from regelum import Graph
import mlflow

TOTAL_TIMESTEPS = 50000

system = PendulumQuanser()
agent = AgentCALFQNode(system)
env = PendulumQuanserNode(system)

graph = Graph(
    nodes=[agent, env],
    initialize_inner_time=True,
    debug=True,
    states_to_log=["agent_1.action", "env_1.observation", "env_1.info"],
    logger_cooldown=1,
)

graph.resolve(graph.variables)

# episodic_observations = []
# episodic_actions = []


# for global_step in range(TOTAL_TIMESTEPS):
#     graph.step()
#     obs = env.state.value
#     action = agent.action.value
#     episodic_observations.append(obs)
#     episodic_actions.append(action)

#     if "final_info" in env.info.value:
#         agent.agent.reset(env.state.value, global_step=global_step)
#         mlflow.log_metric(
#             "charts/relax_probability",
#             agent.agent.relax_probability,
#             step=global_step,
#         )
#         for info in env.info.value["final_info"]:
#             if info and "episode" in info:
#                 print(
#                     f"global_step={global_step}, episodic_return={info['episode']['r']}"
#                 )
#                 save_episodic_data(
#                     info,
#                     global_step,
#                     episodic_observations,
#                     episodic_actions,
#                     ["angle", "angle_vel"],
#                     ["torque"],
#                 )
#                 episodic_observations.clear()
#                 episodic_actions.clear()
