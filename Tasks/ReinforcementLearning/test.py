import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Projects.Model.RL import *
from Projects.Utils.RL_tools import Env_CliffWalking, Env_FrozenLake

# env = Env_CliffWalking(height=3, width=3)
env = Env_FrozenLake()
# agent = DP_ValueIteration(env, 0.001, 0.9)
# agent = DP_PolicyIteration(env, 0.001, 0.9)
agent = SARSA_nstep(env, 3, 0.1, 0.1, 0.9)
# agent = SARSA(env, 0.1, 0.1, 0.9)
agent(150)
print(agent.pi)
