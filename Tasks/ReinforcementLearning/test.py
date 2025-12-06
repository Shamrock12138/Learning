import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Projects.Model.RL import *
from Projects.Utils.RL_tools import *

env = Env_CliffWalking(height=5, width=5)
# env = Env_FrozenLake()
# env = Env_CartPole()
# env = Env_AimBall()

# agent = DP_ValueIteration(env, 0.001, 0.9)
# agent = DP_PolicyIteration(env, 0.001, 0.9)
# agent = SARSA_nstep(env, 5, 0.1, 0.1, 0.9)
# agent = SARSA(env, 0.1, 0.1, 0.9)
# agent = Q_Learning(env, 0.3, 0.1, 0.9)
agent = Dyna_Q(env, 0.6, 0.1, 0.9, 3)

agent(quit_cnt=5, diff_tol=1e-4)
# print(agent.Q)
# print(agent.pi)
env.render(agent.pi)

