import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from myTools.Model.RL import *
from myTools.Utils.RL_tools import *
from myTools.Utils.tools import *
from myTools.Utils.RL_Env import *

device = utils_getDevice()
working_path = r'F:\\GraduateStudent\\Code\\Git_Learning\\Knowledge\\ReinforcementLearning\\Learning\\'

# env = Env_CliffWalking(height=5, width=5)
# env = Env_FrozenLake()
env = Env_CartPole()
# env = Env_Pendulum()
# env = Env_AimBall()
# env = Env_AimBallDynamic(target_move_mode='uniform')
env_name = env.name

# agent = DP_ValueIteration(env, 0.001, 0.9)
# agent = DP_PolicyIteration(env, 0.001, 0.9)
# agent = SARSA_nstep(env, 5, 0.1, 0.1, 0.9)
# agent = SARSA(env, 0.1, 0.1, 0.9)
# agent = Q_Learning(env, 0.3, 0.1, 0.9)
# agent = Dyna_Q(env, 0.6, 0.1, 0.9, 3)
# agent = DQN(env, env._states_num, 256, env._actions_num, 2e-3, 0.98, 0.01, 30, device)
# agent = DoubleDQN(env, env._states_num, 256, env._actions_num, 2e-3, 0.98, 0.01, 30, device)
# agent = DuelingDQN(env, env._states_num, 256, env._actions_num, 2e-3, 0.98, 0.01, 30, device)
# agent = REINFORCE(env, env._states_num, 256, env._actions_num, 1e-3, 0.98, device)
agent = AC(env, env._states_num, 128, env._actions_num, 1e-3, 1e-2, 0.98, device)
# agent = SAC_Discrete(env, env._states_num, 256, env._actions_num, 1e-3, 1e-2,
#                      1e-2, -1, 0.005, 0.98, device)
RL_name = agent.name

train_episodes = 1000
model_save_path = working_path+RL_name+r'\\model\\'
history_save_path = working_path+RL_name+r'\\history\\'
split_string = '_'
train_episodes_string = str(train_episodes)

# agent(train_episodes)
# agent.show_history(history_save_path, env_name+split_string+train_episodes_string+'.png')
# agent.save_model(model_save_path, env_name+split_string+train_episodes_string+'.pt')
agent.load_model(model_save_path, env_name+split_string+train_episodes_string+'.pt')
agent.render(10)

