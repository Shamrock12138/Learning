import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from myTools.Model.RL import *
from myTools.Utils.RL_tools import *
from myTools.Utils.tools import *
from myTools.Utils.RL_Env import *

# 当前工作目录（*修改）
working_path = r'F:\\GraduateStudent\\Code\\Git_Learning\\Knowledge\\ReinforcementLearning\\Learning\\'
# 当前使用的RL算法（*修改）
rl_name = 'DDPG'

device = utils_getDevice()
# 相关参数配置在相应算法文件夹下的config.json文件中
model_config = utils_readParams(working_path+rl_name+'/config.json', 'model')
trainer_config = utils_readParams(working_path+rl_name+'/config.json', 'trainer')

#---------------------- 环境 -------------------------
#                     2026/3/6

# env = Env_CliffWalking(height=5, width=5)
# env = Env_FrozenLake()
# env = Env_CartPole()
env = Env_Pendulum()
# env = Env_AimBall()
# env = Env_AimBallDynamic(target_move_mode='uniform')
env_name = env.name

#---------------------- 算法 -------------------------
#                     2026/3/6

# agent = DP_ValueIteration(env, 0.001, 0.9)
# agent = DP_PolicyIteration(env, 0.001, 0.9)
# agent = SARSA_nstep(env, 5, 0.1, 0.1, 0.9)
# agent = SARSA(env, 0.1, 0.1, 0.9)
# agent = Q_Learning(env, 0.3, 0.1, 0.9)
# agent = Dyna_Q(env, 0.6, 0.1, 0.9, 3)
# agent = DQN(env._states_num, env._actions_num, device, model_config)
# agent = DoubleDQN(env._states_num, env._actions_num, device, model_config)
# agent = DuelingDQN(env._states_num, env._actions_num, device, model_config)
# agent = REINFORCE(env._states_num, env._actions_num, device, model_config)
# agent = AC(env._states_num, env._actions_num, device, model_config)
# agent = SAC_Discrete(env, env._states_num, 256, env._actions_num, 1e-3, 1e-2,
#                      1e-2, -1, 0.005, 0.98, device)
agent = DDPG(env._states_num, env._actions_num, device, model_config)

#---------------------- 训练器 -------------------------
#                     2026/3/6

trainer = Trainer(agent, env, trainer_config)

if __name__ == '__main__':
  train_episodes = trainer_config['train_episodes']
  model_save_path = working_path+rl_name+r'\\model\\'
  history_save_path = working_path+rl_name+r'\\history\\'
  split_string = '_'
  other_string = ''

  trainer.train()
  trainer.show_history(history_save_path+env_name+split_string+str(train_episodes)+other_string+'.png')
  agent.save_model(model_save_path, env_name+split_string+str(train_episodes)+other_string+'.pt')
  agent.load_model(model_save_path, env_name+split_string+str(train_episodes)+other_string+'.pt')
  trainer.render()

