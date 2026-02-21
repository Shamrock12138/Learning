#                       多智能体强化学习模型
#                           2026/2/14
#                            shamrock

from tqdm import tqdm

from myTools.Utils.tools import *
from myTools.Utils.config import *
from myTools.Utils.RL_config import *
from myTools.Utils.MARL_config import *

#---------------------- Independent 框架 -------------------------
#                          2026/2/14

class Independent_Trainer(RL_Trainer):
  '''
    Independent 框架的训练器
  '''
  def __init__(self, rl:RL_Model, rl_config, agents_num, 
               env:MARL_EnvConfig, device):
    '''
      params:
        rl - 传统的RL算法
        rl_config - RL算法的参数配置文件路径(.json)
        agents_num - 需要agents的数量
    '''
    model_params = utils_readParams(rl_config, 'model')
    buffer_params = utils_readParams(rl_config, 'buffer')
    trainer_params = utils_readParams(rl_config, 'trainer')
    utils_setAttr(self, buffer_params)
    utils_setAttr(self, trainer_params)
    utils_autoAssign(self)
    self.agents = rl(**model_params, device=device)
    # self.agents = [rl(**model_params) for _ in range(n_agents)]
    self.name = f'Independent {rl.name}'

  def train(self, replay_buffer:utils_ReplayBuffer=None):
    '''
      训练 episodes_num 次，
        params:
          is_on_policy - 使用的 RL 算法是否是 on-policy
    '''
    history = {'rewards':[]}
    if self.is_on_policy:
      transition_dict = {
        'states': [], 
        'actions': [], 
        'next_states': [], 
        'rewards': [], 
        'dones': []
        }
      transitions = [transition_dict]*self.agents_num
      with tqdm(total=self.episodes_num, desc=f'{self.name}') as pbar:
        for episode in range(self.episodes_num):
          done = [False]*self.agents_num
          state, _ = self.env.reset()
          total_rewards = np.zeros(self.agents_num)
          while not all(done):
            actions = np.array([self.agent.take_action(state[i])
              for i in range(self.agents_num)])
            next_state, reward, done, info = self.env.step(actions)
            for agent_i in range(self.agents_num):
              transitions[agent_i]['states'].append(state[agent_i])
              transitions[agent_i]['actions'].append(actions[agent_i])
              transitions[agent_i]['next_states'].append(next_state[agent_i])
              transitions[agent_i]['rewards'].append(reward[agent_i])
              transitions[agent_i]['dones'].append(done[agent_i])
            state = next_state
            total_rewards += reward
          history['rewards'].append(total_rewards)
          for i in range(self.agents_num):
            self.agent.update(transitions[i])
          pbar.update(1)
    else:
      with tqdm(total=self.episodes_num, desc=f'{self.name}') as pbar:
        for episode in range(self.episodes_num):
          done = [False]*self.agents_num
          state, _ = self.env.reset()
          total_rewards = np.zeros(self.agents_num)
          while not all(done):
            actions = np.array([self.agent.take_action(state[i])
              for i in range(self.agents_num)])
            next_state, reward, done, _ = self.env.step(actions)
            replay_buffer.add_sample(Sample(state, actions, reward, next_state, done))
            state = next_state
            total_rewards += reward
            if len(replay_buffer) > self.min_size:
              transition_dict = replay_buffer.sample_sample(self.batch_size)
              self.agent.update(transition_dict)
          history['rewards'].append(total_rewards)
          pbar.update(1)
    return history

  def eval(self):
    '''
      测试
    '''


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
