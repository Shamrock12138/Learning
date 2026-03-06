#                  Reinforcement Learning 工具函数
#                           2025/11/30
#                            shamrock
import numpy as np
import torch, random
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from myTools.Utils.config import *
from myTools.Utils.tools import *
from myTools.Utils.config import *
from myTools.Utils.RL_config import *

#---------------------- RL Trainer -------------------------
#                       2026/2/22

class Trainer(RL_TrainerConfig):
  def __init__(self, rl:RL_Model, env:ENV_INFO, config:dict) -> None:
    super().__init__(rl, env)
    utils_setAttr(self, config)
    self.train_history = {
      'episode_rewards': [],
    }
    self.eval_history = {
      'episode_rewards': [],
    }
    self.is_train = False

  def show_history(self, save_path=None):
    if self.is_train:
      utils_showHistory(self.train_history, list(self.train_history.keys()), f'{self.rl.name} on {self.env.name}, train', 
                        'episodes', '...', save_path)
    else:
      utils_showHistory(self.eval_history, list(self.eval_history.keys()), f'{self.rl.name} on {self.env.name}, eval', 
                        'episodes', '...', save_path)
      
  def render(self):
    '''
      渲染 times 趟动画
    '''
    self.env.eval()
    pbar = tqdm(iterable=range(self.render_times), desc='test')
    for T in pbar:
      done = False
      state, _ = self.env.reset()
      self.env.render()
      time.sleep(0.02)
      while not done:
        action = self.rl.take_action(state)
        state, _, done, _ = self.env.step(action)
        self.env.render()
        time.sleep(1/60)

  @utils_timer
  def train(self):
    """
      通用训练方法，适用于不同类型的RL算法
    """
    self.is_train = True
    if self.rl.is_on_policy == False and hasattr(self.rl, 'replay_buffer'):
      self.train_history['episode_rewards'] = train_off_policy(self.env, self.rl, self.train_episodes, 
                                                               self.rl.replay_buffer, self.min_size, self.batch_size)
    elif self.rl.is_on_policy == True:
      self.train_history['episode_rewards'] = train_on_policy(self.env, self.rl, self.train_episodes)
    else:
      raise ValueError('RL algorithm must be off-policy or on-policy.')

  @utils_timer
  def eval(self, episodes=10):
    """
      评估模型性能
    """
    self.is_train = False
    self.env.eval()
    for episode in range(episodes):
      state, _ = self.env.reset()
      done = False
      episode_reward = 0
      while not done:
        action = self.rl.take_action(state)
        state, reward, done, _ = self.env.step(action)
        episode_reward += reward
      self.eval_history['episode_rewards'].append(episode_reward)

#---------------------- 获取 action 的方法 -------------------------
#                         2025/12/4

def RTools_epsilon(epsilon, n_actions:int, argmax_action:int):
  '''
    采用 ε-greedy 的方式，epsilon越大，越随机，反之，越容易选择argmax_action。
      params:
        epsilon: float - 
        n_actions - 动作个数
        argmax_action: int - 较优action
  '''
  if np.random.random() < epsilon:
    action = np.random.randint(n_actions)
  else:
    action = argmax_action
  return action

class QNet(torch.nn.Module):
  '''
    面对连续state或者连续action，使用深度神经网络进行储存、学习。
    input - state
    output - action
  '''
  def __init__(self, state_dim, hidden_dim, action_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

  def forward(self, state):
    state = F.relu(self.fc1(state))
    return self.fc2(state)

class VAnet(torch.nn.Module):
  '''
    dueling DQN 用
  '''
  def __init__(self, state_dim, hidden_dim, action_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
    self.fc_V = torch.nn.Linear(hidden_dim, 1)

  def forward(self, X):
    A = self.fc_A(F.relu(self.fc1(X)))
    V = self.fc_V(F.relu(self.fc1(X)))
    Q = V+A-A.mean(1).view(-1, 1)
    return Q

class PolicyNet(torch.nn.Module):
  '''
    REINFORCE 用
  '''
  def __init__(self, state_dim, hidden_dim, action_dim):
    super(PolicyNet, self).__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

  def forward(self, X):
    X = F.relu(self.fc1(X))
    return F.softmax(self.fc2(X), dim=1)

class ValueNet(torch.nn.Module):
  '''
    Actor-Critic 用
  '''
  def __init__(self, state_dim, hidden_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, 1)

  def forward(self, X):
    X = F.relu(self.fc1(X))
    return self.fc2(X)

class DDPG_PolicyNet(torch.nn.Module):
  '''
    DDPG 用
  '''
  def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
    super().__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return torch.tanh(self.fc2(x))*self.action_bound

class DDPG_QValueNet(torch.nn.Module):
  def __init__(self, state_dim, hidden_dim, action_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(state_dim+action_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.fc_out = torch.nn.Linear(hidden_dim, 1)

  def forward(self, x, a):
    cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
    x = F.relu(self.fc1(cat))
    x = F.relu(self.fc2(x))
    return self.fc_out(x)

class PolicyNetContinuous(torch.nn.Module):
  '''
    SAC 用
  '''
  def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
    super().__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)  # mean
    self.fc_std = torch.nn.Linear(hidden_dim, action_dim) # std
    self.action_bound = action_bound

  def forward(self, X):
    X = F.relu(self.fc1(X))
    mu = self.fc_mu(X)
    std = F.softplus(self.fc_std(X))
    dist = Normal(mu, std)
    normal_sample = dist.rsample()  # rsample()是重参数化采样
    log_prob = dist.log_prob(normal_sample)
    action = torch.tanh(normal_sample)
    # 计算tanh_normal分布的对数概率密度
    log_prob = log_prob-torch.log(1-torch.tanh(action).pow(2)+1e-7)
    action = action*self.action_bound
    return action, log_prob

#---------------------- 探索方式 -------------------------
#                      2025/11/29

def train_off_policy(env:ENV_INFO, agent:RL_Model, num_episodes:int, replay_buffer:utils_ReplayBuffer, 
                     min_size, batch_size):
  '''
    可以 take action 的同时 update
  '''
  rewards_list = []
  with tqdm(total=int(num_episodes), desc=agent.name+' Iteration') as pbar:
    for i_episode in range(num_episodes):
      episode_return = 0
      state, _ = env.reset()
      done = False
      while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add_sample(Sample(state, action, reward, next_state, done))
        state = next_state
        episode_return += reward
        if len(replay_buffer) > min_size:
          transition_dict = replay_buffer.sample_sample(batch_size)
          # transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
          agent.update(transition_dict)
      rewards_list.append(episode_return)
      pbar.update(1)
  return rewards_list

def train_on_policy(env:ENV_INFO, agent:RL_Model, num_episodes:int):
  '''
    只能 take action 完成后，进行update
  '''
  return_list = []
  with tqdm(total=num_episodes, desc=agent.name+'Iteration') as pbar:
    for i_episode in range(num_episodes):
      episode_return = 0
      transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
      state, _ = env.reset()
      done = False
      while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        state = next_state
        episode_return += reward
      return_list.append(episode_return)
      agent.update(transition_dict)
      pbar.update(1)
  return return_list

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--'