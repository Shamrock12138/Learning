#                  Reinforcement Learning 工具函数
#                           2025/11/30
#                            shamrock
import numpy as np
from .RL_config import ENV_INFO
import torch, random
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

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

def train_off_policy(env:ENV_INFO, agent, num_episodes:int, replay_buffer, 
                     min_size, batch_size):
  return_list = []
  with tqdm(total=int(num_episodes), desc=agent.name+' Iteration') as pbar:
    for i_episode in range(num_episodes):
      episode_return = 0
      state, _ = env.reset()
      done = False
      while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        if replay_buffer.size() > min_size:
          transition_dict = replay_buffer.sample(batch_size)[0]
          # transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
          agent.update(transition_dict)
      return_list.append(episode_return)
      pbar.update(1)
  return return_list

def train_on_policy(env:ENV_INFO, agent, num_episodes:int):
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

def RTools_Sample(MDP, INFO, pi, timestep_max, number):
  '''
    对INFO环境下的MDP进行采样，返回采样得到的所有“路径”[s, a, r, s_next]
      params:
        MDP - 描述环境马尔可夫决策过程 提供(S, A, P, R, gamma)
          S: list [states_num] 所有状态
          A: list [actions_num] 所有动作  
          P: list [states_num, actions_num, states_num] s采取了a动作后的next_s的所有概率
          R: list [states_num, actions_num] s采取了a动作后的奖励
        INFO - 提供环境信息 (end: list, states_num: int, actions_num: int)
        pi: dict - 策略
        timestep_max - 单条轨迹的最大时间步数
        number - 要生成的 episode 总数
      return:
        episodes - number 次 episode 的探索列表 [[s, a, r, s_next], ...]
  '''
  S, A, P, R, gamma = MDP
  end, states_num, actions_num = INFO
  episodes = []
  for _ in range(number):
    episode, timestep = [], 0
    s = S[np.random.randint(states_num)]
    while s not in end and timestep <= timestep_max:
      timestep += 1
      rand, temp = np.random.rand(), 0
      for a_opt in A:
        # temp += pi.get(s+' '+a_opt, 0)
        temp += pi[s][a_opt]
        if temp > rand:
          a = a_opt
          # r = R.get(s+' '+a, 0)
          r = R[s][a]
          break
      rand, temp = np.random.rand(), 0
      for s_opt in S:
        # temp += P.get(join(join(s, a), s_opt), 0)
        temp += P[s][a][s_opt]
        if temp > rand:
          s_next = s_opt
          break
      episode.append((s, a, r, s_next)) 
      s = s_next
    episodes.append(episode)
  return episodes

#---------------------- 评价指标 -------------------------
#                      2025/11/29

def RTools_OccupancyMeasure(episodes, s, a, timestep_max, gamma):
  '''
    计算 (s, a) 的占用度量，即：(s, a) 在环境交互过程的频率
      params:
        episodes - 轨迹列表，格式[(s₀, a₀, r₀, s₁), (s₁, a₁, r₁, s₂), ...]
        s, a - 要统计的状态和动作
        timestep_max - 最大步长
        gamma - 折扣因子
      return:
        rho - (s, a) 在环境交互过程的频率
  '''
  rho = 0
  total_times = np.zeros(timestep_max)  # total_times[i] - 总步长>=i的次数
  occur_times = np.zeros(timestep_max)  # occur_times[i] - 第i步出现(s, a)的次数
  for episode in episodes:
    for i in range(len(episode)):
      (s_opt, a_opt, r, s_next) = episode[i]
      total_times[i] += 1
      if s == s_opt and a == a_opt:
        occur_times[i] += 1
  for i in reversed(range(timestep_max)):
    if total_times[i]:
      rho += gamma**i*occur_times[i]/total_times[i]
  return (1-gamma)*rho

#---------------------- 获取 value 的方法 -------------------------
#                         2025/11/30

def RTools_MonteCorlo(episodes, gamma, first_visit=True):
  '''
    用 Monte Corlo 的方式，计算 episodes 中每个 state 的 V
      params:
        episodes - 探索列表 [[s, a, r, s_next], ...]
        gamma - 折扣因子
        first_visit - 是否使用首次访问 MC（未实现）
      return:
        V: dir - {s1: v1, ...} S 状态的 V
  '''
  V, N = {}, {}
  for episode in episodes:
    G = 0.0
    for i in range(len(episode)-1, -1, -1):
      (s, a, r, next_s) = episode[i]
      G = r+gamma*G
      if s not in V:
        V[s] = 0.0
        N[s] = 0
      N[s] += 1
      V[s] += (G-V[s])/N[s]
  return V

#---------------------- 存储经验 -------------------------
#                       2025/12/9
import collections

class ReplayBuffer:
  '''
    经验回放池
  '''
  def __init__(self, capacity=1000):
    self.buffer = collections.deque(maxlen=capacity)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size=None):
    '''
      从 buffer 中采样数据,数量为 batch_size ，如果 batch_size == None，则全部取出 
      return:  
        transitions_dict - {
        'states': (state1, state2 ...),
        'actions': ...,
        'next_states': ...,
        'rewards': ...,
        'dones': ...
        }
      example:  
        state, action, reward, next_state, done = zip(*transitions)
        np.array(state), action, reward, np.array(next_state), done
    '''
    if batch_size is None:
      batch_size = self.size()
    transitions = random.sample(self.buffer, batch_size)
    state, action, reward, next_state, done = zip(*transitions)
    transitions_dict = {
      'states': np.array(state),
      'actions': action,
      'next_states': np.array(next_state),
      'rewards': reward,
      'dones': done
    }
    return transitions_dict, np.array(state), action, reward, np.array(next_state), done

  def size(self):
    return len(self.buffer)

class HER:
  '''
    Hindsight Experience Replay - 事后经验回放
    论文：
      "Hindsight Experience Replay" - arXiv:1707.01495（2017）
  '''
  def __init__(self):
    pass

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
