#                           强化学习模型
#                           2025/10/25
#                            shamrock

from ..Utils.RL_config import ENV_INFO, RL_Model
from ..Utils.RL_tools import RTools_epsilon, Qnet, ReplayBuffer
from ..Utils.tools import utils_timer, utils_autoAssign, utils_showHistory
import copy, random, torch, time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model-Based: DP-P DP-V Dyna-Q
# Model-Free: SARSA Q-Learning DQN

#---------------------- Dyanemic Programming -------------------------
#                           2025/12/1

class DP_PolicyIteration(RL_Model):
  '''
    DP：需要环境数据（MDP）
  '''
  def __init__(self, env:ENV_INFO, theta, gamma):
    self.env = env
    self.mdp = env.matrix
    self.v = [0]*env._states_num
    self.pi = [[0]*env._actions_num]*env._states_num

    self.theta = theta
    self.gamma = gamma

  def policy_evaluation(self):
    cnt = 1
    while 1:
      max_diff = 0
      new_v = [0]*self.env._states_num
      for s in range(self.env._states_num):
        Q = []
        for a in range(self.env._actions_num):
          q = 0
          for next_s in range(self.env._states_num):
            p = self.mdp.P[s][a][next_s]
            r = self.mdp.R_E[next_s]
            done = self.mdp.done[next_s]
            q += p*(r+self.gamma*self.v[next_s]*(1-done))
          Q.append(self.pi[s][a]*q)
        new_v[s] = sum(Q)
        max_diff = max(max_diff, abs(new_v[s]-self.v[s]))
      self.v = new_v
      if max_diff < self.theta:
        break
      cnt += 1
    print(f'{cnt}轮后完成 policy_evaluation ')
  
  def policy_improvement(self):
    for s in range(self.env._states_num):
      Q = []
      for a in range(self.env._actions_num):
        q = 0
        for next_s in range(self.env._states_num):
          p = self.mdp.P[s][a][next_s]
          r = self.mdp.R_E[next_s]
          done = self.mdp.done[next_s]
          q += p*(r+self.gamma*self.v[next_s]*(1-done))
        Q.append(q)
      maxQ = max(Q)
      cntQ = Q.count(maxQ)
      self.pi[s] = [1/cntQ if q == maxQ else 0 for q in Q]
    return self.pi
  
  @utils_timer
  def run(self, episodes=5):
    while episodes:
      self.policy_evaluation()
      old_pi = copy.deepcopy(self.pi)
      new_pi = self.policy_improvement()
      episodes -= 1

class DP_ValueIteration(RL_Model):
  def __init__(self, env:ENV_INFO, theta, gamma):
    self.env = env
    self.mdp = env.matrix
    self.v = [0]*env._states_num
    self.pi = [[0]*env._actions_num]*env._states_num
    self.theta = theta
    self.gamma = gamma

  def value_iteration(self):
    cnt = 1
    while 1:
      max_diff = 0
      new_v = [0]*self.env._states_num
      for s in range(self.env._states_num):
        Q = []
        for a in range(self.env._actions_num):
          q = 0
          for next_s in range(self.env._states_num):
            p = self.mdp.P[s][a][next_s]
            r = self.mdp.R_E[next_s]
            done = self.mdp.done[next_s]
            q += p*(r+self.gamma*self.v[next_s]*(1-done))
          Q.append(q)
        new_v[s] = max(Q)
        max_diff = max(max_diff, abs(new_v[s]-self.v[s]))
      self.v = new_v
      if max_diff < self.theta:
        break
      cnt += 1
    print(f'{cnt}轮后完成 value_iteration ')
    # self.get_policy()

  def get_policy(self):
    for s in range(self.env._states_num):
      Q = []
      for a in range(self.env._actions_num):
        q = 0
        for next_s in range(self.env._states_num):
          p = self.mdp.P[s][a][next_s]
          r = self.mdp.R_E[next_s]
          done = self.mdp.done[next_s]
          q += p*(r+self.gamma*self.v[next_s]*(1-done))
        Q.append(q)
      maxQ = max(Q)
      cntQ = Q.count(maxQ)
      self.pi[s] = [1/cntQ if q == maxQ else 0 for q in Q]

  @utils_timer
  def run(self, episodes=5):
    while episodes:
      self.value_iteration()
      old_pi = copy.deepcopy(self.pi)
      new_pi = self.get_policy()
      episodes -= 1
    
#---------------------- SARSA -------------------------
#                      2025/12/8

class SARSA(RL_Model):
  '''
    SARSA：不需要环境数据（MDP）
  '''
  def __init__(self, env:ENV_INFO, epsilon, alpha, gamma):
    super().__init__()
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon

    self.Q = np.zeros((env._states_num, env._actions_num))
    self.pi = np.zeros([env._states_num, env._actions_num])

  def take_action(self, state):
    argmax_action = np.argmax(self.Q[state])
    return RTools_epsilon(self.epsilon, self.env._actions_num, argmax_action)
  
  def update(self, s0, a0, r, s1, a1):
    td_error = r+self.gamma*self.Q[s1, a1]-self.Q[s0, a0]
    self.Q[s0, a0] += self.alpha*td_error

  def get_policy(self):
    for state in range(self.env._states_num):
      maxQ = np.max(self.Q[state])
      for i in range(self.env._actions_num):
        if self.Q[state, i] == maxQ:
          self.pi[state, i] = 1

  @utils_timer
  def run(self, episodes=50):
    for episode in range(episodes):
      state, _ = self.env.reset()
      action = self.take_action(state)
      done = False
      while not done:
        n_state, reward, done, _ = self.env.step(action)
        # if hasattr(self.env, 'render'):
        #   self.env.render()
        n_action = self.take_action(n_state)
        self.update(state, action, reward, n_state, n_action)
        state, action = n_state, n_action
    self.get_policy()

class SARSA_nstep(RL_Model):
  '''
    nSteps SARSA
  '''
  def __init__(self, env:ENV_INFO, n_steps, epsilon, alpha, gamma):
    super().__init__()
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_steps = n_steps

    self.Q = np.zeros((env._states_num, env._actions_num))
    self.pi = np.zeros([env._states_num, env._actions_num])
    self.state_list = []
    self.action_list = []
    self.reward_list = []

  def take_action(self, state):
    argmax_action = np.argmax(self.Q[state])
    return RTools_epsilon(self.epsilon, self.env._actions_num, argmax_action)
  
  def update(self, s0, a0, r, s1, a1, done):
    self.state_list.append(s0)
    self.action_list.append(a0)
    self.reward_list.append(r)
    if len(self.state_list) == self.n_steps:
      G = self.Q[s1, a1]
      for i in reversed(range(self.n_steps)):
        G = G*self.gamma+self.reward_list[i]
        if done and i > 0:
          s = self.state_list[i]
          a = self.action_list[i]
          self.Q[s, a] += self.alpha*(G-self.Q[s, a])
      s = self.state_list.pop(0)
      a = self.action_list.pop(0)
      self.reward_list.pop(0)
      self.Q[s, a] += self.alpha*(G-self.Q[s, a])
    if done:
      self.state_list = []
      self.action_list = []
      self.reward_list = []

  def get_policy(self):
    for state in range(self.env._states_num):
      maxQ = np.max(self.Q[state])
      for i in range(self.env._actions_num):
        if self.Q[state, i] == maxQ:
          self.pi[state, i] = 1

  @utils_timer
  def run(self, episodes=50):
    for episode in range(episodes):
      state, _ = self.env.reset()
      action = self.take_action(state)
      done = False
      while not done:
        n_state, reward, done, _ = self.env.step(action)
        n_action = self.take_action(n_state)
        self.update(state, action, reward, n_state, n_action, done)
        state, action = n_state, n_action
    self.get_policy()

#---------------------- Q Learning -------------------------
#                        2025/12/8

class Q_Learning(RL_Model):
  def __init__(self, env:ENV_INFO, epsilon, alpha, gamma):
    super().__init__()
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon

    self.Q = np.zeros((env._states_num, env._actions_num))
    self.pi = np.zeros((env._states_num, env._actions_num))

  def take_action(self, state):
    argmax_action = np.argmax(self.Q[state])
    return RTools_epsilon(self.epsilon, self.env._actions_num, argmax_action)
  
  def update(self, s0, a0, r, s1):
    td_error = r+self.gamma*self.Q[s1].max()-self.Q[s0, a0]
    self.Q[s0, a0] += self.alpha*td_error

  def get_policy(self):
    for state in range(self.env._states_num):
      maxQ = np.max(self.Q[state])
      for i in range(self.env._actions_num):
        if self.Q[state, i] == maxQ:
          self.pi[state, i] = 1
    return self.pi

  @utils_timer
  def run(self, episodes=None, diff_tol=1e-6, quit_cnt=5):
    '''
      params:
        episodes - 
          None时，提前停止，使用 diff_tol quit_cnt 参数
          int时，固定训练 episodes 轮数，不使用 diff_tol quit_cnt 参数
        diff_tol: float - 当 差异 大于 diff_tol 时，退出计数+1
        quit_cnt: int - 退出计数，当退出计数达到 quit_cnt 时，提前停止
    '''
    if episodes is not None:
      for episode in range(episodes):
        state, _ = self.env.reset()
        done = False
        while not done:
          action = self.take_action(state)
          n_state, reward, done, _ = self.env.step(action)
          self.update(state, action, reward, n_state)
          state = n_state
    else:
      times, cnt = 0, 0
      while True:
        times += 1
        last_Q = self.Q.copy()
        state, _ = self.env.reset()
        done = False
        while not done:
          action = self.take_action(state)
          n_state, reward, done, _ = self.env.step(action)
          self.update(state, action, reward, n_state)
          state = n_state
        current_Q = self.Q.copy()
        Q_diff = np.abs(current_Q-last_Q).sum()
        # print(f'Q Δ = {Q_diff:.6f}')
        if Q_diff < diff_tol:
          cnt += 1
          if cnt > quit_cnt:
            print(f'Finished after {times} times.')
            break
        else:
          cnt = 0
    self.get_policy()

#---------------------- Dyna_Q -------------------------
#                      2025/12/8

class Dyna_Q(RL_Model):
  def __init__(self, env:ENV_INFO, epsilon, alpha, gamma, n_planning):
    super().__init__()
    utils_autoAssign(self)
    self.model = dict()     # 脑内模拟库

    self.Q = np.zeros((env._states_num, env._actions_num))
    self.pi = np.zeros((env._states_num, env._actions_num))

  def take_action(self, state):
    argmax_action = np.argmax(self.Q[state])
    return RTools_epsilon(self.epsilon, self.env._actions_num, argmax_action)
  
  def q_learning(self, s0, a0, r, s1):
    td_error = r+self.gamma*self.Q[s1].max()-self.Q[s0, a0]
    self.Q[s0, a0] += self.alpha*td_error

  def update(self, s0, a0, r, s1):
    self.q_learning(s0, a0, r, s1)
    self.model[(s0, a0)] = r, s1
    for _ in range(self.n_planning):
      (s, a), (r, s_) = random.choice(list(self.model.items()))
      self.q_learning(s, a, r, s_)

  def get_policy(self):
    for state in range(self.env._states_num):
      maxQ = np.max(self.Q[state])
      for i in range(self.env._actions_num):
        if self.Q[state, i] == maxQ:
          self.pi[state, i] = 1
    return self.pi

  @utils_timer
  def run(self, episodes=None, diff_tol=1e-6, quit_cnt=5):
    '''
      params:
        episodes - 
          None时，提前停止，使用 diff_tol quit_cnt 参数
          int时，固定训练 episodes 轮数，不使用 diff_tol quit_cnt 参数
        diff_tol: float - 当 差异 大于 diff_tol 时，退出计数+1
        quit_cnt: int - 退出计数，当退出计数达到 quit_cnt 时，提前停止
    '''
    if episodes is not None:
      for episode in range(episodes):
        state, _ = self.env.reset()
        done = False
        while not done:
          action = self.take_action(state)
          n_state, reward, done, _ = self.env.step(action)
          self.update(state, action, reward, n_state)
          state = n_state
    else:
      times, cnt = 0, 0
      while True:
        times += 1
        last_Q = self.Q.copy()
        state, _ = self.env.reset()
        done = False
        while not done:
          action = self.take_action(state)
          n_state, reward, done, _ = self.env.step(action)
          self.update(state, action, reward, n_state)
          state = n_state
        current_Q = self.Q.copy()
        Q_diff = np.abs(current_Q-last_Q).sum()
        # print(f'Q Δ = {Q_diff:.6f}')
        if Q_diff < diff_tol:
          cnt += 1
          if cnt > quit_cnt:
            print(f'Finished after {times} times.')
            break
        else:
          cnt = 0
    self.get_policy()

#---------------------- Deep Q Network -------------------------
#                        2025/12/8

class DQN(RL_Model):
  def __init__(self, env:ENV_INFO, state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
               target_update, device):
    '''
      params:
        state_dim, hidden_dim, action_dim - 维度
        lr, gamma - learning rate, gamma
        epsilon - epsilon-greedy
        target_update - 目标网络更新频率
        device - device
    '''
    super().__init__()
    utils_autoAssign(self)
    self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
    self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
    self.counter = 0
    self.replay_buffer = ReplayBuffer(1000)

  def take_action(self, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    argmax_action = self.q_net(state).argmax().item()
    return RTools_epsilon(self.epsilon, self.env._actions_num, argmax_action)
  
  def update(self, transition_dict):
    states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
    actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
    rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
    next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
    dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

    q_value = self.q_net(states).gather(1, actions)   # Q(s, a) -> n_s
    max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
    q_target = rewards+self.gamma*max_next_q_values*(1-dones)  # Q*(n_s, max_a)
    dqn_loss = torch.mean(F.mse_loss(q_value, q_target))

    self.optimizer.zero_grad()
    dqn_loss.backward()
    self.optimizer.step()
    if self.counter%self.target_update == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())
    self.counter += 1

  def show_history(self, returns_list):
    utils_showHistory(returns_list, 'DQN on {}'.format(self.env.name), 
                      'Episodes', 'Returns')

  def render(self, times:int=1):
    '''
      渲染 times 趟动画
    '''
    self.env.eval()
    pbar = tqdm(iterable=range(times), desc='test')
    for T in pbar:
      done = False
      state, _ = self.env.reset()
      self.env.render()
      time.sleep(0.02)
      while not done:
        action = self.take_action(state)
        state, _, done, _ = self.env.step(action)
        self.env.render()
        time.sleep(1/60)

  @utils_timer
  def run(self, episodes=None, diff_tol=1e-6, quit_cnt=5):
    '''
      params:
        episodes - 
          None时，提前停止，使用 diff_tol quit_cnt 参数
          int时，固定训练 episodes 轮数，不使用 diff_tol quit_cnt 参数
        diff_tol: float - 当 差异 大于 diff_tol 时，退出计数+1
        quit_cnt: int - 退出计数，当退出计数达到 quit_cnt 时，提前停止
    '''
    returns_list = []
    if episodes is not None:
      pbar = tqdm(iterable=range(episodes), desc='DQN Iterable')
      for _ in pbar:
        state, _ = self.env.reset()
        done = False
        episode = 0
        while not done:
          action = self.take_action(state)
          n_state, reward, done, _ = self.env.step(action)
          self.replay_buffer.add(state, action, reward, n_state, done)
          episode += reward
          if self.replay_buffer.size() > 100:
            transition_dict, _, _, _, _, _ = self.replay_buffer.sample(64)
            self.update(transition_dict)
          state = n_state
        returns_list.append(episode)
    else:
      raise ImportError('dont finish')
      times, cnt = 0, 0
      while True:
        times += 1
        last_Q = self.Q.copy()
        state, _ = self.env.reset()
        done = False
        while not done:
          action = self.take_action(state)
          n_state, reward, done, _ = self.env.step(action)
          if self.replay_buffer.size() > 50:
            transition_dict, _, _, _, _, _ = self.replay_buffer.sample(32)
            self.update(transition_dict)
          state = n_state
        current_Q = self.Q.copy()
        Q_diff = np.abs(current_Q-last_Q).sum()
        # print(f'Q Δ = {Q_diff:.6f}')
        if Q_diff < diff_tol:
          cnt += 1
          if cnt > quit_cnt:
            print(f'Finished after {times} times.')
            break
        else:
          cnt = 0
    self.show_history(returns_list)

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
