#                           强化学习模型
#                           2025/10/25
#                            shamrock

from ..Utils.RL_config import ENV_INFO, RL_Model
from ..Utils.RL_tools import RTools_epsilon
import copy
import numpy as np

#---------------------- Dyanemic Programming -------------------------
#                           2025/12/1

class DP_PolicyIteration(RL_Model):
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

  def run(self, episodes=5):
    while episodes:
      self.value_iteration()
      old_pi = copy.deepcopy(self.pi)
      new_pi = self.get_policy()
      episodes -= 1
    
class SARSA(RL_Model):
  def __init__(self, env:ENV_INFO, epsilon, alpha, gamma):
    super().__init__()
    self.env = env
    self.matrix = env.matrix
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

  def run(self, episodes=50):
    for episode in range(episodes):
      state, _ = self.env.reset()
      action = self.take_action(state)
      done = False
      while not done:
        n_state, reward, done, _ = self.env.step(action)
        n_action = self.take_action(n_state)
        self.update(state, action, reward, n_state, n_action)
        state, action = n_state, n_action
    self.get_policy()

class nstep_SARSA(RL_Model):
  def __init__(self, env:ENV_INFO, n_steps, epsilon, alpha, gamma):
    super().__init__()
    self.env = env
    self.matrix = env.matrix
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

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
