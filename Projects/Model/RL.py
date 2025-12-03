#                           强化学习模型
#                           2025/10/25
#                            shamrock

from ..Utils.RL_config import ENV_INFO, MDP, RL_Model
import copy

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
  
  def run(self):
    cnt = 0
    while 1:
      self.policy_evaluation()
      old_pi = copy.deepcopy(self.pi)
      new_pi = self.policy_improvement()
      if old_pi == new_pi:
        cnt += 1
        if cnt > 5:
          break
      else:
        cnt = 0

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
    
#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
