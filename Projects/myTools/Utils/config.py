#                           各种配置类
#                           2025/10/24
#                            shamrock

from abc import ABC, abstractmethod
from dataclasses import dataclass

class Trajectory:
  '''
    用来记录一条完整轨迹（一趟 episode 的轨迹）
  '''
  def __init__(self, init_state):
    self.states = [init_state]
    self.actions = []
    self.rewards = []
    self.dones = []
    self.length = 0

  def __len__(self):
    return self.length

  def store_step(self, state, action, reward, done):
    '''
      存储一条轨迹(state, action, reward, done)
    '''
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.dones.append(done)
    self.length += 1



#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 