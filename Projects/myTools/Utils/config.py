#                           全局配置类
#                           2025/10/24
#                            shamrock

import numpy as np
from abc import ABC, abstractmethod
from typing import TypedDict

class Sample:
  '''
    用来记录一次采样
  '''
  __slots__ = ('state', 'action', 'reward', 'next_state', 'done')

  def __init__(self, 
               state:np.ndarray, action:np.ndarray,
               reward:np.ndarray, next_state:np.ndarray,
               done:np.ndarray):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done

  def __repr__(self):
    return (f"Sample(s{self.state.shape}, a{self.action.shape}, "
            f"r={self.reward.item():.2f}, d={self.done.item()})")

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

class SampleBatch(TypedDict):
  """经验回放缓冲区采样批次的类型定义"""
  states: np.ndarray      # shape: (batch_size, state_dim)
  actions: np.ndarray     # shape: (batch_size,) 离散动作 或 (batch_size, action_dim) 连续动作
  next_states: np.ndarray # shape: (batch_size, state_dim)
  rewards: np.ndarray     # shape: (batch_size,)
  dones: np.ndarray       # shape: (batch_size,)

class RL_Trainer:
  def __init__(self) -> None:
    pass

  @abstractmethod
  def train(self, episodes_num) -> dict:
    '''
      开始训练，返回训练过程的历史记录
        returns:
          history: dict - {'loss': [...], 'rewards': [...]}
    '''
    pass

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 