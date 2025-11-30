#                           RL相关的配置类
#                           2025/11/29
#                            shamrock

from pydantic import BaseModel
from typing import List, Optional, Tuple
import numpy as np

class ENV_INFO:
  '''
    配置环境信息，提供向外接口
  '''
  def __init__(self):
    self._state = None    # 当前状态
    self._done = None     # 是否终止
    self._info = {}       # 附加信息
    self._states_num = None   # 状态数
    self._actions_num = None  # 动作数
  
  def reset(self, seed, options):
    '''
      重置环境到初始状态
      返回 (initial_state, info)
    '''
    pass

  def step(self, action) -> Tuple[int, float, bool, dict]:
    '''
      执行一步动作
      返回 (next_state, reward, info)
    '''
    pass

class MDP(BaseModel):
  '''
    配置马尔可夫过程
  '''
  States: List[int] = []          # 状态索引列表，如 [0, 1, 2, 3, 4]
  Actions: List[int] = []         # 动作索引列表，如 [0, 1, 2]
  P: List[List[List[float]]] = [] # P[s][a][s_next] = Pr(s_next | s, a)
  R: List[List[float]] = []       # R[s][a] = r(s, a)
  gamma: float = 0.95

  def __init__(self, info: ENV_INFO, **data):
    super().__init__(info=info, **data)
    if not self.States:
      self.States = list(range(info.states_num))  # [0, 1, ..., n-1]
    if not self.Actions:
      self.Actions = list(range(info.actions_num))  # [0, 1, ..., m-1]
    if not self.P:
      raise ValueError('Transition matrix P must be provided explicitly.')
    if not self.R:
      raise ValueError("Reward matrix R must be provided explicitly.")

if __name__ == '__main__':
  mdp = MDP(
    info=ENV_INFO(states_num=3, actions_num=2),
    States=[0,1,2],
    Actions=[0,1],
    P=[[[1,0,0], [0,1,0]], [[0,1,0], [0,0,1]], [[0,0,1], [0,0,1]]],  # 必须传
    R=[[0,0], [0,0], [1,1]]
  )

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
