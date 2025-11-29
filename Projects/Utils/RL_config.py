#                           RL相关的配置类
#                           2025/11/29
#                            shamrock

from pydantic import BaseModel
from typing import List, Optional
import numpy as np

class ENV_INFO(BaseModel):
  '''
    配置环境信息
  '''
  states_num: int = 10
  actions_num: int = 3
  end_states: List[int] = []

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
