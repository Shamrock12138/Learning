#                           RL相关的配置类
#                           2025/11/29
#                            shamrock

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class ENV_INFO(ABC):
  '''
    为各类 RL 环境提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `reset` `step` 方法 以及 提供相应信息（`self._state` 等）；
      - 提供 _states_num、_actions_num、matrix
  '''
  def __init__(self):
    self._state = None    # 当前状态
    self._done = None     # 是否终止
    self._info = {}       # 附加信息
    self._states_num = None   # 状态数
    self._actions_num = None  # 动作数
    self.matrix = None    # 例如MDP提供的矩阵
  
  @abstractmethod
  def train(self):
    '''
      切换训练模式
    '''
    pass

  @abstractmethod
  def eval(self):
    '''
      切换测试模式
    '''
    pass

  @abstractmethod
  def reset(self, seed, options):
    '''
      重置环境到初始状态
      返回 (initial_state, info)
    '''
    pass

  @abstractmethod
  def step(self, action) -> Tuple[int, float, bool, dict]:
    '''
      执行一步动作
      返回 (next_state, reward, done, info)
    '''
    pass

  @abstractmethod
  def render(self) -> None:
    '''
      渲染一帧动画
    '''
    pass

class RL_Model:
  '''
    为各类 RL 算法提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `run` 方法，作为模型的核心执行入口
      - 子类建议实现 show_history 方法，作为训练过程的展示
      - 子类建议实现 render 方法，作为测试过程的展示
      - 子类建议实现 save_model, load_model 方法，作为保存加载模型
  '''
  def __call__(self, *input, **kwds):
    return self.run(*input, **kwds)
  
  def show_history(self, history):
    '''
      绘制 history 的变化表
    '''
    pass

  def render(self):
    '''
      渲染一趟的动画
    '''
    pass
  
  def save_model(self, dir_path:str, name:str):
    '''
      保存在 dir_path/name 路径中
    '''
    pass

  def load_model(self, dir_path:str, name:str):
    '''
      加载 dir_path/name 路径中的模型
    '''
    pass

  @abstractmethod
  def run(self, *input, **kwds):
    '''
      模型程序入口
    '''
    pass

class MDP:
  '''
    配置 马尔可夫 过程
  '''
  States: List[int] = []          # 状态索引列表，如 [0, 1, 2, 3, 4]
  Actions: List[int] = []         # 动作索引列表，如 [0, 1, 2]
  P: List[List[List[float]]] = [] # P[s][a][s_next] = Pr(s_next | s, a)
  R_E: List[float] = []               # R[s] = r
  R_SA: List[List[float]] = []        # R[s][a] = r
  R_SAE: List[List[List[float]]] = [] # R[s][a][s_next] = r
  done: List[bool] = []      # done[s] = False

  def __init__(self, states_num, actions_num):
    self.States = list(range(states_num))  # [0, 1, ..., n-1]
    self.Actions = list(range(actions_num))  # [0, 1, ..., m-1]

  def test(self):
    '''
      测试 MDP 是否有效
    '''
    if not self.States:
      raise ValueError('State matrix States must be provided explicitly.')
    if not self.Actions:
      raise ValueError('Action matrix Actions must be provided explicitly.')
    if not self.P:
      raise ValueError('Transition matrix P must be provided explicitly.')
    if not self.R_E and not self.R_SA and not self.R_SAE:
      raise ValueError("Reward matrix R must be provided explicitly.")
    if not self.done:
      raise ValueError("Done matrix done must be provided explicitly.")

if __name__ == '__main__':
  pass
  # env = ENV_INFO()
  # env._states_num
  # mdp = MDP(
  #   info=ENV_INFO(),
  # )
  # mdp.P = [[[1,0,0], [0,1,0]], [[0,1,0], [0,0,1]], [[0,0,1], [0,0,1]]]
  # States=[0,1,2],
  #   Actions=[0,1],
  #   P=[[[1,0,0], [0,1,0]], [[0,1,0], [0,0,1]], [[0,0,1], [0,0,1]]],  # 必须传
  #   R=[[0,0], [0,0], [1,1]]

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
