#                         MARL相关的配置类
#                           2026/2/14
#                            shamrock

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple, Any, Dict

class MARL_EnvConfig(ABC):
  '''
    为各类 MARL 环境提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `reset` `step` `render` `train` `eval`方法；
      - 提供 _states_num、_actions_num、matrix、name
  '''
  def __init__(self, n_agents=0, states_dim=0, actions_dim=0):
    self._n_agents: int = n_agents          # 智能体数量
    self._states_dim: int = states_dim      # 单个智能体状态维度
    self._action_dim: int = actions_dim     # 单个智能体动作维度
    self._current_state: np.ndarray = None  # 形状: (n_agents, state_dim)
    self._done: bool = False      # 全局终止标志

    self._info = {}       # 附加信息

  @abstractmethod
  def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
    '''
      重置环境到初始状态，返回 (initial_state, info)
        returns:
          initial_state - np.ndarray, shape=(n_agents, state_dim)
    '''
    pass

  @abstractmethod
  def step(self, action: np.ndarray) -> Tuple[
    np.ndarray,  # next_states: (n_agents, state_dim)
    np.ndarray,  # rewards:     (n_agents,)
    np.ndarray,  # dones:       (n_agents,)
    Dict[str, Any]  # info: 附加信息
  ]:
    '''
      所有的 agents 执行一步动作，返回 (next_state, reward, done, info)
        returns:
          next_states - (n_agents, state_dim)
          rewards - (n_agents,)
          dones - (n_agents,)
    '''
    pass

  @abstractmethod
  def render(self) -> None:
    '''
      渲染一帧动画
    '''
    pass

class MARL_ModelConfig(ABC):
  '''
    为各类 MARL 算法提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `take_action`方法，作为模型的训练入口
      - 子类建议实现 show_history 方法，作为训练过程的展示
      - 子类建议实现 render 方法，作为测试过程的展示
      - 子类建议实现 save_model, load_model 方法，作为保存加载模型
  '''
  def __init__(self):
    super().__init__()

  def __call__(self, *input, **kwds):
    return self.train(*input, **kwds)
  
  @abstractmethod
  def take_action(self, state):
    '''
      获取state的下一动作
    '''
    pass
  
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

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
