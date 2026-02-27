#                         MARL相关的配置类
#                           2026/2/14
#                            shamrock

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple, Any, Dict

from myTools.Utils.tools import *

class MARL_EnvConfig(ABC):
  '''
    为各类 MARL 环境提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `reset` `step` `render`方法；
      - 提供 _states_num、_actions_num、name
  '''
  def __init__(self, n_agents=0, states_dim=0, actions_dim=0):
    self._n_agents: int = n_agents          # 智能体数量
    self._states_dim: int = states_dim      # 单个智能体状态维度
    self._action_dim: int = actions_dim     # 单个智能体动作维度
    self._current_state: np.ndarray = None  # 形状: (n_agents, state_dim)
    self._done: bool = False      # 全局终止标志
    self.name = None

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

#---------------------- 无人机MEC基类 -------------------------
#                         2026/2/18

class BaseStation:
  '''
    基站基类
  '''
  def __init__(self, position: Tuple[int, int]):
    self.position = np.array(position, dtype=np.float32)

class UAV:
  '''
    无人机基类
  '''
  def __init__(self, uav_id:int, position:np.ndarray, battery:float, uav_type:int) -> None:
    self.id = uav_id
    self.type = uav_type
    self.battery = float(battery)
    self.position = np.array(position, dtype=np.float32)
    self.state = {
      "alive": True,
      "charging": False,
      "target": None,  # 当前目标位置或目标ID
      "steps_since_charge": 0
    }

  def move(self, action: int, grid_size: Tuple[int, int]) -> Tuple[float, bool]:
    '''
      执行移动动作，返回移动距离和是否越界
    '''
    old_pos = self.position.copy()
    is_out_of_bounds = False
    
    # 先计算目标位置
    target_pos = self.position.copy()
    if action == 1:  # 上
        target_pos[1] = self.position[1] + 1
    elif action == 2:  # 下
        target_pos[1] = self.position[1] - 1
    elif action == 3:  # 左
        target_pos[0] = self.position[0] - 1
    elif action == 4:  # 右
        target_pos[0] = self.position[0] + 1
    # action == 0: 静止，不移动

    # ===== 边界检查（关键修改）=====
    # 检测是否尝试越界（即使最终被限制，也视为越界尝试）
    if (target_pos[0] < 0 or target_pos[0] >= grid_size[0] or 
        target_pos[1] < 0 or target_pos[1] >= grid_size[1]):
        is_out_of_bounds = True
    
    # 无论是否越界，都限制在边界内
    self.position[0] = np.clip(target_pos[0], 0, grid_size[0] - 1)
    self.position[1] = np.clip(target_pos[1], 0, grid_size[1] - 1)
    
    # ===== 移动距离计算 =====
    distance = np.linalg.norm(self.position - old_pos)
    
    return distance, is_out_of_bounds
  
  def consume_battery(self, amount:float) -> bool:
    '''
      消耗 amount 电量，返回是否耗尽电量
    '''
    self.battery = max(0.0, self.battery - amount)
    self.state["alive"] = self.battery > 0
    return self.state["alive"]
  
  def charge_battery(self, amount:float) -> float:
    """
      充电 amount ，返回实际充入电量
    """
    actual_charge = min(amount, 100.0-self.battery)
    self.battery += actual_charge
    return actual_charge
  
  def reset(self, position:np.ndarray, battery:float):
    '''
      重置无人机
    '''
    self.position = np.array(position, dtype=np.float32)
    self.battery = float(battery)
    self.state = {
      "alive": True,
      "charging": False,
      "target": None,
      "last_action": None,
      "steps_since_charge": 0
    }

  def get_info(self) -> Dict[str, Any]:
    return {
      "id": self.id,
      "type": self.type,
      "position": tuple(self.position),
      "battery": self.battery,
      "alive": self.state["alive"],
    }

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
