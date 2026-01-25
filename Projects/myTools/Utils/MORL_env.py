#                         MORL的环境函数
#                           2026/1/22
#                            shamrock

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from myTools.Utils.MORL_config import *

#---------------------- deep-sea-treasure-v0 -------------------------
#                              2026/1/25

class Env_DeepSeaTreasure(MORL_EnvConfig):
  def __init__(self):
    super().__init__()
    self.env = mo_gym.make('deep-sea-treasure-v0')

    self._states_size = self.env.observation_space.shape[0] # 状态维度 (通常是2D位置)
    self._actions_size = self.env.action_space.n            # 动作数量 (离散动作空间)
    self._rewards_size = 2                                  # 多目标奖励维度 (时间惩罚和宝藏价值)
    
    self.name = 'DeepSeaTreasure'
    self.train_model = True

    self.info = {
      'env_type': 'Deep Sea Treasure',
      'objectives': ['time_penalty', 'treasure_value'],
      'grid_size': (11, 10)  # 典型的深海宝藏网格尺寸
    }

    self.train()

  def _create_env(self):
    if hasattr(self, 'env') and self.env is not None:
      self.env.close()
    if self.train_model:
      render_mode = 'rgb_array'
    else:
      render_mode = 'human'
    self.env = mo_gym.make('deep-sea-treasure-v0', render_mode=render_mode)

  def train(self):
    self.train_model = True
    self._create_env()
  
  def eval(self):
    self.train_model = False
    self._create_env()
  
  def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
    '''
      重置环境到初始状态
      返回 (initial_state, info)
    '''
    if seed is not None:
      self.env.reset(seed=seed)
    
    obs, info = self.env.reset()
    self._state = obs
    self._done = False
    
    # 合并环境info和自定义info
    combined_info = {**info, **self._info}
    return self._state, combined_info
  
  def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
    '''
      执行一步动作
      返回 (next_state, vector_reward, terminated, truncated, info)
    '''
    next_state, vector_reward, terminated, truncated, info = self.env.step(action)
    
    self._state = next_state
    self._done = terminated or truncated
    
    # 确保奖励是numpy数组格式
    if not isinstance(vector_reward, np.ndarray):
      vector_reward = np.array(vector_reward)
    
    return next_state, vector_reward, terminated, truncated, info
  
  def render(self) -> None:
    '''
      渲染一帧动画，只在 eval 模式下显示
    '''
    try:
      self.env.render()
    except Exception as e:
      print(f"渲染失败: {e}")

  def get_current_state(self):
    """获取当前状态"""
    return self._state.copy() if self._state is not None else None
  
  def is_done(self):
    """检查环境是否终止"""
    return self._done
  
  def close(self):
    """关闭环境"""
    self.env.close()

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
