#                         MORL相关的配置类
#                           2026/1/22
#                            shamrock

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class MORL_EnvConfig(ABC):
  '''
    为各类 MORL 环境提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `reset` `step` `render` `train` `eval`方法；
      - 提供 _states_num、_actions_num、matrix、name
  '''
  def __init__(self):
    self._state = None    # 当前状态
    self._done = None     # 是否终止
    self._states_size = None    # 状态尺寸（几维state）
    self._actions_size = None   # 动作尺寸（几维action）
    self._rewards_size = None   # 奖励尺寸（几维reward）

    self._info = {}       # 附加信息
  
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

class MORL_ModelConfig(ABC):
  '''
    为各类 MORL 算法提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `train` `take_action`方法，作为模型的训练入口
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

  @abstractmethod
  def train(self, *input, **kwds):
    '''
      开始训练
    '''
    pass

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

