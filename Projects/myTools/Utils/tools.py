#                           各种工具函数
#                           2025/10/24
#                            shamrock

from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Any

import time, torch, inspect, os, json, random

from myTools.Utils.config import *

#---------------------- 其他 -------------------------
#                      2026/1/13

def utils_getDevice():
  '''
    返回可用设备，优先GPU
  '''
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')
  return device

def utils_setSeed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)

#---------------------- 函数 -------------------------
#                      2026/1/13

def utils_timer(func):
  '''
    计时器修饰器
  '''
  @wraps(func)
  def wrapper(*args, **kwargs):
    begin_time = time.time()
    ret = func(*args, **kwargs)
    end_time = time.time()
    print(f'Run Time: {end_time-begin_time} s')
    return ret
  return wrapper

def utils_autoAssign(self):
  '''
    自动将当前 __init__ 方法的所有参数（除 'self'）赋值为实例属性 self.xxx

    示例：
      class MyModel:
        def __init__(self, a, b, c=1):
          auto_assign(self)   # → self.a, self.b, self.c 自动创建
  '''
  frame = inspect.currentframe().f_back
  func_name = frame.f_code.co_name
  init_method = getattr(self.__class__, '__init__')
  sig = inspect.signature(init_method)
  local_vars = frame.f_locals
  for name in sig.parameters:
    if name == 'self':
      continue
    if name not in local_vars:
      continue
    setattr(self, name, local_vars[name])

def utils_setAttr(self, d:dict):
  '''
    将 dict 中的 key, value 设为 self 的参数
  '''
  for key, value in d.items():
    setattr(self, key, value)

#---------------------- 模型保存函数 -------------------------
#                        2026/1/13

def utils_showHistory(histories:list, labels, title:str, x_lable:str, y_lable:str, save_path=None):
  '''
    显示 history 的曲线图，x轴y轴为 x_lable, y_lable，标题为 title
      params:
        histories - 列表的列表
        labels - 每个历史数据序列的标签
        title, x_lable, y_lable - 字符串
        save_path - 保存路径，默认None不保存
  '''
  episodes_list = list(range(len(histories[0])))

  for i, history in enumerate(histories):
    plt.plot(episodes_list[:len(history)], history, label=labels[i])

  plt.xlabel(x_lable)
  plt.ylabel(y_lable)
  plt.title(title)
  plt.legend()        # 显示图例
  plt.grid(True)      # 增强可读性
  plt.tight_layout()  # 防止标签被裁剪

  if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

  plt.show()
  
def utils_saveModel(dir_path, name, checkpoint):
  '''
    将 checkpoint 的参数的模型保存到 dir_path+name 的.pt文件中，
    注：name后缀名为.pt
      params:
        dir_path - 一般为文件夹路径
        name - 要保存的文件名 .pt
        checkout - 模型参数，例：
          checkpoint = {
            'q_net_state': self.q_net.state_dict(),
            'target_q_net_state': self.target_q_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
          }
  '''
  path = dir_path+name
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save(checkpoint, path)
  print(f"Model saved to {path}")

def utils_loadModel(dir_path, name, device):
  '''
    加载 dir_path+name 的.pt文件中的参数的模型，返回 checkpoint
      params:
        dir_path - 一般为文件夹路径
        name - 要保存的文件名 .pt
      return:
        checkout - 模型参数，例：
          checkpoint = {
            'q_net_state': self.q_net.state_dict(),
            'target_q_net_state': self.target_q_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
          }
  '''
  path = dir_path+name
  if not os.path.exists(path):
    raise FileNotFoundError(f"Model file not found: {path}")
  checkpoint = torch.load(path, map_location=device)
  # self.q_net.load_state_dict(checkpoint['q_net_state'])
  # self.target_q_net.load_state_dict(checkpoint['target_q_net_state'])
  print(f"Model loaded from {path}")
  return checkpoint

#---------------------- 管理json格式参数函数 -------------------------
#                        2026/1/22

def utils_readParams(json_path:str, sub_name:str) -> dict:
  '''
    从 JSON 文件中读取指定顶层子配置块。
    params:
      json_path - json格式参数文件路径
      sub_name - 要读取的顶层键名（如 "model", "env"）
    returns:
      dict - 配置子块
  '''
  with open(json_path, 'r') as f:
    config = json.load(f)
  if sub_name not in config:
    raise KeyError(f"Key '{sub_name}' not found in {json_path}")
  sub_config = config[sub_name]
  return sub_config

#---------------------- 实用工具函数 -------------------------
#                        2026/1/23

class utils_ReplayBuffer:
  '''
    普通 Replay Buffer
  '''
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def add_trajectory(self, trajectory:Trajectory):
    self.buffer.append(trajectory)

  def sample(self, batch_size):
    batch = dict(states=[], actions=[], next_states=[], rewards=[], dones=[])
    for _ in range(batch_size):
      traj = random.sample(self.buffer, 1)[0]
      step_state = np.random.randint(len(traj))
      batch['states'].append(traj.states[step_state])
      batch['next_states'].append(traj.states[step_state+1])
      batch['actions'].append(traj.actions[step_state])
      batch['rewards'].append(traj.rewards[step_state])
      batch['dones'].append(traj.dones[step_state])
    batch['states'] = np.array(batch['states'])
    batch['next_states'] = np.array(batch['next_states'])
    batch['actions'] = np.array(batch['actions'])
    return batch

  def __len__(self):
    '''
      buffer 中的轨迹数，而非样本数
    '''
    return len(self.buffer)

class utils_prioritReplayBuffer:
  '''
    带优先级的经验回放池
  '''
  def __init__(self, capacity, transition_type:tuple, alpha:float=0.6, beta:float=0.4,
               beta_increment:float=0.001):
    '''
      params:
        capacity - 缓冲区最大容量
        transition_type - 用于存储经验的命名元组类型及属性
          (如 ('state', 'action', 'reward', 'next_state', 'done') )
        alpha - 优先级指数，控制采样策略 (0=均匀采样，1=完全按优先级)
        beta - 重要性采样权重参数，用于纠正偏差
        beta_increment - beta的增量（每次采样后增加）
    '''
    utils_autoAssign(self)
    self.Transition = namedtuple('Transition', field_names=transition_type)
    self.buffer = deque(maxlen=capacity)
    self.priorities = deque(maxlen=capacity)
    self.max_priority = 1.0

  def add(self, priority=None, **kwargs) -> None:
    '''
      params:
        **kwargs - 必须包含transition_type定义的所有字段
          例如: (state, action, reward, next_state, done)
        priority - 初始优先级，如果为None则使用当前最大优先级
    '''
    missing_fields = [field for field in self.Transition._fields if field not in kwargs]
    if missing_fields:
      raise ValueError(f"Missing fields: {missing_fields}")
    transition = self.Transition(**{field: kwargs[field] for field in self.Transition._fields})
    
    self.buffer.append(transition)

    if priority is None:
      priority = self.max_priority
    elif priority > self.max_priority:
      priority = self.max_priority
    self.priorities.append(priority)

  def sample(self, batch_size):
    '''
      从 buffer 中采样数据,数量为 batch_size ，如果 batch_size == None，则全部取出 
      return:  
        samples - [(s1, a1, r1, n_s1, d1), ...]
        transitions_dict - {
        'states': (state1, state2 ...),
        'actions': ...,
        'next_states': ...,
        'rewards': ...,
        'dones': ...
        }
      example:  
        state, action, reward, next_state, done = zip(*transitions)
        np.array(state), action, reward, np.array(next_state), done
    '''
    priority_array = np.array(self.priorities, dtype=np.float32)
    probs = priority_array**self.alpha
    probs /= probs.sum()+1e-9

    indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
    samples = [self.buffer[i] for i in indices]

    transitions_dict = {}
    for field in self.Transition._fields:
      field_values = [getattr(sample, field) for sample in samples]
      transitions_dict[field+'s'] = field_values

    return samples, transitions_dict

  def size(self):
    return len(self.buffer)

class utils_ReplayBuffer_HER(utils_ReplayBuffer):
  '''
    带 HER 的 ReplayBuffer，其中 state 包含两部分：[state, goal]
  '''
  def __init__(self, capacity):
    super().__init__(capacity=capacity)
  
  def _HER(self, new_goal, state, action, reward, next_state, done):
    '''
      将 (state, action, reward, next_state, done) 通过设置 new_goal 的方式，
      得到新轨迹 (state, action, reward', next_state, done')
    '''
    dis = np.linalg.norm(next_state[:2]-new_goal)
    reward = -1.0 if dis > 0.15 else 0
    done = False if dis > 0.15 else True
    state = np.hstack((state[:2], new_goal))
    next_state = np.hstack((next_state[:2], new_goal))
    return state, action, reward, next_state, done

  def sample(self, batch_size, her_ratio=0.8):
    batch = dict(states=[], actions=[], next_states=[], rewards=[], dones=[])
    for _ in range(batch_size):
      traj = random.sample(self.buffer, 1)[0]
      step_state = np.random.randint(len(traj))
      state = traj.states[step_state]
      next_state = traj.states[step_state + 1]
      action = traj.actions[step_state]
      reward = traj.rewards[step_state]
      done = traj.dones[step_state]

      if np.random.uniform() <= her_ratio:
        # 使用HER算法的future方案设置目标
        step_goal = np.random.randint(step_state+1, len(traj)+1)
        goal = traj.states[step_goal][:2]

        state, action, reward, next_state, done = self._HER(goal, state, action, reward, next_state, done)

      batch['states'].append(state)
      batch['next_states'].append(next_state)
      batch['actions'].append(action)
      batch['rewards'].append(reward)
      batch['dones'].append(done)

    batch['states'] = np.array(batch['states'])
    batch['next_states'] = np.array(batch['next_states'])
    batch['actions'] = np.array(batch['actions'])
    return batch

if __name__ == '__main__':
  trans = ('state', 'action', 'reward', 'next_state', 'done')
  buffer = utils_prioritReplayBuffer(capacity=100, transition_type=trans)
  for i in range(12):
    buffer.add(
        priority=i*0.1,
        state=i*0.1,
        action=i%2,
        reward=i*0.5,
        next_state=(i+1)*0.1,
        done=(i==9)
    )
  print(buffer.sample(1))

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

