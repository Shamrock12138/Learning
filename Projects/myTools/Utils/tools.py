#                           各种工具函数
#                           2025/10/24
#                            shamrock

from functools import wraps
import numpy as np
import matplotlib.pyplot as plt

import time, torch, inspect, os, json, collections, random

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

def utils_showHistory(history:list, title:str, x_lable:str, y_lable:str, save_path=None):
  '''
    显示 history 的曲线图，x轴y轴为 x_lable, y_lable，标题为 title
      params:
        history - 列表
        title, x_lable, y_lable - 字符串
        save_path - 保存路径，默认None不保存
  '''
  episodes_list = list(range(len(history)))
  plt.plot(episodes_list, history)
  plt.xlabel(x_lable)
  plt.ylabel(y_lable)
  plt.title(title)
  plt.grid(True)  # 增强可读性
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

class utils_replayBuffer:
  def __init__(self, capacity=1000):
    self.buffer = collections.deque(maxlen=capacity)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size=None):
    '''
      从 buffer 中采样数据,数量为 batch_size ，如果 batch_size == None，则全部取出 
      return:  
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
    if batch_size is None:
      batch_size = self.size()
    transitions = random.sample(self.buffer, batch_size)
    state, action, reward, next_state, done = zip(*transitions)
    transitions_dict = {
      'states': np.array(state),
      'actions': action,
      'next_states': np.array(next_state),
      'rewards': reward,
      'dones': done
    }
    return transitions_dict, np.array(state), action, reward, np.array(next_state), done

  def size(self):
    return len(self.buffer)

# TODO
# class Utils_SavePath:
#   '''
#     管理 模型以及训练结果 的保存、格式
#   '''
#   def __init__(self):
#     pass

#   def 

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

