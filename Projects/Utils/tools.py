#                           各种工具函数
#                           2025/10/24
#                            shamrock

from functools import wraps
import numpy as np
import time, torch, inspect, os
import matplotlib.pyplot as plt

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

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

