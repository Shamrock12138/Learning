#                           各种工具函数
#                           2025/10/24
#                            shamrock

from functools import wraps
import numpy as np
import time, torch, inspect

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

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

