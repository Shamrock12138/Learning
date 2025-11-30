#                           各种工具函数
#                           2025/10/24
#                            shamrock

from functools import wraps
import numpy as np
import time, torch

def utils_timer(func):
  '''
    计时器修饰器
  '''
  @wraps(func)
  def wrapper(*args, **kwargs):
    begin_time = time.time()
    ret = func(*args, **kwargs)
    end_time = time.time()
    print(f'run time: {end_time-begin_time} s')
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


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

