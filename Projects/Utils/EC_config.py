#                           EC相关的配置类
#                           2025/12/17
#                            shamrock

from abc import ABC, abstractmethod
from typing import Any

class EC_Model(ABC):
  '''
    为各类 EC 算法提供统一的外部调用接口。

    该类本身不实现具体算法逻辑，而是定义标准接口协议：
      - 子类必须实现 `run` 方法，作为模型的核心执行入口
  '''
  def __call__(self, *input, **kwds):
    return self.run(*input, **kwds)

  @abstractmethod
  def run(self):
    pass

  @abstractmethod
  def init_population(self):
    pass

  def evalution(self):
    pass

  @abstractmethod
  def selection(self):
    pass

  @abstractmethod
  def crossover(self):
    pass

  @abstractmethod
  def mutation(self):
    pass

class EC_Problem(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def fitness(self, x):
    '''
      用户实现，返回 x 的适应度
    '''
    pass

  @abstractmethod
  def decode(self, chorm, x_min=-1, x_max=2, n_bits=16):
    '''
      将 n_bits 位的二进制 chorm 映射到 [x_min, x_max] 范围内
    '''
    binary_str = ''.join(map(str, chorm))
    integer = int(binary_str, 2)
    return x_min+integer*(x_max-x_min)/(2**n_bits-1)

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
