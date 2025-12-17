#                           进化计算模型
#                           2025/12/17
#                            shamrock

from ..Utils.EC_config import *
# from ..Utils.RL_tools import RTools_epsilon, Qnet, ReplayBuffer
from ..Utils.tools import utils_timer, utils_autoAssign

import numpy as np
import random

#---------------------- Genetic Algorithm -------------------------
#                           2025/12/17

class GA(EC_Model):
  def __init__(self, pop_size, chrom_len, max_gen, pc, pm):
    super().__init__()
    utils_autoAssign()
    self.pop = self.init_population()   # population
    self.fits = []  # fitness population
    self.chroms = []

  def init_population(self):
    return [np.random.randint(0, 2, self.chrom_len).tolist() for _ in range(self.pop_size)]

  def selection(self):
    selected = []
    for _ in range(self.pop_size):
      candidates = random.sample(list(zip(self.pop, self.fits)), 3)
      winner = max(candidates, key=lambda x: x[1])[0]
      selected.append(winner.copy())
    return selected
  
  def crossover(self, parent1, parent2):
    """单点交叉"""
    if random.random() < self.pc:
      point = random.randint(1, self.chrom_len - 1)
      child1 = parent1[:point] + parent2[point:]
      child2 = parent2[:point] + parent1[point:]
      return child1, child2
    return parent1.copy(), parent2.copy()

  def mutation(self):
    """位翻转变异"""
    for i in range(len(self.chroms)):
      if random.random() < self.pm:
        self.chroms[i] = 1 - self.chroms[i]
    return self.chroms

  def fitness(x):
    '''
      用户定义 适应度函数
    '''
    pass
  


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
