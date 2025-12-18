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
  def __init__(self, problem:EC_Problem, pop_size, chrom_len, pc, pm):
    super().__init__()
    utils_autoAssign()
    # self.pop = self.init_population()   # population
    # self.fits = []  # fitness population
    # self.chroms = []

  def init_population(self):
    return [np.random.randint(0, 2, self.chrom_len).tolist() for _ in range(self.pop_size)]

  def selection(self, pop, fits):
    selected = []
    for _ in range(self.pop_size):
      candidates = random.sample(list(zip(pop, fits)), 3)
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

  def mutation(self, chroms):
    """位翻转变异"""
    for i in range(len(chroms)):
      if random.random() < self.pm:
        chroms[i] = 1-chroms[i]
    return chroms

  def run(self, x, gens):
    pop = self.init_population()
    for gen in range(gens):
      xs = [self.problem.decode(chorm, n_bits=self.chrom_len) for chorm in pop]
      fits = [self.problem.fitness(x) for x in xs]
      best_idx = np.argmax(fits)
    
      mating_pool = self.selection(population, fits)
      
      # 交叉 + 变异 → 新种群
      new_pop = []
      for i in range(0, self.pop_size, 2):
        p1, p2 = mating_pool[i], mating_pool[(i+1) % self.pop_size]
        c1, c2 = self.crossover(p1, p2)
        new_pop.append(self.mutation(c1))
        new_pop.append(self.mutation(c2))
      population = new_pop[:self.pop_size]

    # 输出结果
    best_x = self.problem.decode(population[best_idx], n_bits=self.chrom_len)
    print(f"\n✅ 最优解: x = {best_x:.6f}, f(x) = {self.problem.fitness(best_x):.6f}")

    


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
