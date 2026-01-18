#                           博弈论工具函数
#                           2025/12/21
#                            shamrock

import numpy as np

#---------------------- 求解 Pareto Optimal Solutoins 的方式-------------------------
#                             2025/12/21

def GTTools_ParetoFront(sets:np.array):
  '''
    通过求 pareto front 的方式，返回 sets 的 pareto optimal solutions。
      params:
        sets: np.array - 集合，行数为个数，列数为目标数。
          例如：np.array([[5, 2],
                  [3, 4],
                  [4, 1],
                  [2, 3]])
  '''
  m, l = sets.shape
  is_pareto = np.ones(m, dtype=bool)
  for i in range(m):
    if not is_pareto[i]:
      continue
    cond1 = np.all(sets >= sets[i], axis=1)   # 所有目标不差
    cond2 = np.any(sets > sets[i], axis=1)    # 至少有一个目标更好
    dominated_by = cond1&cond2&(np.arange(m)!=i)
    if np.any(dominated_by):
      is_pareto[i] = False
  return sets[is_pareto]

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
