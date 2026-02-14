#                         多智能体强化学习环境
#                           2026/1/22
#                            shamrock

from myTools.Utils.MARL_config import *

class MARL_Env_UAVs(MARL_EnvConfig):
  '''
    多无人机充电场景
  '''
  def __init__(self, n_agents=0, states_dim=0, actions_dim=0):
    super().__init__(n_agents, states_dim, actions_dim)

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

