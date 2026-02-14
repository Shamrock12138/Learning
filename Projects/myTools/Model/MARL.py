#                         多智能体强化学习模型
#                           2026/1/22
#                            shamrock

from tqdm import tqdm

from myTools.Utils.tools import *
from myTools.Utils.config import *
from myTools.Utils.RL_config import *
from myTools.Utils.MARL_config import *

#---------------------- Independent 框架 -------------------------
#                            2026/2/14

class Independent_Trainer:
  '''
    Independent 框架的训练器
  '''
  def __init__(self, rl:RL_Model, rl_config, agents_num, env:MARL_EnvConfig):
    '''
      params:
        rl - 传统的RL算法
        rl_config - RL算法的参数配置文件路径(.json)
        agents_num - 需要agents的数量
    '''
    model_params = utils_readParams(rl_config, 'model')
    utils_autoAssign(self)
    self.agents = [rl(**model_params) for _ in range(agents_num)]
    self.name = f'Independent {rl.name}'

  def train(self, episodes_num):
    '''
      开始训练
    '''
    with tqdm(total=episodes_num, desc=f'{self.name}') as pbar:
      for episode in range(episodes_num):
        done = [False]*self.agents_num
        s = self.env.reset()
        trajs = [Trajectory() for _ in range(self.agents_num)]
        while not all(done):
          actions = [self.agent[i].take_action(s[i]) for i in range(self.agents_num)]
          next_s, r, done, info = self.env.step(actions)
          for i in range(self.agents_num):
            self.agents[i].update(trajs[i])



#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
