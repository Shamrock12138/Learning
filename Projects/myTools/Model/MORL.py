#                         多目标强化学习模型
#                           2026/1/22
#                            shamrock

from myTools.Utils.MORL_config import *
from myTools.Utils.MORL_tools import *
from myTools.Utils.tools import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import copy

#---------------------- Envelope Q Learning -------------------------
#                        2026/1/22

class EQL(MORL_ModelConfig):
  def __init__(self, env:MORL_EnvConfig, model:EQL_Network, buffer, 
               args_path, device):
    super().__init__()
    utils_autoAssign(self)
    utils_setAttr(self, utils_readParams(args_path, 'model'))

    self.model_ = model               # 当前网络
    self.model = copy.deepcopy(model) # 目标网络
    self.epsilon_delta = (self.epsilon-0.05)/self.episode_num

    self.beta_init = self.beta
    self.beta_uplim = 1.00
    self.tau = 1000.
    self.beta_expbase = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./self.episode_num))
    self.beta_delta = self.beta_expbase / self.tau

    self.trans_mem = deque()
    self.trans = namedtuple('trans', ['s', 'a', 'n_s', 'r', 'd'])
    self.priority_mem = deque()

    if self.optimizer == 'Adam':
      self.optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

    self.w_kept = None
    self.update_count = 0

  def take_action(self, state:np.ndarray, preference:Optional[torch.Tensor]=None):
    if preference is None:
      if self.w_kept is None:
        self.w_kept = torch.randn(self.model_.reward_size)
        self.w_kept = torch.abs(self.w_kept)/torch.norm(self.w_kept, p=1)
      preference = self.w_kept

    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    preference_tensor = preference.unsqueeze(0)
    with torch.no_grad():
      _, q_values = self.model_(state_tensor, preference_tensor)
    q_values = q_values.squeeze(0)
    scalarized_q = torch.matmul(q_values, preference)
    greedy_action = scalarized_q.argmax().item()
    if self.is_train == 1:
      # 检查是否强制探索（经验池未满或随机探索）
      force_explore = (
        len(self.trans_mem) < self.batch_size or 
        torch.rand(1, device=self.device).item() < self.epsilon
      )
      if force_explore:
        # 随机选择动作
        return np.random.randint(self.model_.action_size)
    return greedy_action


  @utils_timer
  def run(self, episode_num, probe):
    '''
      params:
        episode_num - 训练次数
        probe - 偏好，例：[0.5, 0.5]
    '''
    for episode in tqdm(range(episode_num), desc=self.name+'Iteration'):
      done = False
      tot_reward, gamma = 0, self.gamma
      state, _ = self.env.reset()
      while not done:
        action = self.take_action(state)
        next_state, reward, done, _ = self.env.step(action)
        self.buffer.memorize(state, action, next_state, reward, done)
        tot_reward += (probe.dot(reward))*gamma
        gamma *= self.gamma
        state = next_state



#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

