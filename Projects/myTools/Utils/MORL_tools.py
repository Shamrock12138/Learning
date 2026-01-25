#                        MORL相关工具函数
#                           2026/1/22
#                            shamrock

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from myTools.Utils.tools import *
from myTools.Utils.MORL_config import *

#---------------------- 网络结构 -------------------------
#                      2026/1/22

class EQL_Network(torch.nn.Module):
  def __init__(self, state_size, action_size, reward_size):
    super().__init__()
    utils_autoAssign(self)
    input_size = state_size+reward_size   # 输入 状态+偏好
    output_size = action_size*reward_size # 输出 动作+偏好
    self.feature_net = nn.Sequential(
      nn.Linear(input_size, input_size*16), nn.ReLU(),
      nn.Linear(input_size*16, input_size*32), nn.ReLU(),
      nn.Linear(input_size*32, input_size*64), nn.ReLU(),
      nn.Linear(input_size*64, input_size*32), nn.ReLU(),
    )
    self.q_head = nn.Linear(input_size*32, output_size)

  def H(self, Q: torch.Tensor, w:torch.Tensor, s_num:int, w_num:int):
    '''
      找到当前 w 偏好下标量化最大的Q(s, a*, w'*)值向量
      params:
        Q - [B*action_size, reward_size]
        w - [B, reward_size]
        s_num - 
        w_num - 
      returns:
        HQ - [s_num, w_num, reward_size]
    '''
    reshaped_Q = Q.view(s_num, w_num, self.action_size, self.reward_size)
    reordered_Q = reshaped_Q.permute(1, 0, 2, 3).contiguous()
    flat_Q = reordered_Q.view(-1, self.reward_size)
    # [w_num * s_num * action_size, reward_size]
    extended_Q = flat_Q.repeat_interleave(w_num, dim=0)
    # [w_num * s_num * action_size * w_num, reward_size]

    reordered_w = w.view(s_num, w_num, self.reward_size).permute(1, 0, 2)
    extended_w = reordered_w.unsqueeze(2).repeat(1, 1, self.action_size*w_num, 1)
    extended_w = extended_w.view(-1, self.reward_size)
    # [w_num * s_num * action_size * w_num, reward_size]

    inner_products = torch.einsum('bi,bi->b', extended_Q, extended_w)
    inner_products_grouped = inner_products.view(w_num*s_num, self.action_size*w_num)
    # [w_num * s_num, action_size * w_num]

    max_indices = inner_products_grouped.argmax(dim=1)  # [w_num * s_num]
    batch_indices = torch.arange(w_num*s_num)
    flat_Q_indices = batch_indices*self.action_size+(max_indices//w_num)
    HQ = flat_Q[flat_Q_indices] # [w_num * s_num, reward_size]
    HQ = HQ.view(w_num, s_num, self.reward_size).permute(1, 0, 2).contiguous()
    # [s_num, w_num, reward_size]

    return HQ.view(-1, self.reward_size)

  def forward(self, state, preference, w_num=1):
    '''
      params:
        state - [B, state_size]
        preference - [B, reward_size]
        s_num - states_num
        w_num - weights_num
      returns:
        q - [B, action_size, reward_size]
          在 B 状态下（一般包含s和w，即当前状态和偏好），采取a动作，对于r目标的价值函数
        hq - [s_num, w_num, reward_size]
          HQ(s1, w1, r1) = argQmax( <w1T,Q(s1, w', a', r1)> )
    '''
    s_num = int(preference.size(0)/w_num)
    x = torch.cat((state, preference), dim=1)
    features = self.feature_net(x)
    q = self.q_head(features)
    q = q.view(-1, self.action_size, self.reward_size)
    hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)

    return hq, q

#---------------------- Trainer -------------------------
#                      2026/1/25

# class MORL_Trainer:
#   def __init__(self, agent:MORL_ModelConfig, params):
#     utils_setAttr(self, params)
#     self.agent = agent

#   def train(self):
#     '''
#       开始训练，
#     '''
#     with tqdm(total=int(self.episodes_num), desc=self.agent.name+' Iteration') as pbar:
      
    
  

if __name__ == '__main__':
  a = torch.arange(2*3)
  print(a)
  print(a*3)


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

