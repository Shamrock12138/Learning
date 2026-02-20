#                        MORL相关工具函数
#                           2026/1/22
#                            shamrock

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from myTools.Utils.tools import *
from myTools.Utils.MORL_config import *

#---------------------- 其他 -------------------------
#                      2026/1/27

# def MRTools_GetRandomWeights(shape, device):
#   '''
#     生成随机权重向量或矩阵，并进行归一化

#     params:
#       shape - 权重形状，可以是单个整数或元组
#         整数: 生成向量 (n,)
#         元组: 生成矩阵 (n, m)
      
#   '''

#   weights = torch.randn(*shape, device=device)
#   weights = torch.abs(weights)
#   norm = torch.norm(weights, p=1, dim=1, keepdim=True)
#   weights = weights/norm
#   return weights

#---------------------- Envelope Q Learning 相关 -------------------------
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

class EQL_Trainer:
  def __init__(self, agent:MORL_ModelConfig, env:MORL_EnvConfig, 
               buffer:utils_ReplayBuffer_Priority, model:EQL_Network, 
               target_model:EQL_Network, device, params:dict):
    utils_autoAssign()
    utils_setAttr(self, params)
    self.history = {
      'loss':[],
      'reward':[]
    }

    # 同伦优化相关配置
    self.beta_init = self.beta
    self.beta_uplim = 1.00
    self.tau = 1000.
    self.beta_expbase = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./self.episode_num))
    self.beta_delta = self.beta_expbase / self.tau

    # ε-greedy相关配置
    self.epsilon_delta = (self.epsilon-0.05)/self.episode_num

  def sample(self) -> dict:
    return self.buffer.sample_sample(self.batch_size)

  def memorize(self, state:np.ndarray, action:np.ndarray, next_state:np.ndarray, 
               reward:np.ndarray, done:bool):
    state = torch.from_numpy(state).float().to(self.device)
    next_state = torch.from_numpy(next_state).float().to(self.device)
    reward = torch.from_numpy(reward).float().to(self.device)
    action = int(action.item())
    td_error = self.agent.get_td_error(state, action, next_state, reward, done)
    priority = torch.abs(td_error)+1e-5
    self.buffer.add_sample(Sample(state, action, reward, next_state, done), priority)

  def episode_end(self):
    '''
      每次 episode 结束时更新
    '''
    self.agent.w_kept = None
    # 探索率衰减
    self.epsilon -= self.epsilon_delta
    # 同伦优化
    self.beta += self.beta_delta
    self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

  def take_action(self, state):
    force_explore = (
      len(self.buffer) < self.batch_size or 
      torch.rand(1, device=self.device).item() < self.epsilon
    )
    return self.agent.take_action(state, force_explore)
  
  def update(self):
    d = self.sample()
    batch = {k: v[:5] for k, v in d.items()}
    loss_w = {k: v[-2:] for k, v in d.items()}
    return self.agent.update(batch, loss_w)

  def train(self, episodes_num, probe):
    for episode in tqdm(range(episodes_num), desc=self.agent.name+' Iteration'):
      done = False
      tot_reward = 0
      loss, cnt = 0, 0
      state, _ = self.env.reset()
      while not done:
        action = self.take_action()
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.memorize(state, action, next_state, reward, done)
        if len(self.buffer) > self.buffer.batch_size:
          loss += self.update()
        tot_reward += (probe.cpu().numpy().dot(reward))*np.power(self.agent.gamma, cnt)
        cnt += 1
        state = next_state
        if cnt > 100:
          break
      self.episode_end()
      self.history['reward'].append(tot_reward)
      self.history['loss'].append(loss)
    return self.history

if __name__ == '__main__':
  a = torch.arange(2*3)
  print(a)
  print(a*3)


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

