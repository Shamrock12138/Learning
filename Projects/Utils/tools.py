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

#----------------------掩码生成工具（transformer用）-------------------------
#                             2025/11/11

import torch

def UTools_createPaddingMask(seq, padding_idx=0):
  '''
    创建填充掩码，用来忽略<PAD>位置
      params:
        seq - 输入序列 [batch_size, seq_len]
        padding_idx - 填充token的索引
      return:
        mask - 注意力掩码 [batch_size, 1, 1, seq_len]
  '''
  mask = (seq!=padding_idx).unsqueeze(1).unsqueeze(2)
  return mask

def UTools_createLookAheadMask(seq_len):
  """
    创建look-ahead掩码（防止解码器看到未来信息）
      params:
        seq_len - 序列长度
      return:
        mask - look-ahead掩码 [seq_len, seq_len]
  """
  mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
  mask = mask.masked_fill(mask == 1, float('-inf'))
  return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

def UTools_createDecoderSelfAttentionMask(tgt_seq, padding_idx=0):
  """
    创建解码器自注意力掩码（结合padding掩码和look-ahead掩码）
      params:
        tgt_seq - 目标序列 [batch_size, tgt_seq_len]
        padding_idx - 填充token的索引
      return:
        mask - 解码器自注意力掩码 [batch_size, tgt_seq_len, tgt_seq_len]
  """
  batch_size, tgt_seq_len = tgt_seq.shape
  device = tgt_seq.device  # 获取输入张量的设备
  
  # 创建padding掩码 [batch_size, 1, 1, tgt_seq_len]
  pad_mask = (tgt_seq != padding_idx).unsqueeze(1).unsqueeze(2)
  
  # 创建look-ahead掩码 [1, 1, tgt_seq_len, tgt_seq_len] - 确保在相同设备上
  look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1)
  look_ahead_mask = look_ahead_mask.masked_fill(look_ahead_mask == 1, float('-inf'))
  look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_seq_len, tgt_seq_len]
  
  # 结合两种掩码
  pad_mask_expanded = pad_mask.expand(-1, -1, tgt_seq_len, -1)
  mask = pad_mask_expanded & (look_ahead_mask == 0)
  
  return mask

def UTools_createCrossAttentionMask(tgt_seq, src_seq, padding_idx=0):
    """
    创建编码器-解码器注意力掩码，Decoder忽略源端<PAD>
    params:
      tgt_seq - 目标序列 [batch_size, tgt_seq_len]
      src_seq - 源序列 [batch_size, src_seq_len]
      padding_idx - 填充token的索引
    returns:
      mask - 交叉注意力掩码 [batch_size, 1, tgt_seq_len, src_seq_len]
    """
    # 创建目标序列padding掩码 [batch_size, tgt_seq_len]
    tgt_pad_mask = (tgt_seq != padding_idx)
    
    # 创建源序列padding掩码 [batch_size, src_seq_len]
    src_pad_mask = (src_seq != padding_idx)
    
    # 扩展维度以进行广播
    # [batch_size, tgt_seq_len, 1] & [batch_size, 1, src_seq_len] -> [batch_size, tgt_seq_len, src_seq_len]
    mask = tgt_pad_mask.unsqueeze(2) & src_pad_mask.unsqueeze(1)
    
    # 添加头维度 [batch_size, 1, tgt_seq_len, src_seq_len]
    mask = mask.unsqueeze(1)
    
    return mask

#----------------------探索方式（RL用）-------------------------
#                        2025/11/29

def UTools_MonteCarlo(MDP, INFO, pi, timestep_max, number):
  '''
    Monte Carlo 采样
      params:
        MDP - 描述环境马尔可夫决策过程 提供(S, A, P, R, gamma)
          S: list [states_num] 所有状态
          A: list [actions_num] 所有动作  
          P: list [states_num, actions_num, states_num] s采取了a动作后的next_s的所有概率
          R: list [states_num, actions_num] s采取了a动作后的奖励
        INFO - 提供环境信息 (end: list, states_num: int, actions_num: int)
        pi: dict - 策略
        timestep_max - 单条轨迹的最大时间步数
        number - 要生成的 episode 总数
      return:
        episodes - number 次 episode 的探索列表 [[s, a, r, s_next], ...]
  '''
  S, A, P, R, gamma = MDP
  end, states_num, actions_num = INFO
  episodes = []
  for _ in range(number):
    episode, timestep = [], 0
    s = S[np.random.randint(states_num)]
    while s not in end and timestep <= timestep_max:
      timestep += 1
      rand, temp = np.random.rand(), 0
      for a_opt in A:
        # temp += pi.get(s+' '+a_opt, 0)
        temp += pi[s][a_opt]
        if temp > rand:
          a = a_opt
          # r = R.get(s+' '+a, 0)
          r = R[s][a]
          break
      rand, temp = np.random.rand(), 0
      for s_opt in S:
        # temp += P.get(join(join(s, a), s_opt), 0)
        temp += P[s][a][s_opt]
        if temp > rand:
          s_next = s_opt
          break
      episode.append((s, a, r, s_next)) 
      s = s_next
    episodes.append(episode)
  return episodes

#----------------------评价指标（RL用）-------------------------
#                        2025/11/29

def UTools_OccupancyMeasure(episodes, s, a, timestep_max, gamma):
  '''
    计算 (s, a) 的占用度量，即：(s, a) 在环境交互过程的频率
      params:
        episodes - 轨迹列表，格式[(s₀, a₀, r₀, s₁), (s₁, a₁, r₁, s₂), ...]
        s, a - 要统计的状态和动作
        timestep_max - 最大步长
        gamma - 折扣因子
      return:
        rho - (s, a) 在环境交互过程的频率
  '''
  rho = 0
  total_times = np.zeros(timestep_max)  # total_times[i] - 总步长>=i的次数
  occur_times = np.zeros(timestep_max)  # occur_times[i] - 第i步出现(s, a)的次数
  for episode in episodes:
    for i in range(len(episode)):
      (s_opt, a_opt, r, s_next) = episode[i]
      total_times[i] += 1
      if s == s_opt and a == a_opt:
        occur_times[i] += 1
  for i in reversed(range(timestep_max)):
    if total_times[i]:
      rho += gamma**i*occur_times[i]/total_times[i]
  return (1-gamma)*rho

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

