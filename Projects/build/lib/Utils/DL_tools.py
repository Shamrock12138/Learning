#                       Deep Learning 工具函数
#                           2025/11/30
#                            shamrock


#----------------------掩码生成工具（transformer用）-------------------------
#                             2025/11/11

import torch

def DTools_createPaddingMask(seq, padding_idx=0):
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

def DTools_createLookAheadMask(seq_len):
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

def DTools_createDecoderSelfAttentionMask(tgt_seq, padding_idx=0):
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

def DTools_createCrossAttentionMask(tgt_seq, src_seq, padding_idx=0):
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

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
