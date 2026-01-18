
#                           神经网络模型
#                           2025/10/25
#                            shamrock

import torch, math
import torch.nn as nn
import torch.nn.functional as F

#------------------LSTM---------------------
#               2025/11/05

class LSTM(nn.Module):
  '''
    LSTM模型
  '''
  def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1,
               dropout=0.2):
    '''
      params:
        input_size - 输入特征维度
        hidden_size - 隐藏层神经元数量  
        num_layers - lstm堆叠层数  
        output_size - 输出维度  
        dropout - 层间dropout比例，在LSTM层与层之间应用的dropout概率（仅当 num_layers > 1 时有效）。
    '''
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout if num_layers>1 else 0
    )
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
    lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
    out = self.dropout(lstm_out[:, -1, :])
    out = self.linear(out)
    return out.squeeze(-1)

#------------------ Transformer ---------------------
#                    2025/11/11

# 模块：

class WordEmbedding(nn.Module):
  '''
    将输入的seq_len个 词编号(token ID) 转换成对应的seq_len x d_model位浮点数 
  '''
  def __init__(self, vocab_size, d_model, padding_idx=0):
    '''
      params:
        vocab_size - 词汇表的大小
        d_model - 嵌入维度
        padding_idx - 填充token的索引
    '''
    super().__init__()
    self.embedding = nn.Embedding(
      vocab_size,
      d_model,
      padding_idx=padding_idx
    )
    self.d_model = d_model
    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.embedding.weight)    # 使用 Xavier均匀分布 初始化self.embedding的权重
    if self.embedding.padding_idx is not None:
      with torch.no_grad():
        self.embedding.weight[self.embedding.padding_idx].fill_(0)

  def forward(self, x):
    '''
      params:
        x - 输入token ids, shape:[batch_size, seq_len]
      return:
        embeddings - [batch_size, seq_len, d_model]
    '''
    return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  '''
    给输入的[seq_len x d_model]位浮点数添加位置信息
  '''
  def __init__(self, d_model, max_seq_length=512, dropout=0.1):
    '''
      params:
        d_model - 模型的嵌入维度，同词嵌入维度
        max_seq_length - 支持的最大序列长度
        dropout - 对加入位置编码后的结果应用 dropout
    '''
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_seq_length, d_model) # [max_seq_length, d_model]
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # [max_seq_length, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float()*
                         (-math.log(10000.0)/d_model))    # [d_model/2]
    # position*div_term -> [max_seq_length, d_model/2]

    pe[:, 0::2] = torch.sin(position*div_term)    # 偶数列(0, 2, ...)
    pe[:, 1::2] = torch.cos(position*div_term)    # 奇数列(1, 3, ...)

    pe = pe.unsqueeze(0)    # 添加batch维度，[1, max_seq_length, d_model]
    self.register_buffer('pe', pe)    # 将pe注册为模型的缓冲区，而不是可学习参数

  def forward(self, x):
    '''
      params:
        x - 输入张量 [batch_size, seq_len, d_model]
    '''
    x = x+self.pe[:, :x.size(1)]
    return self.dropout(x)

class ScaleDotProductAttention(nn.Module):
  def __init__(self, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, Q, K, V, mask=None):
    '''
      parmas:
        Q - Query [batch_size, seq_len, d_k]
        K - Key [batch_size, seq_len, d_k]
        V - Value [batch_size, seq_len, d_v]
        mask - 注意力掩码 [batch_size, num_heads, seq_len, seq_len] 或可以广播的形状
          mask中为 0 的位置表示“不应被注意”，比如：未来词
      return:
        output - 注意力输出 [batch_size, seq_len, d_v]
        attn_weights - 注意力权重 [batch_size, seq_len, seq_len]
    '''
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k)   # [batch_size, seq_len, seq_len]
    if mask  is not None:
      scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights) # [batch_size, seq_len, seq_len]
    output = torch.matmul(attn_weights, V)    # [batch_size, seq_len, d_v]
    return output, attn_weights

class MultiHeadAttention(nn.Module):
  '''
    多头注意力机制，包含投影（不同线性变换改变QKV）
  '''
  def __init__(self, d_model, num_heads, dropout=0.1):
    super().__init__()
    assert d_model % num_heads == 0
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
    
    self.W_Q = nn.Linear(d_model, d_model, bias=False)
    self.W_K = nn.Linear(d_model, d_model, bias=False)
    self.W_V = nn.Linear(d_model, d_model, bias=False)
    self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    self.attention = ScaleDotProductAttention(dropout)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_model)
  
  def forward(self, Q, K, V, mask=None):
    '''
      params:
        Q K V - 输入张量 [batch_size, seq_len, d_model]
        mask - 注意力掩码 [batch_size, seq_len, seq_len]
      return:
        output - 多头注意力输出 [batch_size, seq_len, d_model]
        attn_weights - 注意力权重 [batch_size, num_heads, seq_len, seq_len]
    '''
    batch_size = Q.size(0)
    q_len = Q.size(1)
    k_len = K.size(1)
    
    residual = Q
    Q_proj = self.W_Q(Q).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
    K_proj = self.W_K(K).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
    V_proj = self.W_V(V).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
    
    if mask is not None and mask.dim() == 3:
      mask = mask.unsqueeze(1)
    
    attn_output, attn_weights = self.attention(Q_proj, K_proj, V_proj, mask)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
    
    output = self.dropout(self.W_O(attn_output))
    output = self.layer_norm(output + residual)
    return output, attn_weights

class PositionwiseFeedForward(nn.Module):
  '''
    位置前馈网络，升维到d_ff  
    "The feed-forward layers transform the representations independently at each position."
  '''
  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_model)
    self.activation = nn.GELU()

  def forward(self, x):
    residual = x
    output = self.linear1(x)
    output = self.activation(output)
    output = self.dropout(output)
    output = self.linear2(output)
    output = self.dropout(output)
    output = self.layer_norm(output+residual)
    return output

# 层级结构：

class EncoderLayer(nn.Module):
  '''
    编码器层/自注意层，包含多头注意力和前馈网络
  '''
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    '''
      params:
        x - 输入张量[batch_size, seq_len, d_model]
        mask - 注意力掩码[batch_size, seq_len, seq_len]
      return:
        output - 编码器输出 [batch_size, seq_len, d_model]
        attn_weights - 注意力权重 [batch_size, num_heads, seq_len, seq_len]
    '''
    attn_output, attn_weights = self.self_attention(x, x, x, mask)
    output = self.feed_forward(attn_output)
    return output, attn_weights
  
class DecoderLayer(nn.Module):
  '''
    解码器层，包含掩码多头自注意力、编码器-解码器注意力和前馈网络
  '''
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
    self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
    """
    params:
      x - 解码器输入 [batch_size, tgt_seq_len, d_model]
      encoder_output - 编码器输出 [batch_size, src_seq_len, d_model]
      self_attention_mask - 解码器自注意力掩码 [batch_size, tgt_seq_len, tgt_seq_len]
      cross_attention_mask - 编码器-解码器注意力掩码 [batch_size, tgt_seq_len, src_seq_len]
    return:
      output - 解码器层输出 [batch_size, tgt_seq_len, d_model]
      self_attn_weights - 自注意力权重 [batch_size, num_heads, tgt_seq_len, tgt_seq_len]
      cross_attn_weights - 交叉注意力权重 [batch_size, num_heads, tgt_seq_len, src_seq_len]
    """
    self_attn_output, self_attn_weights = self.self_attention(
      x, x, x, mask=self_attention_mask
    ) # 解码器掩码自注意力
    cross_attn_output, cross_attn_weights = self.cross_attention(
      self_attn_output, encoder_output, encoder_output, mask=cross_attention_mask
    ) # 编码器-解码器注意力
    output = self.feed_forward(cross_attn_output)
    return output, self_attn_weights, cross_attn_weights

# 编码器-解码器：

class Transformer_Encoder(nn.Module):
  '''
    tranformer Encoder
  '''
  def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
               max_seq_length=512, dropout=0.1, padding_idx=0):
    super().__init__()
    self.d_model = d_model
    self.num_layers = num_layers

    self.word_embedding = WordEmbedding(vocab_size, d_model, padding_idx)
    self.position_embedding = PositionalEncoding(d_model, max_seq_length, dropout)
    self.attention_layers = nn.ModuleList([
      EncoderLayer(d_model, num_heads, d_ff, dropout)
      for _ in range(num_layers)
    ])
    self.layer_norm = nn.LayerNorm(d_model) # 输出层归一化（可选）

  def forward(self, x, mask=None):
    '''
      params:
        x: 输入token ids [batch_size, seq_len]
        mask: 注意力掩码 [batch_size, seq_len, seq_len]
      return:
        output: 编码器输出 [batch_size, seq_len, d_model]
        all_attn_weights: 所有层的注意力权重列表
    '''
    embeddings = self.word_embedding(x)
    x = self.position_embedding(embeddings)
    all_attn_weights = []
    for layer in self.attention_layers:
      x, attn_weights = layer(x, mask)
      all_attn_weights.append(attn_weights)
    output = self.layer_norm(x)
    return output, all_attn_weights
    
class Transformer_Decoder(nn.Module):
  '''
    transformer decoder
  '''
  def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
               max_seq_length=512, dropout=0.1, padding_idx=0):
    super().__init__()
    self.d_model = d_model
    self.num_layers = num_layers

    self.word_embedding = WordEmbedding(vocab_size, d_model, padding_idx)
    self.position_embedding = PositionalEncoding(d_model, max_seq_length, dropout)
    self.decoder_layers = nn.ModuleList([
      DecoderLayer(d_model, num_heads, d_ff, dropout)
      for _ in range(num_layers)
    ])
    self.layer_norm = nn.LayerNorm(d_model)
    self.output_projection = nn.Linear(d_model, vocab_size)

  def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
    '''
      params:
        tgt - 目标序列token ids [batch_size, tgt_seq_len]
        encoder_output - 编码器输出 [batch_size, src_seq_len, d_model]
        tgt_mask - 目标序列掩码（防止看到未来词）[batch_size, tgt_seq_len, tgt_seq_len]
        memory_mask - 编码器输出掩码 [batch_size, tgt_seq_len, src_seq_len]
      return:
        output - 解码器输出logits [batch_size, tgt_seq_len, vocab_size]
        all_self_attn_weights - 所有层的自注意力权重
        all_cross_attn_weights - 所有层的交叉注意力权重
    '''
    embeddings = self.word_embedding(tgt)
    x = self.position_embedding(embeddings)
    all_self_attn_weights = []
    all_cross_attn_weights = []
    for layer in self.decoder_layers:
      x, self_attn_weights, cross_attn_weights = layer(
        x, encoder_output, tgt_mask, memory_mask
      )
      all_self_attn_weights.append(self_attn_weights)
      all_cross_attn_weights.append(cross_attn_weights)
    x = self.layer_norm(x)
    output = self.output_projection(x)
    return output, all_self_attn_weights, all_cross_attn_weights
  
# 最终实现：

class Transformer(nn.Module):
  '''
    transformer(encoder-decoder)
  '''
  def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff,
               max_seq_length=512, dropout=0.1, padding_idx=0):
    '''
      params:
        src_vocab_size: int - 源语言（输入序列）的词汇表大小，包含特殊符号（如 <PAD>, <UNK>, <BOS>, <EOS>）
        tgt_vocab_size: int - 目标语言（输出序列）的词汇表大小
        d_model: int - 模型的隐藏层维度（512、1024）
        num_layers: int - 编码器/解码器层数
        num_heads: int - 多头注意力机制中的头数
        d_ff: int - 前馈网络的中间层维度
        max_seq_length: int - 支持的最大序列长度，决定位置编码矩阵的大小
        dropout: float - Dropout概率
        padding_idx: int - 填充符 <PAD> 对应的词索引
    '''
    super().__init__()
    self.d_model = d_model
    self.num_layers = num_layers

    self.encoder = Transformer_Encoder(
      vocab_size=src_vocab_size,
      d_model=d_model,
      num_layers=num_layers,
      num_heads=num_heads,
      d_ff=d_ff,
      max_seq_length=max_seq_length,
      dropout=dropout,
      padding_idx=padding_idx
    )
    self.decoder = Transformer_Decoder(
      vocab_size=tgt_vocab_size,
      d_model=d_model,
      num_layers=num_layers,
      num_heads=num_heads,
      d_ff=d_ff,
      max_seq_length=max_seq_length,
      dropout=dropout,
      padding_idx=padding_idx
    )

    # 共享权重（可选：编码器和解码器的词嵌入可以共享权重）
    # self.encoder.word_embedding.embedding.weight = self.decoder.word_embedding.embedding.weight

    self._init_weights()

  def _init_weights(self):
    # 使用Xavier初始化输出投影层
    nn.init.xavier_uniform_(self.decoder.output_projection.weight)
    if self.decoder.output_projection.bias is not None:
      nn.init.constant_(self.decoder.output_projection.bias, 0)

  def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
    '''
      params:
        src - 源序列token ids [batch_size, src_seq_len]
        tgt - 目标序列token ids [batch_size, tgt_seq_len]
        src_mask - 源序列掩码 [batch_size, src_seq_len, src_seq_len]
        tgt_mask - 目标序列掩码 [batch_size, tgt_seq_len, tgt_seq_len]
        memory_mask - 编码器-解码器注意力掩码 [batch_size, tgt_seq_len, src_seq_len]
      return:
        output - 解码器输出logits [batch_size, tgt_seq_len, tgt_vocab_size]
        encoder_output - 编码器输出 [batch_size, src_seq_len, d_model]
        all_encoder_attn_weights - 编码器注意力权重
        all_decoder_self_attn_weights - 解码器自注意力权重
        all_decoder_cross_attn_weights - 解码器交叉注意力权重
    '''
    encoder_output, all_encoder_attn_weights = self.encoder(src, src_mask)
    output, all_decoder_self_attn_weights, all_decoder_cross_attn_weights = self.decoder(
      tgt, encoder_output, tgt_mask, memory_mask
    )
    return output, encoder_output, all_encoder_attn_weights, all_decoder_self_attn_weights, all_decoder_cross_attn_weights

  def encode(self, src, src_mask=None):
    """仅运行编码器"""
    return self.encoder(src, src_mask)

  def decode(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
    """仅运行解码器"""
    return self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

#-----------------Word2Vec-------------------
#                 2025/11/12

class CBOWModel(nn.Module):
  '''
    CBOW(Continuous Bag-of-Words) 实现，用上下文预测中心词
  '''
  def __init__(self, vocab_size, embed_dim, num_neg=10):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_neg = num_neg
    
    self.in_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    self.out_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    # 更好的初始化
    nn.init.normal_(self.in_embed.weight, mean=0, std=0.1)
    nn.init.normal_(self.out_embed.weight, mean=0, std=0.1)
    # 固定 <PAD> 向量为 0
    self.in_embed.weight.data[0] = 0
    self.out_embed.weight.data[0] = 0

  def forward(self, context_words, target_words, noise_words):
    '''
      params:
        context_words - 上下文词索引 [batch_size, max_context_len]
        target_words - 中心词索引 [batch_size]
        noise_words - 负样本词索引 [batch_size, num_neg]
      return:
        loss - 负采样损失
    '''
    # Step 1: 平均上下文向量
    context_embeds = self.in_embed(context_words)  # (B, L, d)
    mask = (context_words != 0).unsqueeze(-1).float()
    h = (context_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d)
    
    # Step 2: 正样本得分
    target_embeds = self.out_embed(target_words)  # (B, d)
    pos_score = (h * target_embeds).sum(dim=1)   # (B,)
    
    # Step 3: 负样本得分
    noise_embeds = self.out_embed(noise_words)   # (B, K, d)
    neg_score = torch.bmm(noise_embeds, h.unsqueeze(2)).squeeze(2)  # (B, K)
    
    # Loss
    pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-8)
    neg_loss = torch.log(torch.sigmoid(-neg_score) + 1e-8).sum(dim=1)
    loss = -(pos_loss + neg_loss).mean()
    return loss

  @property
  def embedding(self):
    '''
      返回最终使用的词向量（输入嵌入矩阵）
    '''
    return self.in_embed.weight.data  # (V, d)

#---------------------------------------

if __name__ == '__main__':
  # 参数设置
  vocab_size = 10000  # 词汇表大小
  d_model = 512       # 嵌入维度
  batch_size = 32
  seq_len = 128

  # 创建嵌入层
  embedding_layer = Transformer(vocab_size, d_model)

  # 模拟输入数据 (token ids)
  input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

  # 前向传播
  embeddings = embedding_layer(input_ids)
  print(f"输入形状: {input_ids.shape}")
  print(f"输出嵌入形状: {embeddings.shape}")

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
