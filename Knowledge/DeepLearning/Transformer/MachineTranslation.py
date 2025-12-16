import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(r'f:\GraduateStudent')
from Projects.Model.NN import Transformer
from Projects.Utils.tools import *

device = utils_getDevice()

# def test_transformer_complete():
#     """完整测试Transformer模型"""
#     print("=== 开始测试完整Transformer模型 ===")
    
#     # 小规模参数设置（便于快速测试）
#     src_vocab_size = 100
#     tgt_vocab_size = 80
#     d_model = 32
#     num_layers = 2
#     num_heads = 4
#     d_ff = 64
#     batch_size = 4
#     src_seq_len = 10
#     tgt_seq_len = 8
#     padding_idx = 0
    
#     # 创建模型
#     transformer = Transformer(
#         src_vocab_size=src_vocab_size,
#         tgt_vocab_size=tgt_vocab_size,
#         d_model=d_model,
#         num_layers=num_layers,
#         num_heads=num_heads,
#         d_ff=d_ff,
#         max_seq_length=32,
#         dropout=0.1,
#         padding_idx=padding_idx
#     ).to(device)
    
#     print("✅ 模型创建成功")
    
#     # 创建模拟输入数据
#     src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)).to(device)
#     tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    
#     print(f"源序列形状: {src.shape}")
#     print(f"目标序列形状: {tgt.shape}")
    
#     # 创建各种掩码
#     src_mask = UTools_createPaddingMask(src, padding_idx).to(device)
#     tgt_self_mask = UTools_createDecoderSelfAttentionMask(tgt, padding_idx).to(device)
#     cross_mask = UTools_createCrossAttentionMask(tgt, src, padding_idx).to(device)
    
#     print(f"源序列掩码形状: {src_mask.shape}")
#     print(f"目标序列自注意力掩码形状: {tgt_self_mask.shape}")
#     print(f"交叉注意力掩码形状: {cross_mask.shape}")
    
#     # 前向传播
#     try:
#         transformer.eval()
#         output, encoder_output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights = transformer(
#             src, tgt, src_mask, tgt_self_mask, cross_mask
#         )
        
#         print("✅ 前向传播成功")
#         print(f"解码器输出logits形状: {output.shape}")
#         print(f"编码器输出形状: {encoder_output.shape}")
#         print(f"编码器注意力权重层数: {len(enc_attn_weights)}")
#         print(f"解码器自注意力权重层数: {len(dec_self_attn_weights)}")
#         print(f"解码器交叉注意力权重层数: {len(dec_cross_attn_weights)}")
        
#         # 验证形状
#         assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size), "输出logits形状错误"
#         assert encoder_output.shape == (batch_size, src_seq_len, d_model), "编码器输出形状错误"
#         assert len(enc_attn_weights) == num_layers, "编码器注意力权重层数错误"
#         assert len(dec_self_attn_weights) == num_layers, "解码器自注意力权重层数错误"
#         assert len(dec_cross_attn_weights) == num_layers, "解码器交叉注意力权重层数错误"
        
#         # 验证单个注意力权重形状
#         for i, attn in enumerate(enc_attn_weights):
#             assert attn.shape == (batch_size, num_heads, src_seq_len, src_seq_len), f"编码器第{i}层注意力权重形状错误"
        
#         for i, attn in enumerate(dec_self_attn_weights):
#             assert attn.shape == (batch_size, num_heads, tgt_seq_len, tgt_seq_len), f"解码器自注意力第{i}层权重形状错误"
        
#         for i, attn in enumerate(dec_cross_attn_weights):
#             assert attn.shape == (batch_size, num_heads, tgt_seq_len, src_seq_len), f"解码器交叉注意力第{i}层权重形状错误"
        
#         print("✅ 所有形状验证通过")
        
#         # 测试单独的编码器和解码器
#         print("\n--- 测试单独组件 ---")
#         encoder_output_single, enc_attn_weights_single = transformer.encode(src, src_mask)
#         print(f"单独编码器输出形状: {encoder_output_single.shape}")
        
#         decoder_output_single, dec_self_attn_single, dec_cross_attn_single = transformer.decode(
#             tgt, encoder_output_single, tgt_self_mask, cross_mask
#         )
#         print(f"单独解码器输出形状: {decoder_output_single.shape}")
        
#         # 验证单独组件输出与完整模型一致
#         assert torch.allclose(encoder_output_single, encoder_output, atol=1e-6), "单独编码器输出与完整模型不一致"
#         print("✅ 单独组件测试通过")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ 测试失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def run_all_tests():
#   """运行所有测试"""
#   print("✨ 开始运行Transformer模型测试 ✨\n")
#   test1_success = test_transformer_complete()
#   print(f"完整模型测试: {'✅ 通过' if test1_success else '❌ 失败'}")

def tokenize_and_encode(text, word2id, BOS=1, EOS=2, UNK=3, add_bos_eos=True):
  '''
    将句子转为 ID 序列（自动添加 bos/eos）
  '''
  ids = [word2id.get(w, UNK) for w in text.split()]
  if add_bos_eos:
    ids = [BOS]+ids+[EOS]
  return ids

def simple_transformer_example():
  examples = [
    ("ich spreche fließend englisch .", "i speak fluent english ."),
    ("wir sind im kino .", "we are in the cinema ."),
    ("das ist die toilette .", "this is the toilet ."),
    ("sie geht ins kino .", "she is going to the cinema ."),
    ("ich bin fließend .", "i am fluent ."),
    # ("wo ist das buch ?", "where is this book ?"),
    ("wir lieben dieses buch .", "we love this book ."),
    ("sie ist im kino .", "she is in the cinema ."),
    ("ich gehe ins kino .", "i am going to the cinema ."),
    ("das ist fließend englisch .", "this is fluent english ."),
    ("guten morgen , ich bin fließend .", "good morning , i am fluent ."),
    ("sie liebt dich .", "she loves you ."),
    ("das buch ist fließend .", "this book is fluent ."),
    ("ich liebe dieses buch .", "i love this book ."),
    ("sie liebt das buch .", "she loves the book ."),
    ("ich bin im buch .", "i am in the book ."),
    ("ist das buch fließend ?", "is this book fluent ?"),
    ("wo ist das kino ?", "where is the cinema ?"),
    # ("wo ist dieses kino ?", "where is this cinema ?"),
    ("wo ist das morgen ?", "where is the morning ?"),
    ("wo ist fließend englisch ?", "where is fluent english ?"),
    ("wo ist ich ?", "where is i ?"),
    ("wo ist wir ?", "where is we ?"),
    ("wo ist sie ?", "where is she ?"),
  ]
  test_examples = [
    ("wo ist das buch ?", "where is this book ?"),
    ("wo ist dieses kino ?", "where is this cinema ?"),
  ]
  PAD, BOS, EOS, UNK = 0, 1, 2, 3
  special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

  # 收集词汇
  src_words = set()
  tgt_words = set()
  for de, en in examples:
    src_words.update(de.split())
    tgt_words.update(en.split())
  src_vocab = special_tokens + sorted(src_words)    # 全部的德语（输入数据）
  tgt_vocab = special_tokens + sorted(tgt_words)    # 全部的英语（输出数据）
  print(f"🇩🇪 源词表大小: {len(src_vocab)} | 🇬🇧 目标词表大小: {len(tgt_vocab)}")

  src_word2id = {w: i for i, w in enumerate(src_vocab)}
  tgt_word2id = {w: i for i, w in enumerate(tgt_vocab)}
  tgt_id2word = {i: w for w, i in tgt_word2id.items()}
  src_seqs = [tokenize_and_encode(de, src_word2id) for de, _ in examples]   # 将所有德语句子用token ids表示
  tgt_seqs = [tokenize_and_encode(en, tgt_word2id) for _, en in examples]   # 将所有英语句子用token ids表示

  src_batch = pad_sequence([torch.tensor(s) for s in src_seqs], batch_first=True, padding_value=PAD).to(device)
  tgt_batch = pad_sequence([torch.tensor(t) for t in tgt_seqs], batch_first=True, padding_value=PAD).to(device)
  # 将所有token ids填充后表示
  # pad_sequence(..., batch_first=True, padding_value=PAD): 
  #     batch_first - 输出形状为 (batch_size, L_max)
  #     padding_value=PAD - 指定填充的数值（这里用 0 表示 <pad>）
  print(f'src={src_batch.shape}, tgt={tgt_batch.shape}')
  
  # 创建模型
  model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=256,
    num_layers=6,
    num_heads=8,
    d_ff=512,
    max_seq_length=20,
    dropout=0.1,
    padding_idx=PAD
  ).to(device)
  criterion = nn.CrossEntropyLoss(ignore_index=PAD)
  optimizer = optim.Adam(model.parameters(), lr=3e-4)

  # 训练过程
  model.train()
  for step in range(200):
    optimizer.zero_grad()
    
    # 构建掩码
    src_mask = UTools_createPaddingMask(src_batch, PAD).to(device)
    tgt_self_mask = UTools_createDecoderSelfAttentionMask(tgt_batch, PAD).to(device)
    cross_mask = UTools_createCrossAttentionMask(tgt_batch, src_batch, PAD).to(device)
    
    # 前向 → [B, T, V]
    logits, *_ = model(src_batch, tgt_batch, src_mask, tgt_self_mask, cross_mask)
    
    # 对齐：预测 tgt[i] ← tgt[:i]
    y_true = tgt_batch[:, 1:]      # <bos> w1 w2 ... <eos> → w1 w2 ... <eos>
    y_pred = logits[:, :-1, :]    # 预测长度需匹配
    # print(y_true, y_true.shape)
    
    loss = criterion(y_pred.reshape(-1, len(tgt_vocab)), y_true.reshape(-1))
    loss.backward()
    optimizer.step()
    
    print(f"Step {step+1:2d} | Loss: {loss.item():.4f}")

  # 推理过程
  model.eval()
  results = []
  with torch.no_grad():
    for i, (de_sent, en_ref) in enumerate(test_examples):
      # 编码源句
      src_ids = torch.tensor(tokenize_and_encode(de_sent, src_word2id)).unsqueeze(0).to(device)  # [1, L]
      src_mask = UTools_createPaddingMask(src_ids, PAD).to(device)
      enc_out, _ = model.encode(src_ids, src_mask)

      # 自回归生成
      tgt_input = torch.tensor([[BOS]]).to(device)  # 初始 <bos>
      generated = []

      for _ in range(15):  # 最大生成长度
        tgt_self_mask = UTools_createDecoderSelfAttentionMask(tgt_input, PAD).to(device)
        cross_mask = UTools_createCrossAttentionMask(tgt_input, src_ids, PAD).to(device)
        logits, *_ = model.decode(tgt_input, enc_out, tgt_self_mask, cross_mask)
        # print(logits, logits.shape)
        next_token = logits[0, -1].argmax().item()
        if next_token == EOS:
            break
        generated.append(next_token)
        tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]]).to(device)], dim=1)

      # 解码为文本
      pred_words = [tgt_id2word.get(tid, '<unk>') for tid in generated]
      pred_sent = " ".join(pred_words)
      results.append((de_sent, en_ref, pred_sent))

  # ===== 对比展示 =====
  print("\n{:<25} | {:<30} | {:<30}".format("🇩🇪 德语输入", "🇬🇧 真实英语", "模型生成"))
  print("-" * 90)
  for de, ref, pred in results:
    # 清理标点前的空格（如 " ." → "."）
    ref = ref.replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    pred = pred.replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    print(f"{de:<25} | {ref:<30} | {pred:<30}")  

if __name__ == "__main__":
#   run_all_tests()
  print("\n=================开始Transformer简单使用示例=====================\n")
  simple_transformer_example()
