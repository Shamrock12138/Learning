import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
import math
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 1. 预处理：停用词过滤 + Subsampling
# ----------------------------
STOPWORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to", "of"}

class CBOWDataset(Dataset):
    def __init__(self, corpus, vocab_size=1000, window_size=2, subsample_threshold=1e-3):
        # 先过滤停用词！
        corpus = [
            [w for w in sent if w.lower() not in STOPWORDS]
            for sent in corpus
        ]
        
        self.window_size = window_size
        self.subsample_threshold = subsample_threshold
        
        # 构建词表
        word_counts = Counter(w for sent in corpus for w in sent)
        total_words = sum(word_counts.values())
        
        vocab_words = [w for w, _ in word_counts.most_common(vocab_size - 2)]
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.word2idx.update({w: i+2 for i, w in enumerate(vocab_words)})
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # 计算词频（用于 subsampling 和负采样）
        self.word_freq = {w: count / total_words for w, count in word_counts.items()}
        
        # Subsampling 概率: P_drop = 1 - sqrt(t / f(w))
        self.subsample_prob = {}
        for w in self.word2idx:
            if w in self.word_freq:
                f = self.word_freq[w]
                self.subsample_prob[w] = max(0.0, 1.0 - math.sqrt(subsample_threshold / f))
            else:
                self.subsample_prob[w] = 1.0  # <UNK> always kept
        
        # 构建负采样分布: P_n(w) ∝ count(w)^{0.75}
        self.noise_dist = torch.zeros(self.vocab_size)
        for i in range(2, self.vocab_size):  # 从 2 开始跳过 <PAD>, <UNK>
            word = self.idx2word[i]
            self.noise_dist[i] = (self.word_freq.get(word, 0) + 1e-10) ** 0.75
        self.noise_dist /= self.noise_dist.sum()  # 归一化
        
        # 生成样本
        self.samples = []
        for sentence in corpus:
            # Subsampling
            sentence = [
                w for w in sentence 
                if random.random() > self.subsample_prob.get(w, 1.0)
            ]
            if len(sentence) < 2:
                continue
                
            for i, target_word in enumerate(sentence):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                context_words = [sentence[j] for j in range(start, end) if j != i]
                if not context_words:
                    continue
                    
                context_idx = [self.word2idx.get(w, 1) for w in context_words]
                target_idx = self.word2idx.get(target_word, 1)
                self.samples.append((context_idx, target_idx))
        
        print(f"[Dataset] Vocab: {self.vocab_size}, Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
    def sample_noise(self, batch_size, num_neg):
        """按 unigram^0.75 分布采样负样本"""
        return torch.multinomial(self.noise_dist, batch_size * num_neg, replacement=True).view(batch_size, num_neg)

# ----------------------------
# 2. CBOW 模型（优化版）
# ----------------------------
class CBOWModel(nn.Module):
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
        return self.in_embed.weight.data  # (V, d)

# ----------------------------
# 3. 训练主程序
# ----------------------------
if __name__ == "__main__":
    # 示例语料（保留停用词，预处理会过滤）
    corpus = [
        "the quick brown fox jumps over the lazy dog".split(),
        "a quick brown dog outpaces a fox".split(),
        "foxes are quick and brown".split(),
        "dogs are loyal and lazy".split(),
        "the fox runs fast in the forest".split(),
        "a loyal dog protects its home".split(),
        "quick animals like fox and deer run fast".split(),
        "loyal dogs guard houses and families".split(),
        "brown foxes hide in forests".split(),
        "a quick fox jumps over a lazy dog".split(),
        "the loyal dog follows its master".split(),
    ]
    
    # 构建数据集（启用 subsampling + 停用词过滤）
    dataset = CBOWDataset(
        corpus,
        vocab_size=50,
        window_size=3,
        subsample_threshold=1e-2  # 关键！恢复 subsampling
    )
    
    # DataLoader
    def collate_fn(batch):
        contexts, targets = zip(*batch)
        contexts_padded = nn.utils.rnn.pad_sequence(contexts, batch_first=True, padding_value=0)
        targets = torch.stack(targets)
        return contexts_padded, targets
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 模型 & 优化器
    model = CBOWModel(vocab_size=dataset.vocab_size, embed_dim=128, num_neg=10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 训练
    print("Start training...")
    model.train()
    for epoch in range(200):
        total_loss = 0
        for context_batch, target_batch in dataloader:
            optimizer.zero_grad()
            
            # 采样负样本（使用真实分布）
            noise_batch = dataset.sample_noise(context_batch.size(0), model.num_neg)
            
            loss = model(context_batch, target_batch, noise_batch)
            loss.backward()
            
            # 梯度裁剪（防爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 30 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.5f}")
    
    # ----------------------------
    # 4. 评估词向量质量
    # ----------------------------
    word_vecs = model.embedding
    
    def most_similar(word, topk=5):
        if word not in dataset.word2idx:
            return f"'{word}' not in vocab"
        idx = dataset.word2idx[word]
        vec = word_vecs[idx]
        cos_sim = torch.cosine_similarity(vec.unsqueeze(0), word_vecs, dim=1)
        cos_sim[idx] = -10  # 排除自身
        top_vals, top_idx = torch.topk(cos_sim, topk)
        return [(dataset.idx2word[i.item()], round(v.item(), 3)) for i, v in zip(top_idx, top_vals)]
    
    print("\n" + "="*50)
    print("✅ Final Word Similarities (Optimized):")
    print("="*50)
    for word in ["fox", "dog", "brown", "quick", "loyal"]:
        print(f"{word:>8}: {most_similar(word)}")
    
    # 额外：检查高频词是否被压制
    print("\n[Diagnosis] Norm of frequent words:")
    for w in ["fox", "dog", "brown", "quick", "a", "the"]:
        if w in dataset.word2idx:
            norm = word_vecs[dataset.word2idx[w]].norm().item()
            print(f"  {w:>6}: ||v|| = {norm:.2f}")