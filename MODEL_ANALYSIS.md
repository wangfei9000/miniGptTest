# MiniStoryGPT 模型代码分析文档

本文档梳理 `model.py` 中的 GPT 语言模型实现，并串联数据准备、训练与生成流程，便于阅读与二次实验。

---

## 1. 项目定位与整体逻辑

MiniStoryGPT 是一个**教学向、从零实现**的小型 GPT：结构对齐 *Attention is All You Need* 与 Karpathy nanoGPT 思路，用 TinyStories 儿童故事数据训练，生成简短英文故事。

端到端流水线：

```
TinyStories 原始文本
        │
        ▼
prepare_data.py  ──►  GPT-2 BPE 分词
        │              统计频次，保留 Top-10K token
        │              建立 old↔new 映射
        ▼
data/train.bin, val.bin + old_to_new.pt, new_to_old.pt
        │
        ▼
train.py  ──►  GPTLanguageModel 自回归训练（下一词预测）
        │      周期性评估 loss、采样故事、保存 checkpoint
        ▼
model_*.pth / MiniStoryGPT-30M.pth
        │
        ▼
generate.py  ──►  prompt → remap → generate → decode → 故事文本
```

核心建模目标：**给定前文 token，预测下一个 token**（因果语言模型 / Causal LM）。

---

## 2. 超参数（目标约 20M～30M 参数）

定义于 `model.py` 顶部：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `batch_size` | 32 | 并行序列条数 |
| `block_size` | 512 | 最大上下文长度（训练窗口与注意力 mask 尺寸） |
| `n_embd` | 768 | Token / 位置嵌入维度，也是 Transformer 隐藏维度 |
| `n_head` | 8 | 多头注意力头数；每头维度 `head_size = 768 / 8 = 96` |
| `n_layer` | 2 | Transformer Block 层数（很浅，利于小算力实验） |
| `dropout` | 0.2 | Attention / FFN / 投影 dropout |
| `use_sinusoidal` | `False` | `False` 用可学习位置嵌入；`True` 用正弦位置编码 |
| `device` | CUDA 或 CPU | 自动选择设备 |

训练侧（`train.py`）另有：`max_iters=100000`、`learning_rate=3e-4`、`eval_interval=500` 等。

> 说明：`train.py` 从 `model` 导入 `eval_iters`，但当前 `model.py` 未定义该变量；若直接跑训练需在 `model.py` 中补充，例如 `eval_iters = 200`。

---

## 3. 模型架构总览

```
输入 idx: (B, T)  token id
    │
    ├─ Token Embedding     → (B, T, n_embd)
    ├─ Position Embedding  → (B, T, n_embd)   # 或正弦 PE
    │         +
    ▼
x = tok_emb + pos_emb
    │
    ▼
× n_layer 个 Block:
    Pre-LN → Multi-Head Self-Attention → 残差
    Pre-LN → Feed-Forward (4×n_embd)   → 残差
    │
    ▼
LayerNorm (ln_f)
    │
    ▼
LM Head: Linear(n_embd → vocab_size)
    │
    ▼
logits: (B, T, vocab_size)
    │
    └─ 若有 targets → CrossEntropy Loss（展平后逐位置）
```

这是典型的 **Decoder-only GPT**：仅用因果自注意力（下三角 mask），不能看未来 token。

---

## 4. 关键模块与注释代码

### 4.1 单头自注意力 `Head`

作用：对每个位置计算 Q/K/V，做缩放点积注意力，并用下三角 mask 保证因果性。

```python
class Head(nn.Module):
    """ Single head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        # 三个线性投影：不共享参数
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 下三角因果 mask，尺寸 = block_size × block_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape          # B=batch, T=序列长, C=n_embd
        k = self.key(x)            # (B, T, hs)
        q = self.query(x)          # (B, T, hs)

        # 缩放点积注意力：wei[b,i,j] = 位置 i 对位置 j 的注意力分数
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,T)

        # 把未来位置置为 -inf，softmax 后变为 0 → 无法“偷看”未来
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)          # (B, T, hs)
        out = wei @ v              # (B, T, hs)：按注意力权重聚合 Value
        return out
```

要点：

- `* head_size**-0.5`：防止点积过大导致 softmax 饱和。
- `tril[:T, :T]`：实际序列长度可能小于 `block_size`，只取左上角子矩阵。

---

### 4.2 多头注意力 `MultiHeadAttention`

多个 `Head` 并行，在特征维拼接后再投影回 `n_embd`。

```python
class MultiHeadAttention(nn.Module):
    """ Multi-head attention (paper Section 3.2.2) """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 拼接后维度 = num_heads * head_size，再投回 n_embd
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 在最后一维拼接
        out = self.dropout(self.proj(out))
        return out
```

多头含义：不同头可关注不同子空间（句法、局部依赖、长程依赖等）。

---

### 4.3 前馈网络 `FeedForward`

每个位置独立的两层 MLP（位置级，不跨 token）：

```python
class FeedForward(nn.Module):
    """ Position-wise feed-forward (paper Eq 2) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 先扩到 4 倍（Transformer 惯例）
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 再压回 n_embd
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

---

### 4.4 Transformer Block（Pre-LN + 残差）

```python
class Block(nn.Module):
    """ Transformer block: self-attn + FFN，带残差与 LayerNorm """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN：先 Norm 再子层，再残差相加（比 Post-LN 更易训练）
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

数据流示意（单层）：

```
x ──► LN ──► MHA ──► + ──► LN ──► FFN ──► + ──► x'
      └──────────────┘      └──────────────┘
```

---

### 4.5 完整语言模型 `GPTLanguageModel`

#### 初始化

```python
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # token id → 向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # 默认可学习位置嵌入；也可用正弦编码（use_sinusoidal=True）
        if not use_sinusoidal:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 堆叠 n_layer 个 Block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)                 # 最终 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)     # 投影到词表 logits

        self.apply(self._init_weights)  # 权重 ~ N(0, 0.02)
```

权重初始化：Linear / Embedding 权重正态 `std=0.02`，Linear bias 置 0——与 GPT-2 / nanoGPT 常见设定一致。

#### 前向：嵌入 → Blocks → logits / loss

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)  # (B,T,C)

    if use_sinusoidal:
        # 正弦位置编码（论文 Section 3.5）：固定公式，无可学习参数
        pos = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2, device=device) * -(math.log(1e4) / n_embd)
        )
        pe = torch.zeros(T, n_embd, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pos_emb = pe.unsqueeze(0).expand(B, -1, -1)
    else:
        # 可学习位置嵌入：为每个位置 0..T-1 查表
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)

    x = tok_emb + pos_emb   # 内容 + 位置
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    if targets is None:
        loss = None
    else:
        # CrossEntropy 需要 (N, C) 与 (N,)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss
```

训练时 `targets` 通常是输入右移一位：位置 `t` 的标签是位置 `t+1` 的真实 token。

#### 自回归生成 `generate`

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # 只保留最后 block_size 个 token（超出上下文窗口会截断）
        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond)

        # 只看最后一个时间步的分布
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # 按概率多项式采样（非贪婪 argmax）
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

生成逻辑：循环「预测下一词 → 拼到序列末尾 → 再预测」，共 `max_new_tokens` 步。

---

## 5. 词表压缩与编解码辅助函数

原始 GPT-2 词表约 50k，本项目为控参数量与算力，在 `prepare_data.py` 中只保留语料中**出现频率最高的 10,000 个 token**，并重新编号为 `0 .. vocab_size-1`。

```python
def remap_tokens(encoded_tokens, old_to_new):
    """GPT-2 token id → 压缩词表 id；未知 token 映射为 0（当作 UNK）"""
    return [old_to_new.get(token, 0) for token in encoded_tokens]

def custom_decode(new_tokens_list, new_to_old, enc):
    """压缩词表 id → GPT-2 token id → 用 tiktoken 解码成文本"""
    original_tokens = [new_to_old.get(new_id, 0) for new_id in new_tokens_list]
    return enc.decode(original_tokens)
```

生成时完整路径：

1. `enc.encode(prompt)` → GPT-2 ids  
2. `remap_tokens(...)` → 模型词表 ids  
3. `model.generate(...)` → 新 token 序列  
4. `custom_decode(...)` → 可读英文故事  

---

## 6. 训练侧如何喂数据（与模型配合）

`train.py` 中的 batch 构造：

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # x: 连续 block_size 个 token
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y: 向右错一位，做下一词预测
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
```

单步训练：

```python
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

优化器为 AdamW，`lr=3e-4`；每 5000 step 存 checkpoint，结束时保存 `model_*.pth`。

---

## 7. 形状速查表

| 张量 | 形状 | 说明 |
|------|------|------|
| `idx` | `(B, T)` | 输入 token ids，`T ≤ block_size` |
| `tok_emb` / `pos_emb` / `x` | `(B, T, n_embd)` | 768 维 |
| 单头 `q,k,v` | `(B, T, head_size)` | `head_size=96` |
| 注意力权重 `wei` | `(B, T, T)` | 因果 mask 后的 softmax |
| `logits` | `(B, T, vocab_size)` | 未归一化分数；`vocab_size≈10000` |
| `targets`（训练） | `(B, T)` → 展平 `(B*T,)` | 与展平 logits 对齐算 CE |

---

## 8. 设计取舍小结

| 设计 | 选择 | 原因 / 影响 |
|------|------|-------------|
| Decoder-only | 因果自注意力 | 适合故事续写 |
| Pre-LN + 残差 | Block 内先 Norm | 小模型训练更稳 |
| 词表压缩到 ~10K | Top 频 token | 缩小 `Embedding` / `lm_head`，降参数与显存 |
| `n_layer=2` | 很浅 | 教育演示、小算力；表达力有限 |
| `n_embd=768, n_head=8` | 与 GPT-2 small 同宽但更深通常更多层 | 宽度保留、深度压缩 |
| 多项式采样生成 | `multinomial` | 比贪婪解码更多样，也可能更跳戏 |
| 位置编码可切换 | 默认可学习 PE | 与 GPT-2 一致；可选论文正弦 PE |

---

## 9. 文件职责对照

| 文件 | 职责 |
|------|------|
| `model.py` | 网络结构、超参数、`generate`、token 映射辅助函数 |
| `prepare_data.py` | 下载 TinyStories、Top-10K 词表、写出 `data/*` |
| `train.py` | 数据加载、loss 评估、训练循环、采样与 checkpoint |
| `generate.py` | 加载权重，从 prompt 生成故事 |
| `notebook.ipynb` | 注意力小例子与加载/采样演示 |

---

## 10. 阅读建议（按认知顺序）

1. 先看 `Head.forward`：理解 QK 点积、mask、加权 V。  
2. 再看 `Block`：注意力 + FFN + 残差如何叠层。  
3. 再看 `GPTLanguageModel.forward`：嵌入如何变成 logits / loss。  
4. 最后看 `generate` 与 `prepare_data` / `train` 的 batch，把「训练目标」和「推理循环」连起来。

若只改一处做实验，常见切入点：`n_layer`、`block_size`、`dropout`、`use_sinusoidal`，或生成时的采样策略（温度 / top-k 等，当前代码尚未实现，可自行扩展）。
