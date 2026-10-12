# MiniStoryGPT 训练指南与源码分析

本文基于当前仓库中的实际代码、数据文件和模型 checkpoint 编写，目标是回答三个问题：

1. 这个项目的代码是怎样组织和运行的？
2. 如果要训练或继续训练模型，具体应该怎么做？
3. 模型、数据、训练循环中的关键代码分别在做什么，有哪些问题需要先修复？

> 重要结论：当前版本的 `train.py` **不能直接开始训练**，至少需要先解决
> `eval_iters` 未定义、训练数据被整体加载到 GPU、checkpoint 选择方式不可靠等问题。
> 第 7 节给出了最小修复方案。

---

## 1. 项目是什么

MiniStoryGPT 是一个教学用途的 Decoder-only Transformer，也就是 GPT 风格的自回归语言模型。
它使用 TinyStories 英文儿童故事数据，训练目标是：

> 给定前面的 token，预测下一个 token。

例如输入和标签会错开一个位置：

```text
输入 x: Once  upon  a  time
标签 y: upon  a     time ,
```

模型不是在一次前向传播中直接写出整篇故事，而是反复执行：

```text
已有文本 → 预测下一个 token → 拼接到已有文本 → 再预测下一个 token
```

---

## 2. 仓库结构

```text
MiniStoryGPT/
├── model.py                    # GPT 模型、注意力、生成、token 映射函数
├── prepare_data.py             # 下载 TinyStories、分词、压缩词表、保存数据
├── train.py                    # batch 采样、训练、验证、采样、保存 checkpoint
├── generate.py                 # 加载已有 checkpoint 生成故事
├── notebook.ipynb              # 教学和生成示例
├── requirements.txt            # Python 依赖
├── README.md                   # 原始英文说明
├── MODEL_ANALYSIS.md           # 已有的模型结构分析
├── MiniStoryGPT-30M.pth        # 已训练 checkpoint
└── data/
    ├── train.bin               # 训练 token，实际是 torch.save 保存的 Tensor
    ├── val.bin                 # 验证 token
    ├── old_to_new.pt           # GPT-2 token id → 项目 10K 词表 id
    └── new_to_old.pt           # 项目 10K 词表 id → GPT-2 token id
```

各脚本之间的调用关系：

```text
TinyStories
    │
    ▼
prepare_data.py
    │
    ├── data/train.bin
    ├── data/val.bin
    ├── data/old_to_new.pt
    └── data/new_to_old.pt
              │
              ▼
           train.py ─────────► checkpoint_iter*.pth / model_v1.pth
              │                                  │
              └──────── model.py                 ▼
                                             generate.py
```

---

## 3. 当前仓库的真实数据和模型规模

通过读取仓库文件得到：

| 项目 | 当前值 |
|---|---:|
| 训练 token 数 | 471,872,517 |
| 验证 token 数 | 4,743,928 |
| 训练数据逻辑大小 | 约 3.52 GiB |
| 数据类型 | `torch.int64` |
| 压缩后词表大小 | 10,000 |
| 最大上下文长度 | 512 |
| batch size | 32 |
| Transformer 层数 | 2 |
| 注意力头数 | 8 |
| 隐藏维度 | 768 |
| 单头维度 | 96 |
| 可训练参数 | 29,935,888，约 29.94M |
| mask buffer 元素 | 4,194,304，约 16 MiB FP32 |

现有 `MiniStoryGPT-30M.pth` 不是单纯的模型权重，而是一个完整训练 checkpoint：

```python
{
    "iter": 30000,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "loss": 1.7757824659347534,
}
```

所以当前 `generate.py` 对该文件的加载方式是正确的：

```python
checkpoint = torch.load("MiniStoryGPT-30M.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

但 `train.py` 训练结束后保存的 `model_v1.pth` 只是裸 `state_dict`：

```python
torch.save(model.state_dict(), model_file)
```

这两种格式不同，加载时不能混用。

---

## 4. 数据准备流程分析

### 4.1 下载数据

`prepare_data.py` 使用 Hugging Face Datasets：

```python
dataset = load_dataset("roneneldan/TinyStories")
```

首次运行会从网络下载 TinyStories，并写入 Hugging Face 本地缓存。

### 4.2 GPT-2 BPE 分词

```python
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
```

这里没有训练新的 tokenizer，而是直接复用 GPT-2 的 BPE tokenizer。
GPT-2 原始词表约 50K，但本项目只保留 TinyStories 中最常出现的 10K token。

### 4.3 压缩词表

```python
top_tokens = [
    token for token, count in token_counter.most_common(10000)
]
sorted_top_tokens = sorted(top_tokens)

old_to_new = {
    old_id: new_id
    for new_id, old_id in enumerate(sorted_top_tokens)
}
new_to_old = {
    new_id: old_id
    for old_id, new_id in old_to_new.items()
}
```

这里存在两套 id：

```text
GPT-2 原始 token id
        │ old_to_new
        ▼
MiniStoryGPT 的 0～9999
        │ new_to_old
        ▼
GPT-2 原始 token id → tiktoken.decode → 文本
```

压缩词表的直接好处是减少两块最大的参数：

- Token Embedding：`10000 × 768`
- LM Head：`768 × 10000`

### 4.4 重新编码并保存

```python
train_tokens = []

for text in dataset["train"]["text"]:
    original_tokens = enc.encode(text)
    remapped = remap_tokens(original_tokens)
    train_tokens.extend(remapped)

train_data = torch.tensor(train_tokens, dtype=torch.long)
torch.save(train_data, "data/train.bin")
```

虽然文件后缀是 `.bin`，它并不是 NumPy `memmap` 意义上的原始二进制文件，
而是由 `torch.save` 序列化的 PyTorch Tensor。

### 4.5 数据预处理中的关键问题

#### 问题一：预处理峰值内存很高

当前实现先把约 4.72 亿个 token 保存为 Python `int` 列表，再转换为
`torch.long`。Python 整数和列表指针的内存开销远大于 8 字节，
预处理过程中可能占用十几到几十 GiB 内存。

建议后续改成以下任一种方式：

- 分块编码并分块落盘；
- 使用 `numpy.memmap`；
- 直接使用 `uint16`，因为词表只有 10,000，小于 65,536；
- 使用 Hugging Face Dataset 的 batched `map`。

如果直接使用 `uint16`，当前训练集理论大小可以从约 3.52 GiB 降到约 0.88 GiB。

#### 问题二：未知 token 被错误映射为真实 token

```python
return [old_to_new.get(token, 0) for token in encoded_tokens]
```

这里把所有未进入 Top-10K 的 token 映射为新 id `0`。但当前新 id `0`
不是专门的 `<UNK>`，它对应 GPT-2 原始 token id `0`，也就是一个真实 token。
这会让“未知 token”和真实 token 共用同一个 embedding，并且解码时也可能产生错误字符。

更合理的方案是：

- 明确保留一个 `<UNK>` id；
- 或者不压缩 GPT-2 词表；
- 或者训练带 `<UNK>` 的新 tokenizer。

#### 问题三：故事之间没有边界 token

代码使用 `train_tokens.extend(remapped)` 直接连接所有故事，没有插入
`<EOS>` 或 `<|endoftext|>`。模型会把前一篇故事的末尾和下一篇故事的开头
当成连续文本。

推荐在每个故事末尾插入明确的结束 token：

```python
eot_id = enc.eot_token
story_tokens = enc.encode(text) + [eot_id]
```

但如果继续使用压缩词表，必须确保 `eot_id` 被保留并有独立的新 id。

#### 问题四：词表统计使用了验证集

当前 Top-10K 同时统计训练集和验证集。严格实验中，词表只应由训练集确定，
以避免验证集信息泄漏。对于教学项目影响不大，但应了解这一点。

---

## 5. 模型结构与关键代码注释

`model.py` 是这个项目的模型核心代码，但不是完整训练流程的全部：

- `model.py` 定义“模型是什么”，包括注意力、Transformer Block、前向传播、
  loss 计算、自回归生成和 token 映射辅助函数；
- `train.py` 定义“模型怎么训练”，包括 batch 采样、反向传播、验证和 checkpoint；
- `prepare_data.py` 定义“训练数据怎么制作”；
- `generate.py` 定义“怎样加载训练好的权重并生成故事”。

因此，如果目的是学习 GPT 的网络结构，应首先阅读 `model.py`；如果要重新训练，
则必须把 `model.py` 和 `train.py` 结合起来阅读。

`model.py` 的整体数据流可以概括为：

```text
token id
    │
    ▼
Token Embedding + Position Embedding
    │
    ▼
Transformer Block × 2
    ├── 8 头因果自注意力
    ├── LayerNorm + 残差连接
    └── 前馈网络
    │
    ▼
最终 LayerNorm
    │
    ▼
Linear（LM Head）
    │
    ▼
每个位置对下一个 token 的预测
```

### 5.1 超参数

```python
batch_size = 32     # 每个训练 step 并行处理 32 条序列
block_size = 512    # 每条序列最多 512 个 token
n_embd = 768        # 每个 token 的隐藏向量维度
n_head = 8          # 8 个注意力头
n_layer = 2         # 堆叠 2 个 Transformer Block
dropout = 0.2       # 训练时随机丢弃部分激活，抑制过拟合
```

每个头的维度为：

```text
head_size = n_embd / n_head = 768 / 8 = 96
```

建议增加合法性检查：

```python
assert n_embd % n_head == 0
```

### 5.2 单头因果自注意力

核心代码位于 `Head.forward`：

```python
def forward(self, x):
    # x: (B, T, C)
    # B = batch size
    # T = 序列长度
    # C = n_embd
    B, T, C = x.shape

    # 把同一个输入分别投影成 Key、Query、Value。
    # 每个张量形状都是 (B, T, head_size)。
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    # Query 与 Key 做点积，得到每个位置对其他位置的注意力分数。
    # q:              (B, T, hs)
    # k.transpose:    (B, hs, T)
    # wei:            (B, T, T)
    #
    # 除以 sqrt(head_size)，避免维度增大后点积绝对值过大，
    # 从而导致 softmax 饱和、梯度变小。
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

    # 下三角 mask：
    # 第 t 个位置只能看位置 0..t，不能看到未来 token。
    # 被遮挡的位置设为 -inf，softmax 后概率变为 0。
    wei = wei.masked_fill(
        self.tril[:T, :T] == 0,
        float("-inf"),
    )

    # 在最后一个维度归一化为概率。
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    # 用注意力概率对 Value 做加权求和。
    # 输出形状仍是 (B, T, head_size)。
    out = wei @ v
    return out
```

其中最核心的一行是：

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
```

它实现的是缩放点积注意力中的分数计算：

```text
scores = QKᵀ / sqrt(d_k)
```

这里得到的还不是最终注意力概率，而是每两个位置之间的匹配分数。
当前单个注意力头中：

```text
q:                         (B, T, 96)
k:                         (B, T, 96)
k.transpose(-2, -1):       (B, 96, T)
q @ k.transpose(-2, -1):   (B, T, T)
```

结果 `wei[b, i, j]` 表示第 `b` 条序列中，位置 `i` 的 Query 与位置 `j`
的 Key 的匹配程度：

```text
wei[b, i, j] = q[b, i] · k[b, j]
```

可以把三类向量直观地理解为：

- Query：当前位置想寻找什么信息；
- Key：每个位置提供什么类型的信息；
- Value：每个位置实际携带的内容。

例如处理：

```text
The dog ate food
```

当模型计算 `ate` 的表示时，`ate` 的 Query 会分别与 `The`、`dog`、`ate`
和 `food` 的 Key 做点积。分数越高，表示模型认为对应位置与当前计算越相关。
不过，由于是因果语言模型，随后的 mask 会禁止 `ate` 看到未来的 `food`。

乘以：

```python
k.shape[-1] ** -0.5
```

等价于：

```python
1 / sqrt(head_size)
```

本模型的 `head_size` 是 96，所以整个计算等价于：

```python
wei = (q @ k.transpose(-2, -1)) / math.sqrt(96)
```

如果不进行缩放，随着向量维度增大，点积绝对值通常也会变大，使后续
softmax 输出过于接近 0 或 1，容易进入饱和区域并导致梯度变小。
除以 `sqrt(head_size)` 可以把分数维持在更适合训练的尺度。

结合后面的 mask、softmax 和 Value 加权，完整算法是：

```text
Attention(Q, K, V)
    = softmax(QKᵀ / sqrt(d_k) + causal_mask) V
```

执行顺序为：

1. `QKᵀ` 计算每个位置与其他位置的相关性；
2. causal mask 遮住未来位置；
3. softmax 把分数转换成总和为 1 的概率；
4. 用这些概率对 Value 加权求和，得到当前位置汇总后的信息。

因果 mask 的直观形式：

```text
位置 0: [可看, 禁止, 禁止, 禁止]
位置 1: [可看, 可看, 禁止, 禁止]
位置 2: [可看, 可看, 可看, 禁止]
位置 3: [可看, 可看, 可看, 可看]
```

### 5.3 多头注意力

```python
def forward(self, x):
    # 每个头独立学习不同的 Q/K/V 投影。
    # 8 个头分别输出 (B, T, 96)。
    #
    # 在最后一维拼接后：
    # (B, T, 8 × 96) = (B, T, 768)
    out = torch.cat([h(x) for h in self.heads], dim=-1)

    # 混合不同头的信息，再投影回 n_embd。
    out = self.dropout(self.proj(out))
    return out
```

当前实现容易理解，但性能不高，因为每个头都单独执行三个 Linear。
生产级实现通常会用一个融合的 QKV Linear，或 PyTorch 的
`scaled_dot_product_attention`。

### 5.4 前馈网络

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),  # 768 → 3072，扩展通道
    nn.ReLU(),                      # 非线性变换
    nn.Linear(4 * n_embd, n_embd),  # 3072 → 768，压回原维度
    nn.Dropout(dropout),
)
```

注意力负责不同 token 之间的信息交互，前馈网络负责对每个位置的特征做独立变换。
现代 GPT 更常使用 GELU、SiLU 或 SwiGLU，而不是 ReLU。

### 5.5 Transformer Block

```python
def forward(self, x):
    # Pre-LayerNorm：
    # 先归一化，再做自注意力，最后和原输入形成残差连接。
    x = x + self.sa(self.ln1(x))

    # 再经过前馈网络和第二条残差连接。
    x = x + self.ffwd(self.ln2(x))
    return x
```

单个 Block 的数据流：

```text
                  ┌─────────────────┐
x ── LayerNorm ─► Attention ───────► + ──► x1
│                                     ▲
└─────────────────────────────────────┘

                   ┌──────────────┐
x1 ─ LayerNorm ──► FeedForward ──► + ──► x2
│                                    ▲
└────────────────────────────────────┘
```

残差连接让深层网络更容易传播梯度；Pre-LN 通常也比原始 Post-LN
Transformer 更容易稳定训练。

### 5.6 完整前向传播

```python
def forward(self, idx, targets=None):
    # idx 是 token id，形状 (B, T)。
    B, T = idx.shape

    # Token Embedding：把每个整数 token id 映射到 768 维向量。
    tok_emb = self.token_embedding_table(idx)  # (B, T, C)

    # Position Embedding：告诉模型每个 token 位于序列的哪个位置。
    positions = torch.arange(T, device=idx.device)
    pos_emb = self.position_embedding_table(positions)  # (T, C)

    # 内容信息与位置信息相加。
    x = tok_emb + pos_emb

    # 依次通过两个 Transformer Block。
    x = self.blocks(x)

    # 最后的 LayerNorm 和词表分类头。
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    if targets is None:
        return logits, None

    # CrossEntropyLoss 需要：
    # logits:  (样本数, 类别数)
    # targets: (样本数,)
    #
    # 因此把 batch 和时间维展平。
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    loss = F.cross_entropy(logits, targets)
    return logits, loss
```

当前 `model.py` 使用全局变量 `device` 创建 position tensor：

```python
torch.arange(T, device=device)
```

更稳妥的是使用输入张量所在设备：

```python
torch.arange(T, device=idx.device)
```

否则在手动把模型移动到非默认设备时，可能发生 device mismatch。

还应增加长度检查：

```python
if T > block_size:
    raise ValueError(
        f"Sequence length {T} exceeds block_size {block_size}"
    )
```

### 5.7 Loss 到底在优化什么

`F.cross_entropy` 等价于对每个位置计算真实下一个 token 的负对数似然，
再对 batch 和序列位置取平均：

```text
loss = -mean(log P(真实的下一个 token | 前文))
```

loss 越低，说明模型给真实下一个 token 分配的概率越高。

训练 loss 和验证 loss 的常见解释：

| 现象 | 可能含义 |
|---|---|
| train、val 都下降 | 正常学习 |
| train 下降，val 持平或上升 | 过拟合 |
| 二者长期不下降 | 学习率、数据、实现或梯度存在问题 |
| loss 突然变成 NaN | 学习率过高、数值溢出、数据异常 |

### 5.8 自回归生成

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # 模型最多只能读取最后 block_size 个 token。
        idx_cond = idx[:, -block_size:]

        # 计算序列中每个位置的词表 logits。
        logits, _ = self(idx_cond)

        # 生成下一个 token 时，只需要最后一个位置的预测。
        logits = logits[:, -1, :]

        # 转为概率并随机采样一个 token。
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 拼回序列，进入下一轮预测。
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

当前生成没有 `temperature`、`top_k`、`top_p` 或 repetition penalty。
因此结果可能过于随机、重复或跑题。最容易添加的是 temperature：

```python
logits = logits[:, -1, :] / temperature
```

- `temperature < 1`：更保守、更稳定；
- `temperature > 1`：更多样、也更容易混乱；
- 推荐先尝试 `0.7～0.9`。

---

## 6. 训练循环分析

### 6.1 构造输入和标签

```python
def get_batch(split):
    data = train_data if split == "train" else val_data

    # 随机选择 batch_size 个起始位置。
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
    )

    # 每个 x 是长度为 block_size 的连续片段。
    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    # y 比 x 向右移动一个 token。
    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)
```

默认每个训练 step 使用：

```text
32 × 512 = 16,384 token
```

所以：

| 训练步数 | 处理的 token 次数 | 约等于遍历训练集 |
|---:|---:|---:|
| 10,000 | 1.64 亿 | 0.35 epoch |
| 30,000 | 4.92 亿 | 1.04 epoch |
| 50,000 | 8.19 亿 | 1.74 epoch |
| 100,000 | 16.38 亿 | 3.47 epoch |

这里是随机窗口采样，所以“epoch”只是按 token 数量折算，不代表每个 token
恰好被访问一次。

### 6.2 单步训练

```python
xb, yb = get_batch("train")

# 前向传播并计算下一 token 预测 loss。
logits, loss = model(xb, yb)

# 清空上一轮梯度。
optimizer.zero_grad(set_to_none=True)

# 反向传播，计算每个参数的梯度。
loss.backward()

# AdamW 根据梯度更新模型参数。
optimizer.step()
```

### 6.3 验证

```python
@torch.no_grad()
def estimate_loss(model):
    model.eval()

    for split in ["train", "val"]:
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)

    model.train()
```

关键点：

- `torch.no_grad()`：不构建反向传播图，降低显存和计算开销；
- `model.eval()`：关闭 dropout；
- 完成评估后用 `model.train()` 恢复训练模式。

### 6.4 checkpoint

中间 checkpoint 保存：

```python
checkpoint = {
    "iter": iter,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}
torch.save(checkpoint, f"checkpoint_iter{iter}.pth")
```

要真正无缝续训，至少要保存：

- 当前 step；
- 模型权重；
- optimizer 状态；
- 学习率 scheduler 状态（如果使用）；
- AMP GradScaler 状态（如果使用 FP16）；
- 随机数状态（严格复现实验时）。

---

## 7. 开始训练前必须处理的问题

### 7.1 `eval_iters` 未定义，训练会在导入阶段失败

当前 `train.py`：

```python
from model import GPTLanguageModel, device, batch_size, block_size, eval_iters
```

但 `model.py` 没有 `eval_iters`。运行时会出现：

```text
ImportError: cannot import name 'eval_iters' from 'model'
```

最小修复：

```python
# train.py
from model import GPTLanguageModel, device, batch_size, block_size

eval_iters = 50
```

调试阶段可设为 `10`，正式训练可用 `50～200`。数值越大，验证 loss 越稳定，
但评估耗时越长。

### 7.2 不要把 3.52 GiB 训练数据整体加载到 GPU

当前代码：

```python
train_data = torch.load("data/train.bin", map_location=device)
val_data = torch.load("data/val.bin", map_location=device)
```

如果 `device == "cuda"`，会尝试把整个训练 Tensor 放进显存。
训练数据本身就占约 3.52 GiB，还没计算模型、梯度、AdamW 状态和激活，
很容易直接 OOM。

最低限度应把完整数据留在 CPU，只把 batch 移动到 GPU：

```python
train_data = torch.load(
    "data/train.bin",
    map_location="cpu",
    mmap=True,
)
val_data = torch.load(
    "data/val.bin",
    map_location="cpu",
    mmap=True,
)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
    )
    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])
    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])
    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )
```

`mmap=True` 需要较新的 PyTorch。若版本不支持，先去掉该参数，但仍应使用
`map_location="cpu"`。

### 7.3 checkpoint 文件名按字符串排序不可靠

当前逻辑：

```python
sorted(checkpoint_names, reverse=True)
```

这是字符串排序，不是数字排序。例如：

```text
checkpoint_iter5000.pth
checkpoint_iter10000.pth
```

字符串排序可能错误地把 `5000` 当作更大值。

建议按迭代数提取数字：

```python
import re
from pathlib import Path

def checkpoint_step(path):
    match = re.fullmatch(
        r"checkpoint_iter(\d+)\.pth",
        path.name,
    )
    return int(match.group(1)) if match else -1

checkpoint_files = list(Path(".").glob("checkpoint_iter*.pth"))
checkpoint_path = (
    max(checkpoint_files, key=checkpoint_step)
    if checkpoint_files
    else None
)
```

### 7.4 续训会重复已经完成的 step

checkpoint 在 `optimizer.step()` 后保存，恢复时应从下一步开始：

```python
start_iter = checkpoint["iter"] + 1
```

当前代码使用 `checkpoint["iter"]`，会重复一次已完成的迭代。

### 7.5 最终模型与 checkpoint 保存格式不统一

中间文件是：

```python
{"model_state_dict": ..., "optimizer_state_dict": ...}
```

最终文件却是：

```python
model.state_dict()
```

推荐统一保存格式：

```python
torch.save(
    {
        "iter": max_iters - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    },
    model_file,
)
```

或者在生成代码中同时兼容两种格式：

```python
loaded = torch.load(checkpoint_path, map_location=device)
state_dict = loaded.get("model_state_dict", loaded)
model.load_state_dict(state_dict)
```

### 7.6 现有 `MiniStoryGPT-30M.pth` 不会被自动续训

`train.py` 只搜索：

```text
checkpoint_iter*.pth
```

所以它不会自动读取 `MiniStoryGPT-30M.pth`。如果要从现有 30000 step
checkpoint 继续训练，应显式设置 checkpoint 路径，而不是依赖自动扫描：

```python
checkpoint_path = "MiniStoryGPT-30M.pth"
```

更好的做法是为 `train.py` 添加 `--resume` 命令行参数。

### 7.7 CSV 中应记录普通数字

当前 `losses_log` 存的是零维 Tensor：

```python
losses_log.append([
    iter,
    losses["train"],
    losses["val"],
])
```

建议改成：

```python
losses_log.append([
    iter,
    losses["train"].item(),
    losses["val"].item(),
])
```

### 7.8 建议固定随机种子

```python
import random
import numpy as np
import torch

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

这不能保证所有 GPU 算子都完全确定，但能显著提高实验可复现性。

---

## 8. 实际训练步骤

### 8.1 准备 Python 环境

建议使用 Python 3.10 或 3.11：

```bash
cd /Users/edy/Downloads/wangfei/MiniStoryGPT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果使用 NVIDIA GPU，应根据 CUDA 版本从 PyTorch 官方安装对应构建版本，
然后检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出最后一项为 `True` 才说明 PyTorch 能使用 CUDA。

### 8.2 选择使用已有数据还是重新预处理

当前仓库已经有：

```text
data/train.bin
data/val.bin
data/old_to_new.pt
data/new_to_old.pt
```

如果这些文件完整，可以直接跳过数据准备。

只有在以下情况才需要重新运行：

- 想更换数据集；
- 想修改词表大小；
- 想加入 `<UNK>` / `<EOS>`；
- 想修复故事之间没有边界的问题；
- 数据文件损坏或缺失。

重新准备：

```bash
python prepare_data.py
```

注意事项：

- 需要网络下载 TinyStories；
- 需要较大的磁盘空间；
- 当前实现还需要很大的 RAM；
- 运行时间可能较长；
- 会覆盖 `data/train.bin`、`data/val.bin` 和映射文件。

映射文件必须和训练数据、模型 checkpoint 成套使用。改变映射后，旧模型权重
不能再正确解释新 token id。

### 8.3 先做一次最小冒烟训练

不要一开始就跑 100000 step。先临时使用：

```python
# model.py
batch_size = 2
block_size = 128

# train.py
max_iters = 20
eval_interval = 10
eval_iters = 2
checkpoint_interval = 10
```

目标不是训练出好模型，而是确认：

- 数据能加载；
- 前向传播成功；
- loss 是有限数字；
- backward 和 optimizer.step 成功；
- checkpoint 能保存；
- checkpoint 能重新加载；
- 生成流程可运行。

注意：改变 `block_size` 会改变位置 embedding 和 mask 的形状，
因此无法直接加载使用 `block_size=512` 训练的旧 checkpoint。
若要测试旧 checkpoint，保持 `block_size=512`，只减小 `batch_size`。

### 8.4 从头训练

完成第 7 节的必要修复后：

```bash
python train.py
```

当前默认配置：

```text
batch_size       = 32
block_size       = 512
max_iters        = 100000
learning_rate    = 3e-4
eval_interval    = 500
checkpoint every = 5000 step
```

建议第一轮实验使用更小规模：

```text
batch_size       = 4 或 8
block_size       = 256 或 512
max_iters        = 5000
eval_interval    = 250
eval_iters       = 20
```

先观察 train/val loss 和生成样本，再决定是否继续。

### 8.5 从现有 checkpoint 继续训练

现有文件记录的是 step 30000，并且包含 optimizer 状态。
加载核心逻辑应为：

```python
checkpoint_path = "MiniStoryGPT-30M.pth"
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_iter = checkpoint["iter"] + 1
```

注意：

- 模型结构必须和保存时完全一致；
- `vocab_size` 必须仍为 10000；
- `old_to_new.pt` / `new_to_old.pt` 必须是同一套映射；
- `n_embd`、`n_head`、`n_layer`、`block_size` 不能改变；
- 若只改变 `max_iters`、日志频率，通常不影响加载；
- 若改变 batch size，可以加载，但训练轨迹会改变。

### 8.6 观察训练是否正常

重点观察：

```text
step 0: train loss ..., val loss ...
step 500: train loss ..., val loss ...
...
```

10K 类均匀随机猜测的理论交叉熵约为：

```text
ln(10000) ≈ 9.21
```

初始化后的实际 loss 应在这一数量级附近，并逐渐下降。
如果第一步就远高于此值、出现 NaN 或完全不变，应停止训练并排查。

每次验证时还会生成一段故事。生成质量通常比单独看 loss 更直观：

- 是否形成英文单词；
- 句法是否逐渐完整；
- 是否能结束句子；
- 是否反复重复同一句；
- 是否与 prompt 有关联。

### 8.7 测试生成

当前脚本把第一个命令行参数之后的所有内容拼成 prompt，所以可运行：

```bash
python generate.py Once upon a time,
```

或用引号：

```bash
python generate.py "Once upon a time,"
```

注意：README 中提到的：

```bash
python generate.py --checkpoint ...
```

当前代码并不支持。`generate.py` 会固定加载：

```text
MiniStoryGPT-30M.pth
```

如需选择文件，应使用 `argparse` 增加 `--checkpoint` 参数。

---

## 9. 设备和显存建议

### 9.1 CPU

可以用于：

- 导入和单元测试；
- 小 batch 冒烟运行；
- 加载 checkpoint；
- 少量 token 生成。

不适合用当前默认配置完整训练 30M 模型，速度会非常慢。

### 9.2 Apple Silicon

当前代码只检测 CUDA：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

不会自动使用 MPS。可改为：

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

但需要测试当前 PyTorch 版本对所用算子的 MPS 支持和稳定性。

### 9.3 NVIDIA GPU

推荐先使用 16～24 GB 显存的 GPU。原因是训练显存不只包含模型参数，还包括：

- 参数；
- 梯度；
- AdamW 的一阶、二阶状态；
- 前向激活；
- 注意力矩阵；
- `B × T × vocab_size` 的 logits；
- CUDA workspace。

尤其当前 logits 形状为：

```text
32 × 512 × 10000 = 163,840,000 个元素
```

仅 FP32 logits 就约 625 MiB，反向传播还需要更多中间状态。

如果只有 8 GB 显存：

- 先把数据保留在 CPU；
- `batch_size` 从 2 或 4 开始；
- 使用混合精度；
- 使用梯度累积维持有效 batch；
- 必要时把 `block_size` 降到 256；
- 可以考虑把 `n_embd` 降到 384 或 512，但这将无法加载旧 checkpoint。

---

## 10. 建议的训练增强

以下不是“能运行”的必要条件，但能明显改善训练效率和稳定性。

### 10.1 自动混合精度

在支持的 NVIDIA GPU 上可使用 BF16 或 FP16，降低激活显存并提升吞吐。
Ampere 或更新架构通常优先尝试 BF16。

PyTorch 2.x 示例思路：

```python
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

with torch.autocast(
    device_type="cuda",
    dtype=amp_dtype,
):
    logits, loss = model(xb, yb)
```

如果使用 FP16，通常还需 `GradScaler`；BF16 一般不需要。

### 10.2 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

放在 `loss.backward()` 之后、`optimizer.step()` 之前，用于降低梯度爆炸风险。

### 10.3 学习率调度

当前学习率始终为 `3e-4`。更常见的做法：

```text
短暂 warmup → cosine decay → 最小学习率
```

从已有 checkpoint 继续训练时，scheduler 状态也要一起保存和恢复。

### 10.4 梯度累积

假设显存只能容纳 `batch_size=4`，但希望有效 batch 仍相当于 32：

```text
gradient_accumulation_steps = 8
有效 batch = 4 × 8 = 32
```

每次小 batch 的 loss 要除以累积步数，在第 8 次 backward 后再执行
`optimizer.step()`。

### 10.5 权重共享

当前输入 embedding 和输出 lm head 各自维护一份 `10000 × 768` 权重。
可共享：

```python
self.lm_head.weight = self.token_embedding_table.weight
```

这会减少约 768 万参数，但会改变模型结构，不能直接加载当前 checkpoint。

### 10.6 更高效的注意力实现

当前每个 Head 都保存一份 `512 × 512` mask：

```text
2 层 × 8 头 = 16 份重复 mask
```

这些 mask 总计约 16 MiB，还会进入 `state_dict`。可以改为：

- 每层或整个模型共享一个 bool mask；
- mask buffer 设置 `persistent=False`；
- 使用 `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`。

### 10.7 减少验证和生成开销

每 500 step：

- train loss 评估 `eval_iters` 次；
- val loss 评估 `eval_iters` 次；
- 逐 token 生成 300 次前向传播。

如果训练耗时主要卡在评估，可：

- 减小 `eval_iters`；
- 增大 `eval_interval`；
- 把 `sample_interval` 设置得比 `eval_interval` 更大；
- 生成更少 token。

---

## 11. 当前代码风险清单

| 优先级 | 问题 | 影响 |
|---|---|---|
| 必修复 | `eval_iters` 未定义 | `train.py` 无法导入 |
| 必修复 | 整个训练集按 `device` 加载 | CUDA 显存很容易 OOM |
| 高 | checkpoint 按字符串排序 | 可能恢复到错误的旧 checkpoint |
| 高 | 现有 30K checkpoint 不会自动恢复 | 实际会从头训练 |
| 高 | 最终模型与中间 checkpoint 格式不统一 | 加载方式容易报错 |
| 中 | resume 从同一个 step 开始 | 重复一次训练 step |
| 中 | 未知 token 与真实 id 0 冲突 | 数据语义错误 |
| 中 | 故事之间没有 EOS | 模型学习跨故事连接 |
| 中 | 预处理使用超大 Python list | RAM 峰值很高 |
| 中 | `device` 是模型全局变量 | 手动移动设备时可能不一致 |
| 中 | 不支持自动选择 MPS | Apple GPU 不会被使用 |
| 中 | 无 AMP、梯度裁剪、scheduler | 效率和稳定性较弱 |
| 低 | 每个注意力头重复保存 mask | 增加内存和 checkpoint 体积 |
| 低 | README 的生成命令与代码不一致 | `--checkpoint` 实际无效 |
| 低 | requirements 未锁定上限或精确版本 | 不同环境复现性较弱 |

---

## 12. 推荐实施顺序

如果目标是尽快训练成功，建议按下面顺序推进：

1. 保留现有 `data/`，暂时不要重新预处理 4.72 亿 token。
2. 修复 `eval_iters`。
3. 把完整数据固定留在 CPU，并优先使用 `mmap=True`。
4. 增加明确的 `--resume` / `--checkpoint` 参数。
5. 统一 checkpoint 保存格式。
6. 使用小 batch 做 20 step 冒烟训练。
7. 验证保存、恢复和生成全部成功。
8. 根据 GPU 显存逐步增加 batch size。
9. 开启 BF16/FP16、梯度裁剪和学习率调度。
10. 先训练 1000～5000 step 观察曲线，再决定是否跑到 50000 或 100000。
11. 下一轮实验再重构数据格式、加入 EOS 和正确的 UNK。

---

## 13. 一次完整实验应记录什么

为了让模型结果可复现，每次训练建议记录：

```text
实验名称 / version
Git commit
随机种子
数据集版本
tokenizer 和 token 映射哈希
训练 / 验证 token 数
vocab_size
batch_size
block_size
gradient accumulation
n_embd / n_head / n_layer
dropout
optimizer 参数
学习率和 scheduler
训练步数
混合精度类型
GPU 型号
训练耗时
最佳 train / val loss
checkpoint 路径
固定 prompt 的生成样本
```

尤其要保存 token 映射的版本。对这个项目来说，同一个数字 token id 在不同映射中
可能代表完全不同的文本，映射丢失后模型基本无法正确使用。

---

## 14. 总结

这个项目的优点是代码短、模型链路完整，很适合理解：

- GPT-2 BPE token 如何进入模型；
- 因果自注意力为什么不能看未来；
- 多头注意力、前馈网络、LayerNorm、残差怎样组成 GPT Block；
- 下一 token loss 怎样计算；
- 训练 checkpoint 和自回归生成怎样衔接。

当前模型有约 29.94M 可训练参数，训练集约 4.72 亿 token，具备继续训练的基础。
但在正式开跑之前，应优先修复训练入口和数据加载问题。最关键的两点是：

```text
1. 在 train.py 中定义 eval_iters；
2. 不要把 3.52 GiB 的完整训练 Tensor 放到 GPU，只移动当前 batch。
```

修复后先完成小规模冒烟训练，再开始长时间训练，会比直接运行默认的
100000 step 更安全，也更容易定位问题。
