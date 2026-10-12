"""用超小尺寸演示单个注意力头里的 q / k / v / wei。

公式与 model.py 的 Head.forward 完全一致，只是把维度缩小到可以整表打印：
    q, k, v = Linear(x)
    scores  = q @ k^T / sqrt(hs)
    wei     = softmax(causal_mask(scores))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


# 教学尺寸：1 条样本、4 个 token、4 维词向量、3 维注意力头
B = 1
T = 4
n_embd = 4
head_size = 3
tokens = ["小", "猫", "吃", "鱼"]


def print_matrix(title: str, tensor: torch.Tensor, row_labels: list[str], col_labels: list[str]) -> None:
    matrix = tensor.detach().cpu()
    if matrix.ndim == 3:
        matrix = matrix[0]
    print(f"\n{title}")
    print(f"shape = {tuple(tensor.shape)}")
    header = "        " + "".join(f"{label:>10}" for label in col_labels)
    print(header)
    for row_label, row in zip(row_labels, matrix.tolist()):
        print(f"{row_label:>6}  " + "".join(f"{value:10.4f}" for value in row))


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)

    print("=" * 72)
    print("单头注意力：q / k / v / wei 四个矩阵")
    print("=" * 72)
    print(f"句子: {''.join(tokens)}")
    print(f"token: {tokens}")
    print(f"B={B}, T={T}, n_embd={n_embd}, head_size={head_size}")

    # 为了演示可读，这里用手写的 4 个 token 向量，而不是随机 embedding。
    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.2],  # 小
                [0.0, 1.0, 0.0, 0.2],  # 猫
                [0.0, 0.0, 1.0, 0.2],  # 吃
                [0.2, 0.2, 0.2, 1.0],  # 鱼
            ]
        ],
        dtype=torch.float32,
    )
    print_matrix("输入 x  (B,T,n_embd) = (1,4,4)", x, tokens, ["e0", "e1", "e2", "e3"])

    key = nn.Linear(n_embd, head_size, bias=False)
    query = nn.Linear(n_embd, head_size, bias=False)
    value = nn.Linear(n_embd, head_size, bias=False)

    print("\n三个线性层的权重，形状都是 (head_size, n_embd) = (3, 4)")
    print_matrix("W_Q = query.weight", query.weight, ["q0", "q1", "q2"], ["e0", "e1", "e2", "e3"])
    print_matrix("W_K = key.weight", key.weight, ["k0", "k1", "k2"], ["e0", "e1", "e2", "e3"])
    print_matrix("W_V = value.weight", value.weight, ["v0", "v1", "v2"], ["e0", "e1", "e2", "e3"])

    q = query(x)  # (B,T,hs)
    k = key(x)    # (B,T,hs)
    v = value(x)  # (B,T,hs)
    print_matrix("query q = x @ W_Q^T   (B,T,hs) = (1,4,3)", q, tokens, ["q0", "q1", "q2"])
    print_matrix("key   k = x @ W_K^T   (B,T,hs) = (1,4,3)", k, tokens, ["k0", "k1", "k2"])
    print_matrix("value v = x @ W_V^T   (B,T,hs) = (1,4,3)", v, tokens, ["v0", "v1", "v2"])

    scale = head_size ** -0.5
    scores = q @ k.transpose(-2, -1) * scale  # (B,T,T)
    tril = torch.tril(torch.ones(T, T))
    masked_scores = scores.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(masked_scores, dim=-1)  # (B,T,T)

    print_matrix(
        f"原始分数 scores = q @ k^T / sqrt({head_size})   (B,T,T) = (1,4,4)",
        scores,
        tokens,
        tokens,
    )
    print_matrix("因果 mask 后的 scores（未来位置是 -inf）", masked_scores, tokens, tokens)
    print_matrix("注意力权重 wei = softmax(masked_scores)   (B,T,T) = (1,4,4)", wei, tokens, tokens)

    print("\nwei 每一行的含义：当前 token（行）对前文 token（列）的关注比例，行和为 1。")
    print("右上角全是 0，因为因果 mask 禁止看未来。")
    for i, token in enumerate(tokens):
        weights = wei[0, i].tolist()
        parts = [f"{other}={weight:.4f}" for other, weight in zip(tokens, weights)]
        print(f"  位置 {i} 的「{token}」看向: {', '.join(parts)}  | 行和={sum(weights):.4f}")

    out = wei @ v  # (B,T,hs)
    print_matrix("注意力输出 out = wei @ v   (B,T,hs) = (1,4,3)", out, tokens, ["h0", "h1", "h2"])

    print("\n验算：位置 3「鱼」的输出 = 四个 value 的加权和")
    print("out[鱼] =", (wei[0, 3:4] @ v[0]).squeeze(0).tolist())


if __name__ == "__main__":
    main()
