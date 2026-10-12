import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Hyperparameters (Target: ~20M params)
batch_size = 32  # Parallel sequences
block_size = 512  # Max context length
n_embd = 768 # Embedding dimension
n_head = 8    # Number of attention heads
n_layer = 2   # Number of Transformer blocks
dropout = 0.2
use_sinusoidal = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    """ Single head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v     # (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multi-head attention (paper Section 3.2.2) """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Position-wise feed-forward (paper Eq 2) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block (paper Section 3.1: self-attn + FFN with residuals/norm) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not use_sinusoidal:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        if use_sinusoidal:
            # Sinusoidal positions (paper Section 3.5)
            pos = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, n_embd, 2, device=device) * -(math.log(1e4) / n_embd))
            pe = torch.zeros(T, n_embd, device=device)
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            pos_emb = pe.unsqueeze(0).expand(B, -1, -1)  # (B,T,C)
        else:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B,T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)     # (B,T,C)
        x = self.ln_f(x)       # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def remap_tokens(encoded_tokens, old_to_new):
    return [old_to_new.get(token, 0) for token in encoded_tokens]  # 0 as UNK

def custom_decode(new_tokens_list, new_to_old, enc):
    original_tokens = [new_to_old.get(new_id, 0) for new_id in new_tokens_list]  # Map back, 0 as UNK
    return enc.decode(original_tokens)