# ── Self-Attention / Transformer Block ───────────────
"""Self-Attention and Transformer Block.

Use F.scaled_dot_product_attention (Flash/Memory-efficient attention where
available): much faster and uses less memory than the manual version.

"""
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


class SelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, max_seq_length=147):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.unifyheads = nn.Linear(d_model, d_model, bias=False)

        causal = torch.tril(torch.ones(max_seq_length, max_seq_length,
                                       dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x, padding_mask=None):
        B, T, D = x.size()

        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))

        # combined mask: True = trusted location
        attn_mask = self.causal_mask[:T, :T]              # (T, T)
        if padding_mask is not None:
            keep = padding_mask.bool()                     # (B, T)
            attn_mask = attn_mask.unsqueeze(0) & keep.unsqueeze(1)  # (B, T, T)
            # the diagonal is always reliable: avoid all-False rows
            # (padding queries) that produce NaNs in the softmax    
            eye = torch.eye(T, dtype=torch.bool, device=x.device)
            attn_mask = attn_mask | eye.unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(1)             # (B, 1, T, T)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.0, max_seq_length=147):
        super().__init__()
        self.self_attention = SelfAttention(d_model, n_heads, max_seq_length)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.dropout(self.self_attention(self.layer_norm1(x),
                                                 padding_mask))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
