import torch
import torch.nn as nn

from multi_head_attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)#定义了一个函数
        self.attention = MultiHeadSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        #print("Input x shape:", x.shape)

        # =========================
        # 1. LayerNorm + Attention
        # =========================

        norm_x = self.ln1(x)
        #print("After ln1 shape:", norm_x.shape)

        attn_out, attention_weights = self.attention(norm_x, mask=mask)
        #print("Attention output shape:", attn_out.shape)

        # Residual connection
        x = x + attn_out
        #print("After attention residual shape:", x.shape)

        # =========================
        # 2. LayerNorm + FFN
        # =========================

        norm_x = self.ln2(x)
        #print("After ln2 shape:", norm_x.shape)

        ffn_out = self.ffn(norm_x)
        #print("FFN output shape:", ffn_out.shape)

        # Residual connection
        x = x + ffn_out
        #print("Final block output shape:", x.shape)

        return x, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    hidden_dim = 32

    x = torch.randn(batch_size, seq_len, d_model)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim
    )

    output, attention_weights = block(x, mask=causal_mask)

    #print("\nFinal result:")
    #print("output shape:", output.shape)
    #print("attention_weights shape:", attention_weights.shape)

    #print("\nAttention weights of batch 0, head 0:")
    #print(attention_weights[0, 0])