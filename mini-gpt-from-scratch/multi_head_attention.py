import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q、K、V 三个线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 多个 head 合并后的输出投影
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        #print("Input x shape:", x.shape)

        # 1. 计算 Q、K、V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        #print("Q before split:", Q.shape)
        #print("K before split:", K.shape)
        #print("V before split:", V.shape)

        # 2. 拆成多个 head
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        #print("Q after split:", Q.shape)
        #print("K after split:", K.shape)
        #print("V after split:", V.shape)

        # 3. 计算 attention score
        # Q: [B, H, T, D]
        # K.transpose(-2, -1): [B, H, D, T]
        # scores: [B, H, T, T]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        #print("Attention scores shape:", scores.shape)

        # 4. 加 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. softmax
        attention_weights = torch.softmax(scores, dim=-1)

        #print("Attention weights shape:", attention_weights.shape)

        # 6. attention_weights 乘 V
        # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
        head_output = attention_weights @ V

        #print("Head output shape:", head_output.shape)

        # 7. 合并多个 head
        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        output = head_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        #print("Output before o_proj:", output.shape)

        # 8. 输出投影
        output = self.o_proj(output)

        #print("Final output shape:", output.shape)

        return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

    output, attention_weights = attention(x, mask=causal_mask)

    #print("output shape:", output.shape)
    #print("attention_weights shape:", attention_weights.shape)

    #print("causal_mask:")
    #print(causal_mask[0, 0])

    #print("attention weights of batch 0, head 0:")
    #print(attention_weights[0, 0])