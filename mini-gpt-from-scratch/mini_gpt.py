import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, hidden_dim, num_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # 1. token embedding：把 token id 变成向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. position embedding：给每个位置一个向量
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 3. 堆叠多个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                hidden_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])

        # 4. 最后的 LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # 5. LM Head：把 hidden state 映射到词表大小
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, targets=None):
        # input_ids shape: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape

        #print("input_ids shape:", input_ids.shape)

        # =========================
        # 1. Token Embedding
        # =========================

        token_emb = self.token_embedding(input_ids)
        #print("token_emb shape:", token_emb.shape)

        # =========================
        # 2. Position Embedding
        # =========================

        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(positions)

        #print("positions shape:", positions.shape)
        #print("pos_emb shape:", pos_emb.shape)

        # token_emb: [B, T, C]
        # pos_emb:   [T, C]
        # PyTorch 会自动广播成 [B, T, C]
        x = token_emb + pos_emb

        #print("x after embedding shape:", x.shape)

        # =========================
        # 3. Causal Mask
        # =========================

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

        #print("causal_mask shape:", causal_mask.shape)

        # =========================
        # 4. Transformer Blocks
        # =========================

        attention_weights_list = []

        for i, block in enumerate(self.blocks):
            #print(f"\n===== Transformer Block {i} =====")
            x, attention_weights = block(x, mask=causal_mask)
            attention_weights_list.append(attention_weights)

        # =========================
        # 5. Final LayerNorm
        # =========================

        x = self.ln_f(x)
        #print("\nafter final LayerNorm shape:", x.shape)

        # =========================
        # 6. LM Head
        # =========================

        logits = self.lm_head(x)
        #print("logits shape:", logits.shape)
        #logits对于每个位置，模型预测“下一个字符”是词表中每个字符的分数

        # =========================
        # 7. 如果给了 targets，就计算 loss
        # =========================

        loss = None

        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()

            # logits:  [B, T, vocab_size]
            # targets: [B, T]
            # CrossEntropyLoss 需要：
            # logits reshape 成 [B*T, vocab_size]
            # targets reshape 成 [B*T]

            loss = loss_fn(
                logits.view(batch_size * seq_len, self.vocab_size),
                targets.view(batch_size * seq_len)
            )

            #print("loss:", loss.item())

        return logits, loss, attention_weights_list


if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 100
    max_seq_len = 16
    batch_size = 2
    seq_len = 6

    d_model = 8
    num_heads = 2
    hidden_dim = 32
    num_layers = 2

    model = MiniGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    # 模拟输入 token id
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 模拟训练目标
    # 这里先随便生成一个 targets，只是为了跑通 loss
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    #print("input_ids:")
    #print(input_ids)

    logits, loss, attention_weights_list = model(input_ids, targets=targets)

    #print("\nFinal result:")
    #print("logits shape:", logits.shape)
    #print("loss:", loss.item())
    #print("number of attention layers:", len(attention_weights_list))