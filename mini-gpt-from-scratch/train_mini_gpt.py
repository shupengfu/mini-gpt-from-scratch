import torch
from mini_gpt import MiniGPT


# =========================
# 1. 准备一个极小文本数据集
# =========================

text = (
    "transformer is based on self attention. "
    "attention lets each token look at previous tokens. "
    "gpt predicts the next token one by one. "
) * 100

# 构建字符级词表
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# 把文本转成 token id
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

print("vocab_size:", vocab_size)
print("data length:", len(data))


# =========================
# 2. 构造训练 batch
# =========================

batch_size = 16
seq_len = 16


def get_batch():
    # 随机选 batch_size 个起点
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))

    # x 是当前 token 序列
    x = torch.stack([data[i:i + seq_len] for i in ix])

    # y 是下一个 token
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])

    return x, y


# =========================
# 3. 创建 MiniGPT 模型
# =========================

model = MiniGPT(
    vocab_size=vocab_size,
    max_seq_len=seq_len,
    d_model=64,
    num_heads=4,
    hidden_dim=256,
    num_layers=2
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# =========================
# 4. 训练循环
# =========================

num_steps = 1000

for step in range(num_steps):
    x, y = get_batch()

    logits, loss, _ = model(x, targets=y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "max_seq_len": seq_len,
    "d_model": 64,
    "num_heads": 4,
    "hidden_dim": 256,
    "num_layers": 2,
    "stoi": stoi,
    "itos": itos,
}

torch.save(checkpoint, "mini_gpt_checkpoint.pt")

print("模型已保存到 mini_gpt_checkpoint.pt")

# =========================
# 5. 生成文本
# =========================

@torch.no_grad()
def generate(model, start_text, max_new_tokens=100):
    model.eval()

    input_ids = torch.tensor([[stoi[ch] for ch in start_text]], dtype=torch.long)

    for _ in range(max_new_tokens):
        # 只取最后 seq_len 个 token，避免超过最大长度
        input_cond = input_ids[:, -seq_len:]

        logits, _, _ = model(input_cond)

        # 取最后一个位置的 logits
        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

    result = "".join([itos[int(i)] for i in input_ids[0]])

    return result


print("\nGenerated text:")
print(generate(model, start_text="g", max_new_tokens=100))

