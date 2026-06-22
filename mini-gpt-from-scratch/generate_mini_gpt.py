import torch
from mini_gpt import MiniGPT


checkpoint = torch.load("mini_gpt_checkpoint.pt", map_location="cpu")

vocab_size = checkpoint["vocab_size"]
max_seq_len = checkpoint["max_seq_len"]
d_model = checkpoint["d_model"]
num_heads = checkpoint["num_heads"]
hidden_dim = checkpoint["hidden_dim"]
num_layers = checkpoint["num_layers"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = MiniGPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@torch.no_grad()
def generate(start_text, max_new_tokens=100, mode="argmax"):
    input_ids = torch.tensor([[stoi[ch] for ch in start_text]], dtype=torch.long)

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -max_seq_len:]

        logits, _, _ = model(input_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        if mode == "argmax":
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

    result = "".join([itos[int(i)] for i in input_ids[0]])

    return result


while True:
    start_text = input("请输入起始字符或字符串：")

    if start_text.lower() in ["exit", "quit", "退出"]:
        print("已退出。")
        break

    # 检查输入字符是否在词表中
    valid = True
    for ch in start_text:
        if ch not in stoi:
            print(f"字符 {ch} 不在训练词表中，请重新输入。")
            valid = False
            break

    if not valid:
        continue

    print("\nargmax 生成结果：")
    print(generate(start_text, max_new_tokens=100, mode="argmax"))

    print("\n随机采样生成结果：")
    print(generate(start_text, max_new_tokens=100, mode="sample"))