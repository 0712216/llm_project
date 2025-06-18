import torch
from model.decoder import MiniDecoder
from tokenizer.tokenizer import CharTokenizer
import json

# --- 載入 config 與 tokenizer ---
with open("config.json", "r") as f:
    config = json.load(f)
tokenizer = CharTokenizer.from_vocab_file("checkpoints/tokenizer.pkl")
config["vocab_size"] = tokenizer.vocab_size

# --- 建立模型 ---
model = MiniDecoder(**config)
model.load_state_dict(torch.load("checkpoints/decoder_model.pt", map_location=torch.device("cpu")))
model.eval()

# --- Top-k & Top-p 工具函式 ---
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_indices, sorted_mask)
    logits[mask] = -float('Inf')
    return logits

# --- 生成文字的函式 ---
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, strategy="top_k", k=10, p=0.9):
    device = next(model.parameters()).device
    model.eval()
    idx = tokenizer.encode(prompt).unsqueeze(0).to(device)
    block_size = config["block_size"]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if strategy == "greedy":
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        elif strategy == "top_k":
            logits = top_k_logits(logits, k)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        elif strategy == "top_p":
            logits = top_p_logits(logits, p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            raise ValueError("Unknown sampling strategy")

        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0])

# --- 推論展示 ---
print("\n=== Sample Generation ===")
prompt = "To be or not "
sample = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, strategy="top_k", k=20)
print(sample)
