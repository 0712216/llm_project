import torch
from tokenizer.tokenizer import CharTokenizer
from model.decoder import MiniDecoder
import json

@torch.no_grad()
def generate(model, tokenizer, prompt, block_size=64, max_new_tokens=200, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    idx = tokenizer.encode(prompt).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0])

# --- 測試 Prompt 設定 ---
test_prompts = [
    "ROMEO:",
    "JULIET:",
    "My love is like a",
    "Who art thou that",
    "Thou shalt not",
    "I am the very",
    "What light through yonder",
    "To be or not to be"
]

# --- 模型與 tokenizer 載入 ---
tokenizer = CharTokenizer.from_vocab_file("checkpoints/tokenizer.pkl")
with open("config.json", "r") as f:
    config = json.load(f)
config["vocab_size"] = tokenizer.vocab_size
model = MiniDecoder(**config)
model.load_state_dict(torch.load("checkpoints/decoder_model.pt", map_location=torch.device("cpu")))

# --- 分析輸出 ---
print("\n=== Model Understanding Test ===")
for i, prompt in enumerate(test_prompts):
    print(f"\n[{i+1}] Prompt: {prompt}")
    out = generate(model, tokenizer, prompt=prompt, max_new_tokens=100, temperature=0.8)
    print(out)

print("\n=== End of Test ===")
