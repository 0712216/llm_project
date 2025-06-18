import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.decoder import MiniDecoder
from dataset import CharDataset
from tqdm import tqdm
from tokenizer.tokenizer import CharTokenizer
import json

# --- 超參數與資料設定 ---
dataTXT = "data/tiny_shakespeare.txt"
with open(dataTXT, "r", encoding="utf-8") as f:
    text = f.read()

# 載入 tokenizer
tokenizer = CharTokenizer.from_text(text)
tokenizer.save("checkpoints/tokenizer.pkl")

# 載入 config 並補上 vocab_size
with open("config.json", "r") as f:
    config = json.load(f)
config["vocab_size"] = tokenizer.vocab_size

# 訓練參數
num_epochs = 10
learning_rate = 1e-3
batch_size = 32
block_size = config["block_size"]

# 建立資料與模型
dataset = CharDataset(tokenizer, text, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = MiniDecoder(**config)

# --- 訓練流程 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits.view(-1, config["vocab_size"]), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# --- 儲存模型 ---
torch.save(model.state_dict(), "checkpoints/decoder_model.pt")

# --- 簡易生成展示 ---
@torch.no_grad()
def generate(model, tokenizer, prompt, block_size, max_new_tokens=200, temperature=1.0):
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

print("\n=== Sample Generation ===")
prompt = "To be or not "
sample = generate(model, tokenizer, prompt, block_size, max_new_tokens=200, temperature=0.8)
print(sample)
