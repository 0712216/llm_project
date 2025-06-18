import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
from dataset import CharDataset
from model.decoder import MiniDecoder
from tokenizer.bpe_tokenizer import BPETokenizer

# --- 自動訓練 tokenizer（若尚未存在） ---
tokenizer_path = "checkpoints/bpe_tokenizer.json"
if not os.path.exists(tokenizer_path):
    print("Tokenizer not found. Training BPE tokenizer...")
    BPETokenizer.train_from_text(["tiny_shakespeare.txt"], tokenizer_path=tokenizer_path)

# --- 載入 tokenizer ---
tokenizer = BPETokenizer(tokenizer_path)

# --- 載入或建立 config ---
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        config = json.load(f)
else:
    config = {
        "embed_dim": 128,
        "num_heads": 4,
        "block_size": 64,
        "num_layers": 2
    }

config["vocab_size"] = tokenizer.vocab_size

# --- 儲存 config（確保同步） ---
os.makedirs("checkpoints", exist_ok=True)
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# --- 準備資料集與 DataLoader ---
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

dataset = CharDataset(tokenizer, text, config["block_size"])
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- 建立模型與訓練元件 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniDecoder(**config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- 訓練迴圈 ---
epochs = 5
model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# --- 自動產生時間戳檔名 ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"checkpoints/decoder_model_bpe_{ts}.pt"
tokenizer_out_path = f"checkpoints/bpe_tokenizer_{ts}.json"

# --- 儲存模型與 tokenizer ---
torch.save(model.state_dict(), model_path)
tokenizer.tokenizer.save(tokenizer_out_path)
print(f"\n✅ 模型儲存完成：{model_path}")
print(f"✅ Tokenizer 儲存完成：{tokenizer_out_path}")
