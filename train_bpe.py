import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
from tqdm import tqdm
from dataset import CharDataset
from model.decoder import MiniDecoder
from tokenizer.bpe_tokenizer import BPETokenizer

# --- 自動訓練 tokenizer（若尚未存在） ---
tokenizer_path = "checkpoints/bpe_tokenizer.json"
if not os.path.exists(tokenizer_path):
    print("Tokenizer not found. Training BPE tokenizer...")
    BPETokenizer.train_from_text(["data/tiny_shakespeare.txt"], tokenizer_path=tokenizer_path)

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
batch_size = config.get("batch_size", 64)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 建立模型與訓練元件 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniDecoder(
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    block_size=config["block_size"],
    num_layers=config["num_layers"],
    vocab_size=config["vocab_size"]
).to(device)
learning_rate = config.get("learning_rate", 1e-3)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# --- 設定 Learning Rate Scheduler（由 config 控制） ---
scheduler_config = config.get("scheduler", {})
scheduler_type = scheduler_config.get("type", None)
scheduler = None

if scheduler_type == "step":
    from torch.optim.lr_scheduler import StepLR
    step_size = scheduler_config.get("step_size", 2)
    gamma = scheduler_config.get("gamma", 0.5)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
elif scheduler_type == "cosine":
    from torch.optim.lr_scheduler import CosineAnnealingLR
    T_max = scheduler_config.get("T_max", config.get("epochs", 5))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
elif scheduler_type == "plateau":
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    factor = scheduler_config.get("factor", 0.1)
    patience = scheduler_config.get("patience", 2)
    scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

# 可選 scheduler 設定樣板：
# "scheduler": {
#   "type": "step",
#   "step_size": 2,
#   "gamma": 0.5
# }
# "scheduler": {
#   "type": "cosine",
#   "T_max": 10
# }
# "scheduler": {
#   "type": "plateau",
#   "factor": 0.5,
#   "patience": 2
# }

# --- 訓練迴圈 ---
epochs = config.get("epochs", 5)
model.train()
for epoch in range(epochs):
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}")

    if scheduler_type == "plateau":
        scheduler.step(avg_loss)
    elif scheduler:
        scheduler.step()

# --- 自動產生時間戳檔名 ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"checkpoints/decoder_model_bpe_{ts}.pt"
tokenizer_out_path = f"checkpoints/bpe_tokenizer_{ts}.json"

# --- 儲存模型與 tokenizer ---
torch.save(model.state_dict(), model_path)
tokenizer.tokenizer.save(tokenizer_out_path)
print(f"\n✅ 模型儲存完成：{model_path}")
print(f"✅ Tokenizer 儲存完成：{tokenizer_out_path}")

# --- 若需要推送到 GitHub，請於 Colab notebook 內執行 ---
# push 指令請獨立寫於 notebook 內以提高控制彈性與安全性
