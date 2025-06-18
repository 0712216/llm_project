import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim).transpose(1, 2)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, block_size, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, block_size) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# Optional: Load config if running directly
if __name__ == "__main__":
    from tokenizer.tokenizer import CharTokenizer
    tokenizer = CharTokenizer.from_vocab_file("checkpoints/tokenizer.pkl")
    with open("config.json", "r") as f:
        config = json.load(f)
    config["vocab_size"] = tokenizer.vocab_size
    model = MiniDecoder(**config)
    print(model)
