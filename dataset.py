import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, tokenizer, text, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.encoded = tokenizer.encode(text)

    def __len__(self):
        return len(self.encoded) - self.block_size

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

