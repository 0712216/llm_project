from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import os
import torch

class BPETokenizer:
    def __init__(self, tokenizer_path="checkpoints/bpe_tokenizer.json"):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        if not text:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train_from_text(cls, files, tokenizer_path="checkpoints/bpe_tokenizer.json", vocab_size=5000, special_tokens=None):
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = Whitespace()
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train(files=files, trainer=trainer)
        tokenizer.save(tokenizer_path)
        return cls(tokenizer_path)

# 測試用：
if __name__ == "__main__":
    tokenizer = BPETokenizer.train_from_text(["data/tiny_shakespeare.txt"])
    print("Vocab size:", tokenizer.vocab_size)
    encoded = tokenizer.encode("To be or not to be")
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
    assert tokenizer.decode(encoded) == "To be or not to be"
