import pickle
import torch

class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def from_text(cls, text):
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    @classmethod
    def from_vocab_file(cls, path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        return cls(vocab['stoi'], vocab['itos'])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'stoi': self.stoi, 'itos': self.itos}, f)

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.itos[i] for i in indices])