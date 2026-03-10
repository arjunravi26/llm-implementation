import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
import re


class TextProcessor(Dataset):
    def __init__(self, seq_len=8):
        super().__init__()
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK"
        self.seq_len = seq_len
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        self.id_to_token = {0: self.pad_token, 1: self.unk_token}

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def build_vocab(self, texts: List[str]):
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
                    self.id_to_token[len(self.id_to_token)] = token
        print(f"Token to id: {self.token_to_id}")
        print(f"Id to token: {self.id_to_token}")

    def pad_or_truncate_ids(self, ids: List):
        if len(ids) < self.seq_len:
            pad_len = self.seq_len - len(ids)
            padded_ids = [0] * pad_len
            ids.extend(padded_ids)
        elif len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
        return ids

    def tokenize(self, text: str):
        text = self.normalize_text(text)
        return text.split()

    def encode(self, texts: List[str]):
        token_ids = []
        for text in texts:
            tokens = self.tokenize(text=text)
            ids = self.pad_or_truncate_ids(
                ids=[self.token_to_id[token] for token in tokens])
            token_ids.append(ids)
        return torch.tensor(token_ids)


if __name__ == "__main__":
    from core.calculate_seq_len import calculate_seq_len
    data = [
        "Hello, How are you?",
        "What are you doing now?",
        "The smell of fresh rain on dry earth is called petrichor.",
        "Could you please pass the salt?",
        "I'm planning to go for a long walk once the sun sets.",
        "The old lighthouse stood as a silent sentinel against the crashing waves.",
        "A flicker of doubt crossed his face, but he quickly masked it with a smile"
    ]

    seq_len = calculate_seq_len(texts=data)
    print(f"Sequence length is {seq_len}")

    text_processor = TextProcessor(seq_len=seq_len)
    text_processor.build_vocab(texts=data)
    token_ids = text_processor.encode(texts=data)
    print(f"Token ids are: {token_ids}")
