import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model: int, d_head: int, num_embedding: int = 10000):

        super().__init__()

        assert d_model % d_head == 0, "`d_head` must be a divisible of `d_model`"

        self.d_model = d_model
        self.d_head = d_head
        self.d_key = self.d_model // self.d_head

        self.vocab = {}
        self.sequence_length = []

        self.embedding_fn = nn.Embedding(
            num_embeddings=num_embedding, embedding_dim=d_model)

        self.Wkey = nn.Linear(in_features=d_model,
                              out_features=d_model, bias=False)
        self.Wquery = nn.Linear(in_features=d_model,
                                out_features=d_model, bias=False)
        self.Wvalue = nn.Linear(in_features=d_model,
                                out_features=d_model, bias=False)

        self.Woutput = nn.Linear(
            in_features=d_model, out_features=d_model, bias=False)

    def convert_token_embeddings(self, token_ids: torch.Tensor):
        return self.embedding_fn(token_ids)

    def single_head_attn(self, embeddings: torch.Tensor) -> torch.Tensor:

        key = self.Wkey(embeddings)
        query = self.Wquery(embeddings)
        value = self.Wvalue(embeddings)

        B, S, _ = key.shape

        key = key.view(B, S, self.d_head, self.d_key).transpose(1, 2)
        query = query.view(B, S, self.d_head, self.d_key).transpose(1, 2)
        value = value.view(B, S, self.d_head, self.d_key).transpose(1, 2)

        attn_weights = query @ key.transpose(-1, -2)
        attn_weights_normalized = attn_weights / math.sqrt(self.d_key)
        attn_scores = F.softmax(attn_weights_normalized, dim=-1)

        context_vector = attn_scores @ value

        context_vector = context_vector.transpose(1, 2).contiguous()
        context_vector = context_vector.view(B, S, self.d_model)

        return self.Woutput(context_vector)

    def forward(self, token_ids: torch.Tensor):
        embeddings = self.convert_token_embeddings(token_ids=token_ids)
        context_vector = self.single_head_attn(embeddings=embeddings)
        return context_vector


if __name__ == "__main__":
    data = [
        "Hello, How are you?",
        "What are you doing now?",
        "The smell of fresh rain on dry earth is called petrichor.",
        "Could you please pass the salt?",
        "I'm planning to go for a long walk once the sun sets.",
        "The old lighthouse stood as a silent sentinel against the crashing waves.",
        "A flicker of doubt crossed his face, but he quickly masked it with a smile"
    ]
    # multi_head_attn = MultiHeadAttn(d_model=128, d_head=8)
    # context_vector = multi_head_attn.forward(data=data)
    # print(f"Context vector generated is: {context_vector}")
    # print(f"Dim of context vector: {context_vector.shape}")
