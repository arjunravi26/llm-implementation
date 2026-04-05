import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GQA(nn.Module):
    def __init__(self, d_model, query_head, key_head, num_embedding=1000):
        super().__init__()
        assert d_model % query_head == 0, "`query_head` must be the factor of `d_model`"
        assert d_model % key_head == 0, "`key_head` must be the factor of `d_model`"
        assert query_head % key_head == 0, "`query_head` must be the factor of `key_head`"

        self.d_model = d_model
        self.query_head = query_head
        self.key_head = key_head

        self.d_key = d_model // query_head

        self.Wkey = nn.Linear(self.d_model, self.d_key *
                              self.key_head, bias=False)
        self.Wvalue = nn.Linear(
            self.d_model, self.d_key * self.key_head, bias=False)
        self.Wquery = nn.Linear(self.d_model, self.d_model, bias=False)

        self.Woutput = nn.Linear(self.d_model, self.d_model)
        self.embedding_fn = nn.Embedding(
            num_embeddings=num_embedding, embedding_dim=d_model)

    def embedding(self, X):
        return self.embedding_fn(X)

    def gqa(self, embeddings):

        key: torch.Tensor = self.Wkey(embeddings)
        value: torch.Tensor = self.Wvalue(embeddings)
        query: torch.Tensor = self.Wquery(embeddings)

        B, S, _ = key.shape

        key = key.view(B, S, self.key_head, self.d_key).transpose(1, 2)
        value = value.view(B, S, self.key_head, self.d_key).transpose(1, 2)
        query = query.view(B, S, self.query_head, self.d_key).transpose(1, 2)

        group_size = self.query_head // self.key_head

        # key = key.repeat_interleave(group_size, dim=1)
        # value = value.repeat_interleave(group_size, dim=1)

        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

        key = key.expand(B, self.key_head, group_size, S, self.d_key)
        value = value.expand(B, self.key_head, group_size, S, self.d_key)

        key = key.reshape(B,self.query_head,S,self.d_key)
        value = value.reshape(B, self.query_head, S, self.d_key)

        attn_score = query @ key.transpose(-1, -2)
        attn_normalized = attn_score / math.sqrt(self.d_key)
        attn_weights = F.softmax(attn_normalized, dim=-1)

        context_vector = attn_weights @ value

        context_vector = context_vector.transpose(1, 2).contiguous()

        context_vector = context_vector.view(B, S, self.d_model)

        return self.Woutput(context_vector)

    def forward(self, X):
        embeddings = self.embedding(X)
        output = self.gqa(embeddings=embeddings)
        return output
