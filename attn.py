import torch
import torch.nn as nn
import torch.nn.functional as F


def process_text(text: str):
    words = text.split()
    word_dct = {word: idx for idx, word in enumerate(words)}
    tokens = torch.tensor(list(word_dct.values()))
    return words, tokens, word_dct


def single_head_attn(tokens, d_model=12):

    # Create embeddings
    embedding_fn = nn.Embedding(embedding_dim=d_model, num_embeddings=10)
    embeddings = embedding_fn(tokens)
    print(f"Embedding values: {embeddings}")
    print(f"Dim of embeddings: {embeddings.shape}")

    # Key, Query, Value
    Wkey = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    Wquery = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    Wvalue = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    key = Wkey(embeddings)
    query = Wquery(embeddings)
    value = Wvalue(embeddings)

    attn_scores = query @ key.T
    attn_scores = attn_scores / d_model
    attn_scores = F.softmax(attn_scores, dim=1)

    context_vector = attn_scores @ value
    return context_vector


if __name__ == "__main__":
    text = "Hello, how're you?"
    words, tokens, word_dct = process_text(text)
    print(f"Dim of tokens: {tokens.shape}")
    context_vector = single_head_attn(tokens=tokens)
    print(f"Context vector: {context_vector}")
