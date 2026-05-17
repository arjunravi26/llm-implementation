# 🧠 LLM Implementation from Scratch

> A ground-up PyTorch implementation of the core building blocks that power modern Large Language Models — tokenizers, attention mechanisms, and text processing utilities — built for deep understanding, not abstraction.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Topics](https://img.shields.io/badge/topics-attention%20%7C%20tokenizer%20%7C%20transformer-lightgrey)](#)

---

## 📖 Overview

This repository implements the fundamental components of transformer-based LLMs **entirely from scratch** using PyTorch — no HuggingFace, no `transformers` library, no black boxes. Every tensor operation, every matrix reshape, every merge rule is written by hand and is directly traceable to the underlying math.

The goal is not to replicate a production LLM, but to **deeply internalize how they work** — from how raw text becomes byte-level token IDs, all the way to how grouped query attention produces context vectors with a fraction of the KV memory cost.

---

## 📁 Repository Structure

```
llm-implementation/
│
├── attn/                        # Attention mechanisms
│   ├── attn.py                  # Scaled dot-product single-head attention (functional)
│   ├── mha.py                   # Multi-head attention (nn.Module, batched)
│   ├── gqa.py                   # Grouped Query Attention (GQA)
│   └── sliding_window.py        # Sliding Window Attention built on GQA
│
├── core/
│   └── calculate_seq_len.py     # Sequence length estimation via percentile
│
├── tokenizer/
│   ├── bpe.py                   # Character-level BPE from scratch
│   ├── bbpe.py                  # Byte-level BPE (BBPE) from scratch
│   ├── text_processing.py       # Vocab builder + encode/pad pipeline (nn.Dataset)
│   └── sample.py                # UTF-8 byte encoding demo (Devanagari script)
│

```

---

## 🔩 Components Deep Dive

### 1. Tokenizers (`tokenizer/`)

Tokenization is the gateway between raw text and numerical tensors. Two tokenization strategies are implemented from scratch.

#### `bpe.py` — Character-level Byte Pair Encoding

Classic BPE as described in the original [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) paper. Words are initialized as sequences of characters with an `</w>` end-of-word marker, then the most frequent adjacent symbol pair is iteratively merged.

**Key methods:**

| Method | Description |
|---|---|
| `_build_corpus()` | Converts corpus sentences into `{char_tuple: frequency}` vocab |
| `_get_pair_count()` | Scans all word tuples and counts adjacent symbol pair frequencies |
| `_merge_pair(pair)` | Rewrites every word tuple, replacing the target bigram with its merged token |
| `train_bpe(corpus, merge_count)` | Runs the full training loop, returns ordered merge rules + final vocab |
| `tokenize(text)` | Applies learned merge rules sequentially to unseen text |

**Key design detail:** Tie-breaking in `max()` is done by `(frequency, pair)` — this ensures deterministic training across runs even when multiple pairs share the same count.

```python
from tokenizer.bpe import BPE

bpe = BPE()
tokens, vocab = bpe.train_bpe(corpus=["low low low", "lower lower", "newest newest newest"], merge_count=10)
print(bpe.tokenize("low lower"))
# → ['low</w>', 'low', 'er</w>']
```

**Limitation:** Character-level BPE cannot handle characters absent from its training corpus — unknown characters become unknown tokens.

---

#### `bbpe.py` — Byte-level BPE (BBPE)

BBPE operates on **raw UTF-8 bytes** (integers 0–255) rather than characters. This is the approach used by GPT-2, LLaMA, and most modern LLMs. Because every possible UTF-8 string decomposes into bytes 0–255, **there are no unknown tokens** — any input, including Emoji, Devanagari, CJK characters, etc., can be tokenized.

**Key differences from character BPE:**

| Aspect | BPE (`bpe.py`) | BBPE (`bbpe.py`) |
|---|---|---|
| Base vocabulary | Individual characters | Bytes 0–255 |
| Unknown tokens | Yes (OOV chars) | Never |
| Unicode support | Limited | Full (any encoding) |
| Word boundary | `</w>` suffix | Leading space prepended to non-first words |
| Vocab representation | `tuple[str]` | `tuple[int]` (byte values) |

**Implementation detail — leading-space encoding:** Following GPT-2 convention, all non-first words in a sentence are encoded with a leading space (`' ' + word`). This means "lower" as a sentence-initial token is a different byte sequence from " lower" mid-sentence, giving the model positional information within the word.

```python
from tokenizer.bbpe import BBPE

bpe = BBPE()
bpe.train_bbpe(corpus=["low low low", "lower lower", "newest newest newest"], merge_count=10)
print(bpe.tokenize("newest"))
# Byte-level tokens decoded back to strings
```

---

#### `text_processing.py` — `TextProcessor` (Vocab + Encode Pipeline)

A `torch.utils.data.Dataset` subclass that handles the full text → token ID pipeline used by the attention modules.

**Pipeline:**
```
Raw text
  └─► normalize_text()    # lowercase + BOS/EOS wrapping
        └─► tokenize()    # whitespace split
              └─► build_vocab()  # frequency-ranked vocab with <PAD>, <UNK>
                    └─► encode()  # token → ID lookup + pad/truncate to seq_len
                          └─► torch.Tensor  [B, seq_len]
```

**Special tokens:**

| Token | ID | Role |
|---|---|---|
| `<PAD>` | 0 | Padding for shorter sequences |
| `<UNK>` | 1 | Any token not in vocabulary |
| `<BOS>` | — | Prepended to every sequence during normalization |
| `<EOS>` | — | Appended to every sequence during normalization |

**Sequence length handling:** `pad_or_truncate_ids()` ensures all sequences are exactly `seq_len` long — shorter sequences are right-padded with 0s, longer sequences are truncated.

---

#### `sample.py` (root) — Annotated BPE Walkthrough

A standalone, heavily commented script that walks through every stage of BPE training as pure functions (not a class). Each function has a docstring explaining the "why" behind the operation. Ideal for learning or for porting the logic to a new context.

---

### 2. Core Utilities (`core/`)

#### `calculate_seq_len.py`

```python
def calculate_seq_len(texts: List[str], p=90) -> int
```

Computes the `p`-th percentile word count across all input texts (after lowercasing and stripping punctuation), cast to `int64`. Using p=90 avoids the sequence length being dominated by one unusually long outlier while still covering most inputs without truncation — a practical heuristic for batched training.

---

### 3. Attention Mechanisms (`attn/`)

All attention variants follow the same fundamental architecture but differ in how they allocate query, key, and value heads.

#### `attn.py` — Single-Head Scaled Dot-Product Attention (Functional)

The simplest possible implementation. Takes raw tokens, creates an embedding, and computes attention in a single head with no batching.

```
Q, K, V = Wq(E), Wk(E), Wv(E)
scores   = Q @ K.T / d_model          ← scaling by d_model (note: should be √d_k)
weights  = softmax(scores, dim=1)
context  = weights @ V
```

> ⚠️ **Reviewer note:** The scaling factor here is `d_model` rather than `√d_k`. The canonical formula from *Attention Is All You Need* uses `√d_k`. For single-head where `d_k = d_model`, the correct divisor is `√d_model`. This is a subtle but important detail when extending to multi-head (see `mha.py`, which correctly uses `math.sqrt(self.d_key)`).

---

#### `mha.py` — Multi-Head Attention (Batched `nn.Module`)

A proper, production-style class implementing multi-head attention with full batch support and the output projection matrix `W_O`.

**Tensor shapes throughout the forward pass:**

```
Input token_ids:     [B, S]
Embeddings:          [B, S, d_model]
After W_q/k/v:       [B, S, d_model]
After reshape:       [B, S, n_heads, d_key]
After transpose:     [B, n_heads, S, d_key]      ← standard MHA layout
Attn weights:        [B, n_heads, S, S]
Context vector:      [B, n_heads, S, d_key]
After transpose:     [B, S, n_heads, d_key]
After view:          [B, S, d_model]              ← heads concatenated
After W_O:           [B, S, d_model]
```

**Constraint enforced:** `assert d_model % d_head == 0` — each head gets exactly `d_model // d_head` dimensions. This is the standard MHA constraint.

**Scaling:** Correctly uses `math.sqrt(self.d_key)` where `d_key = d_model // d_head`.

---

#### `gqa.py` — Grouped Query Attention (GQA)

GQA is the attention variant used in **LLaMA 2/3, Mistral, Gemma, and Qwen** — the dominant architecture in modern open-weight LLMs. Instead of having `n_heads` separate K/V projection heads (like MHA), GQA uses `n_kv_heads` shared K/V heads across groups of query heads.

**Memory savings:** With `query_head=8` and `key_head=2`, the K/V cache is 4× smaller than full MHA — critical for long-context inference.

**This implementation uses the `expand` + `reshape` strategy** (more memory-efficient than `repeat_interleave` at inference time since it avoids materializing extra copies):

```python
# key: [B, n_kv_heads, S, d_key]
key = key.unsqueeze(2)                                        # [B, n_kv_heads, 1, S, d_key]
key = key.expand(B, key_head, group_size, S, d_key)          # virtual broadcast
key = key.reshape(B, query_head, S, d_key)                   # materialized for bmm
```

**Constraints enforced:**
- `d_model % query_head == 0`
- `d_model % key_head == 0`
- `query_head % key_head == 0` — ensures groups divide evenly

**Full forward flow:**

```
embeddings: [B, S, d_model]
  ├─ Wq → [B, S, d_model]      → reshape → [B, query_head, S, d_key]
  ├─ Wk → [B, S, d_key*kv_h]  → reshape → [B, key_head,   S, d_key]
  └─ Wv → [B, S, d_key*kv_h]  → reshape → [B, key_head,   S, d_key]
                                              ↓ expand + reshape
                                           [B, query_head, S, d_key]
attn_score  = Q @ K.T / √d_key
attn_weight = softmax(attn_score, dim=-1)
context     = attn_weight @ V → [B, query_head, S, d_key]
            → transpose → view → [B, S, d_model]
            → W_output → [B, S, d_model]
```

---

#### `sliding_window.py` — Sliding Window Attention (SWA)

SWA restricts each token's attention span to a local window of size `window_size` preceding it. This is the mechanism behind **Mistral 7B**, **Gemma**, and the local layers of hybrid attention models. For a sequence of length `S`, standard attention is O(S²); SWA brings this down to O(S × window_size).

**This implementation builds SWA on top of the GQA architecture** (GQA + SWA = the Mistral design).

**Core sliding window logic (`_build_sliding_win`):**

```python
for start in range(S):
    kv_start = max(0, start - window_size)      # clamp to sequence start
    chunked_query = query[:, :, start:start+1, :]
    chunked_key   = key  [:, :, kv_start:start+1, :]
    chunked_value = value[:, :, kv_start:start+1, :]

    attn = softmax(chunked_query @ chunked_key.T / √E, dim=-1)
    output[:, :, start:start+1, :] = attn @ chunked_value
```

Each query token attends only to the `window_size` tokens before it (plus itself), so the effective attention matrix is a narrow band diagonal.

> **Note:** This is a pedagogically clear O(S × W) loop implementation. Production implementations (e.g., in vLLM or Flash Attention 2) use chunked CUDA kernels that avoid materializing the full S×S matrix entirely.

---

## ⚡ Getting Started

### Prerequisites

```bash
pip install torch numpy spacy nltk
```

### Run Single-Head Attention

```bash
cd llm-implementation
python attn/attn.py
```

### Run Multi-Head Attention

```python
from attn.mha import MultiHeadAttn
import torch

model = MultiHeadAttn(d_model=128, d_head=8, num_embedding=10000)
token_ids = torch.randint(0, 10000, (4, 16))   # [batch=4, seq_len=16]
output = model(token_ids)
print(output.shape)   # → torch.Size([4, 16, 128])
```

### Run Grouped Query Attention

```python
from tokenizer.text_processing import TextProcessor
from core.calculate_seq_len import calculate_seq_len
from attn.gqa import GQA

corpus = ["Hello, How are you?", "What are you doing now?"]
seq_len = calculate_seq_len(texts=corpus)

processor = TextProcessor(seq_len=seq_len)
processor.build_vocab(texts=corpus)
token_ids = processor.encode(texts=corpus)   # [B, seq_len]

gqa = GQA(d_model=512, query_head=8, key_head=2)
context = gqa(token_ids)
print(context.shape)   # → torch.Size([2, seq_len, 512])
```

### Train a BPE Tokenizer

```python
from tokenizer.bpe import BPE

corpus = ["low low low low", "lower lower", "newest newest newest", "widest widest"]
bpe = BPE()
merges, vocab = bpe.train_bpe(corpus=corpus, merge_count=10)

print(bpe.tokenize("low lower widest"))
```

### Train a Byte-level BPE Tokenizer

```python
from tokenizer.bbpe import BBPE

bpe = BBPE()
bpe.train_bbpe(corpus=["low low low", "lower lower", "newest newest"], merge_count=10)

# Works on any Unicode input — no unknown tokens
print(bpe.tokenize("नमस्ते"))   # Devanagari
print(bpe.tokenize("hello world"))
```

---

## 🗺️ Component Relationship Map

```
                          Raw Text
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           BPE            BBPE         TextProcessor
        (char-level)   (byte-level)  (word-level + vocab)
              │               │               │
              └───────────────┴───────────────┘
                              │
                         Token IDs
                    [Batch, SeqLen] Tensor
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
              Single-Head  Multi-Head   GQA
                Attn         Attn        │
                                        ▼
                                Sliding Window Attn
                                (GQA + local mask)
                                        │
                              Context Vectors
                             [B, S, d_model]
```

---

## 🔬 Technical Notes & Known Issues

| Module | Observation |
|---|---|
| `attn.py` | Scales by `d_model` instead of `√d_model`. Should be `math.sqrt(d_model)` for correctness. |
| `attn.py` | No causal mask — all positions attend to all other positions. Suitable for encoder-only; add upper-triangular mask for decoder. |
| `mha.py` | `__main__` block defines `data` list but does not instantiate `MultiHeadAttn` (lines are commented out). |
| `gqa.py` | Uses `expand` + `reshape` for KV head broadcast — more memory-efficient than `repeat_interleave`. Both approaches are valid. |
| `sliding_window.py` | Loop-based chunked attention — correct but O(S × W) sequential Python loop. Vectorizable with masked attention for production use. |
| `text_processing.py` | Whitespace tokenizer — does not strip punctuation before building vocab, so `"you?"` and `"you"` are different tokens. |
| `calculate_seq_len.py` | Strips punctuation before counting words, which may cause a slight length mismatch with `TextProcessor.tokenize()`. |

---


---

## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al. (2023)
---

## 👤 Author

**Arjun Ravi**
AI Engineer | specializing in LLM, SLMs, local LLM inference, and document extraction pipelines

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
