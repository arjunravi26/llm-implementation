from collections import defaultdict
import re

# ─────────────────────────────────────────────
# STEP 1: Build initial vocabulary from corpus
# ─────────────────────────────────────────────

def build_vocab(corpus: list[str]) -> dict[tuple, int]:
    """
    Represent each word as a tuple of characters + end-of-word marker.
    Count word frequencies.
    
    'low' → ('l', 'o', 'w', '</w>')
    """
    vocab = defaultdict(int)
    for sentence in corpus:
        for word in sentence.strip().split():
            # '</w>' marks word boundary — crucial for decoding later
            token = tuple(list(word) + ['</w>'])
            vocab[token] += 1
    return dict(vocab)


# ─────────────────────────────────────────────
# STEP 2: Count all adjacent pairs
# ─────────────────────────────────────────────

def get_pair_counts(vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Count frequency of every adjacent symbol pair across all words."""
    pairs = defaultdict(int)
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pairs[pair] += freq
    return dict(pairs)


# ─────────────────────────────────────────────
# STEP 3: Merge the best pair in vocab
# ─────────────────────────────────────────────

def merge_pair(pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
    """
    Replace all occurrences of `pair` in every word tuple with merged token.
    ('l','o') → 'lo'
    """
    new_vocab = {}
    bigram = pair  # e.g. ('l', 'o')
    merged = ''.join(pair)  # e.g. 'lo'

    for word_tuple, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            # Check if current + next matches our target pair
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == bigram:
                new_word.append(merged)
                i += 2  # skip both symbols
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_word)] = freq

    return new_vocab


# ─────────────────────────────────────────────
# STEP 4: Train BPE
# ─────────────────────────────────────────────

def train_bpe(corpus: list[str], num_merges: int):
    """
    Returns:
      - merges: ordered list of merge rules [(pair, merged_token), ...]
      - final_vocab: {word_tuple: freq}
      - token_vocab: set of all tokens (base chars + merged)
    """
    vocab = build_vocab(corpus)

    # Base vocab = all unique characters (including </w>)
    token_vocab = set()
    for word_tuple in vocab:
        token_vocab.update(word_tuple)

    merges = []

    for i in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break

        # Pick the most frequent pair (tie-break by pair value for determinism)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged_token = ''.join(best_pair)

        vocab = merge_pair(best_pair, vocab)
        merges.append((best_pair, merged_token))
        token_vocab.add(merged_token)

        print(f"Merge {i+1:3d}: {best_pair} → '{merged_token}'  (freq={pair_counts[best_pair]})")

    return merges, vocab, token_vocab


# ─────────────────────────────────────────────
# STEP 5: Tokenize new text using learned merges
# ─────────────────────────────────────────────

def tokenize(text: str, merges: list) -> list[str]:
    """
    Apply merge rules in ORDER to a new string.
    Order matters — earlier merges take priority.
    """
    words = text.strip().split()
    all_tokens = []

    for word in words:
        # Start as individual characters
        symbols = list(word) + ['</w>']

        # Apply each merge rule sequentially
        for (pair, merged) in merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        all_tokens.extend(symbols)

    return all_tokens