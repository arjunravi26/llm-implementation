from collections import defaultdict
from typing import List, Dict, Tuple


def build_vocab(corpus: List[str]) -> Dict[Tuple, int]:
    vocab = defaultdict(int)
    for sentence in corpus:
        for word in sentence.strip().split():
            token = tuple(list(word) + ['</w>'])
            vocab[token] += 1
    return vocab


def get_pair_count(vocab: Dict[Tuple, int]):
    pairs = defaultdict(int)
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pairs[pair] += freq
    return dict(pairs)


def merge_pair(pair: Tuple[str, str], vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
    new_vocab = {}
    bigram = pair
    merged = ''.join(pair)

    for word_tuple, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == bigram:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return dict(new_vocab)


def train(corpus: List[str], merge_count: int):
    vocab = build_vocab(corpus=corpus)
    merged = []
    for i in range(merge_count):
        pair_count = get_pair_count(vocab=vocab)
        if not pair_count:
            break
        best_pair = max(pair_count, key=lambda p: (pair_count[p], p))
        merged_token = ''.join(best_pair)
        vocab = merge_pair(pair=best_pair, vocab=vocab)
        merged.append((best_pair, merged_token))
    return merged, vocab


def tokenize(text: str, merges: List):
    all_tokens = []
    for word in text.strip().split():
        symbol = list(word) + ['</w>']
        for pair, token in merges:
            i = 0
            new_symbol = []
            while i < len(symbol):
                if i < len(symbol) - 1 and (symbol[i], symbol[i+1]) == pair:
                    new_symbol.append(token)
                    i += 2
                else:
                    new_symbol.append(symbol[i])
                    i += 1
            symbol = new_symbol
        all_tokens.extend(symbol)
    return all_tokens


if __name__ == "__main__":

    corpus = [
        "low low low low",
        "lower lower",
        "newest newest newest",
        "widest widest",
    ]

    merges, vocab = train(corpus=corpus, merge_count=10)
    print(f"Merges is: {merges}")
    print(f"Vocabulary is: {vocab}")
    print(f" Tokens: for low newer: {tokenize("low newer", merges)}")
    print(f" Tokens: for lowest: {tokenize("lowest", merges)}")
    print(f" Tokens: for abc wer: {tokenize("abc est", merges)}")
    print(tokenize("", merges))
