# Character level BPE
from collections import defaultdict
from typing import List


class BPE:
    def __init__(self):
        pass

    def _build_corpus(self):
        for sentence in self.corpus:
            for word in sentence.strip().split():
                token = tuple(list(word) + ["</w>"])
                self.vocab[token] += 1

    def _get_pair_count(self):
        pairs = defaultdict(int)

        for word_tuple, freq in self.vocab.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pairs[pair] += freq
        return pairs

    def _merge_pair(self, pair):
        new_vocab = {}
        bigram = pair
        merged_token = ''.join(pair)

        for word_tuple, freq in self.vocab.items():
            i = 0
            new_word = []
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == bigram:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return dict(new_vocab)

    def train_bpe(self, corpus: List[str], merge_count: int):

        self.vocab = defaultdict(int)
        self.bpe_tokens = []
        self.corpus = corpus
        self._build_corpus()

        for _ in range(merge_count):
            pairs = self._get_pair_count()
            if not pairs:
                break
            best_pair = max(pairs, key=lambda x: (pairs[x], x))
            token = ''.join(best_pair)
            self.vocab = self._merge_pair(pair=best_pair)
            self.bpe_tokens.append((best_pair, token))
        return self.bpe_tokens, self.vocab

    def tokenize(self, text: str):
        all_tokens = []
        for word in text.split():
            symbol = list(word) + ["</w>"]
            for pair, token in self.bpe_tokens:
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
    bpe = BPE()
    tokens, vocab = bpe.train_bpe(corpus=corpus, merge_count=10)
    print(f"Tokens are: {tokens}")
    print(f"Vocab are: {vocab}")

    print(bpe.tokenize(text="low lower widest abc"))
    print(bpe.tokenize(text="low neww abc"))
