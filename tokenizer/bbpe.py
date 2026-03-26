from collections import defaultdict
from typing import List, Dict


class BBPE:
    def __init__(self):
        pass

    def _build_corpus(self, corpus: List[str]):
        for sentence in corpus:
            for idx, word in enumerate(sentence.strip().split()):
                if idx == 0:
                    encoded = tuple(word.encode('utf-8'))
                else:
                    encoded = tuple((' ' + word).encode('utf-8'))
                self.vocab[encoded] += 1

    def _get_pair_count(self) -> Dict[tuple, int]:
        pairs = defaultdict(int)
        for word_tuple, freq in self.vocab.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pairs[pair] += freq
        return dict(pairs)

    def _merge_pair(self, pair) -> Dict:
        new_vocab: Dict[tuple, str] = {}
        bigram = pair

        for word_tuple, freq in self.vocab.items():
            i = 0
            token = []
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == bigram:
                    token.append(bigram)
                    i += 2
                else:
                    token.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(token)] = freq
        return dict(new_vocab)

    def train_bbpe(self, corpus, merge_count) -> List[tuple, str]:
        self.vocab: Dict[tuple, int] = defaultdict(int)
        self.tokens = []
        self._build_corpus(corpus=corpus)
        for _ in range(merge_count):
            pairs = self._get_pair_count()
            if not pairs:
                break
            best_pair = max(pairs, key=lambda x: (pairs[x], str(x)))
            self.vocab = self._merge_pair(pair=best_pair)
            self.tokens.append(best_pair)
        return self.tokens

    def tokenize(self, text: str):
        if not text.strip():
            return []
        all_tokens = []
        words = text.strip().split()

        for idx, word in enumerate(words):
            raw = word if idx == 0 else ' ' + word
            symbols = list(raw.encode('utf-8'))

            for token in self.tokens:
                i = 0
                new_symbol = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == token:
                        new_symbol.append(token)
                        i += 2
                    else:
                        new_symbol.append(symbols[i])
                        i += 1
                symbols = new_symbol
            all_tokens.extend([self._decode_token(s) for s in symbols])
        return all_tokens

    def _decode_token(self, token: List[int]) -> str:
        if isinstance(token, int):
            return bytes([token]).decode('latin-1')
        elif isinstance(token, tuple):
            return ''.join(self._decode_token(t) for t in token)
        return str(token)


if __name__ == "__main__":
    corpus = [
        "low low low low",
        "lower lower",
        "newest newest newest",
        "widest widest",
    ]

    bpe = BBPE()
    merges = bpe.train_bbpe(corpus, merge_count=10)
    print(merges)
    print(bpe.tokenize("low lower"))
    print(bpe.tokenize("newest"))
    print(bpe.tokenize("hello"))
