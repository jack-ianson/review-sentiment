from __future__ import annotations
import re
from collections import Counter


class Tokeniser:
    def __init__(self, max_vocab_size: int = 5000, seq_len: int = None):
        self.max_vocab_size = max_vocab_size
        self.seq_len = seq_len

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self._counter = Counter()
        self._vocal_finalised = False

    def fit_one(self, text: str):
        tokens = self.tokenise(text)
        self._counter.update(tokens)
        self._vocal_finalised = False

    def fit_many(self, texts: list[str]):
        for text in texts:
            self.fit_one(text)
        self._vocal_finalised = False

    def finalise_vocab(self):
        if self._vocal_finalised:
            return

        most_common = self._counter.most_common(self.max_vocab_size - 2)

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self._vocab_finalized = True

    def encode(self, text: str) -> list[int]:

        if not self._vocal_finalised:
            self.finalise_vocab()

        tokens = self.tokenise(text)
        token_ids = [
            self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens
        ]

        if self.seq_len is None:
            return token_ids

        if len(token_ids) < self.seq_len:
            token_ids += [self.word2idx["<PAD>"]] * (self.seq_len - len(token_ids))
        else:
            token_ids = token_ids[: self.seq_len]

        return token_ids

    @staticmethod
    def tokenise(text: str) -> list[int]:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return re.findall(r"\b\w+\b", text.lower())
