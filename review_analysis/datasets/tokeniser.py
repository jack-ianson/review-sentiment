from __future__ import annotations
import re


class Tokeniser:
    def __init__(self, max_vocab_size: int = 5000, seq_len: int = None):
        self.max_vocab_size = max_vocab_size
        self.seq_len = seq_len

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit_one(self, text: str):
        tokens = self.tokenise(text)

        for token in tokens:
            if token not in self.word2idx:
                if len(self.word2idx) < self.max_vocab_size:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token

    def fit_many(self, texts: list[str]):
        for text in texts:
            self.fit_one(text)

    def encode(self, text: str) -> list[int]:
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
        return re.findall(r"\b\w+\b", text.lower())
