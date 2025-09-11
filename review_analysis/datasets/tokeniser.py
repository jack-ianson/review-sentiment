from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import json


STOP_WORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
]


class Tokeniser:
    def __init__(self, max_vocab_size: int = 5000, seq_len: int = None):
        self.max_vocab_size = max_vocab_size
        self.seq_len = seq_len

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self._counter = Counter()
        self._vocab_finalised = False

    def fit_one(self, text: str):
        """
        Fit the tokeniser on a single text string.

        Args:
            text (str): input string
        """
        tokens = self.tokenise(text)
        self._counter.update(tokens)
        self._vocab_finalised = False

    def fit_many(self, texts: list[str]):
        """
        Fit the tokeniser on a list of text strings.

        Args:
            texts (list[str]): list of strings
        """
        for text in texts:
            self.fit_one(text)
        self._vocab_finalised = False

    def finalise_vocab(self):
        """
        Finalise the vocabulary by selecting the most common words up to max_vocab_size.
        """
        if self._vocab_finalised:
            return

        most_common = self._counter.most_common(self.max_vocab_size - 2)

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self._vocab_finalized = True

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs, padding or truncating to seq_len if specified.

        Args:
            text (str): input string

        Returns:
            list[int]: list of token IDs
        """
        if not self._vocab_finalised:
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

    def save(self, path: str | Path):
        """
        Save the tokeniser configuration to a JSON file.

        Args:
            path (str | Path): the directory path to save the tokeniser configuration
        """

        if Path(path).name.endswith(".json"):
            path = Path(path)
        else:
            path = Path(path) / "tokeniser.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_vocab_size": self.max_vocab_size,
                    "seq_len": self.seq_len,
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    def load(self, path: str | Path):
        """
        Load the tokeniser configuration from a JSON file.

        Args:
            path (str | Path): the file path to load the tokeniser configuration
        """

        if Path(path).is_dir():
            path = Path(path) / "tokeniser.json"
        else:
            path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.max_vocab_size = data["max_vocab_size"]
            self.seq_len = data["seq_len"]
            self.word2idx = data["word2idx"]
            self.idx2word = {int(k): v for k, v in data["idx2word"].items()}

        self._vocab_finalised = True

    @staticmethod
    def tokenise(text: str) -> list[int]:
        """
        Tokenise a text string into a list of lowercase words.

        Args:
            text (str): input string

        Returns:
            list[int]: list of tokens
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        words = re.findall(r"\b\w+\b", text.lower())
        return [word for word in words if word not in STOP_WORDS]
