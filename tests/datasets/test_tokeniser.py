from __future__ import annotations
import pytest
import torch
import os

from review_analysis.datasets import Tokeniser


def test_tokeniser_1():

    text = "This is an example sentence for testing the Tokeniser. These are a few duplicate words: example example example."

    tokeniser = Tokeniser(seq_len=10)

    tokeniser.fit_one(text)

    assert tokeniser.word2idx["<PAD>"] == 0
    assert tokeniser.word2idx["<UNK>"] == 1

    assert tokeniser.word2idx["example"] == 5
    assert tokeniser.word2idx["this"] == 2

    encoded = tokeniser.encode("This is an unknown word.")
    assert encoded == [2, 3, 4, 1, 1, 0, 0, 0, 0, 0]

    encoded = tokeniser.encode("example example example")
    assert encoded == [5, 5, 5, 0, 0, 0, 0, 0, 0, 0]


def test_tokeniser_2():

    texts = [
        "This is an example sentence for testing the Tokeniser.",
        "These are a few duplicate words: example example example.",
    ]

    tokeniser = Tokeniser(seq_len=10)
    tokeniser.fit_many(texts)

    assert tokeniser.word2idx["<PAD>"] == 0
    assert tokeniser.word2idx["<UNK>"] == 1

    assert tokeniser.word2idx["example"] == 5
    assert tokeniser.word2idx["this"] == 2

    encoded = tokeniser.encode("This is an unknown word.")
    assert encoded == [2, 3, 4, 1, 1, 0, 0, 0, 0, 0]

    encoded = tokeniser.encode("example example example")
    assert encoded == [5, 5, 5, 0, 0, 0, 0, 0, 0, 0]


def test_tokeniser_max_vocab():

    text = "This is an example sentence for testing the Tokeniser. These are a few duplicate words: example example example."

    tokeniser = Tokeniser(max_vocab_size=10, seq_len=10)

    tokeniser.fit_one(text)

    assert len(tokeniser.word2idx) == 10  # 8 unique words + <PAD> + <UNK>

    assert tokeniser.word2idx["<PAD>"] == 0
    assert tokeniser.word2idx["<UNK>"] == 1

    assert tokeniser.word2idx["example"] == 5
    assert tokeniser.word2idx["this"] == 2

    encoded = tokeniser.encode("This is an unknown word.")
    assert encoded == [2, 3, 4, 1, 1, 0, 0, 0, 0, 0]

    encoded = tokeniser.encode("example example example")
    assert encoded == [5, 5, 5, 0, 0, 0, 0, 0, 0, 0]


def test_tokeniser_seq_len():
    text = "This is an example sentence for testing the Tokeniser."

    tokeniser = Tokeniser(seq_len=5)

    tokeniser.fit_one(text)

    encoded = tokeniser.encode(text)
    assert len(encoded) == 5
    assert encoded == [2, 3, 4, 5, 6]  # Truncated to seq_len

    tokeniser = Tokeniser(seq_len=15)

    tokeniser.fit_one(text)

    encoded = tokeniser.encode(text)
    assert len(encoded) == 15

    print(encoded)

    assert encoded == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Padded to seq_len
