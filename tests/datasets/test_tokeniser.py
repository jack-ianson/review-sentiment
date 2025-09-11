from __future__ import annotations
import pytest
import torch
import os

from review_analysis.datasets import Tokeniser


def test_tokeniser_fit_one():

    text = "This is a good example sentence for testing the Tokeniser. These are a few duplicate words: example example example."

    tokeniser = Tokeniser()

    tokeniser.fit_one(text)

    tokeniser.finalise_vocab()

    encoded = tokeniser.encode("This is an example unknown word.")

    assert encoded == [2, 1, 1]

    encoded = tokeniser.encode("example example example")
    assert encoded == [2, 2, 2]


def test_tokeniser_fit_many():

    texts = [
        "This is an example sentence for testing the Tokeniser.",
        "These are a few duplicate words: example example example.",
    ]

    tokeniser = Tokeniser(seq_len=10)
    tokeniser.fit_many(texts)

    tokeniser.finalise_vocab()
    encoded = tokeniser.encode("This is an unknown word.")

    assert encoded == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    encoded = tokeniser.encode("example example example")
    assert encoded == [2, 2, 2, 0, 0, 0, 0, 0, 0, 0]


def test_tokeniser_fit_many_multiple():

    text_1 = "This is an example sentence for testing the Tokeniser."
    text_2 = "These are a few duplicate words: example example example."
    texts = ["This is an example", "test one two three", "another test example"]

    tokeniser = Tokeniser()

    tokeniser.fit_one(text_1)
    tokeniser.fit_one(text_2)
    tokeniser.fit_many(texts)

    tokeniser.finalise_vocab()
    encoded = tokeniser.encode("This is an unknown word.")
    assert encoded == [1, 1]

    encoded = tokeniser.encode("example example example")
    assert encoded == [2, 2, 2]


def test_tokeniser_max_vocab():

    text = "This is an example sentence for testing the Tokeniser. These are a few duplicate words: example example example."

    tokeniser = Tokeniser(max_vocab_size=10, seq_len=10)

    tokeniser.fit_one(text)
    tokeniser.finalise_vocab()

    encoded = tokeniser.encode("This is an unknown word.")
    assert encoded == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    encoded = tokeniser.encode("example example example")
    assert encoded == [2, 2, 2, 0, 0, 0, 0, 0, 0, 0]


def test_tokeniser_seq_len():
    text = "This is an example sentence for testing the Tokeniser."

    tokeniser = Tokeniser(seq_len=5)

    tokeniser.fit_one(text)
    tokeniser.finalise_vocab()

    encoded = tokeniser.encode(text)
    assert len(encoded) == 5
    assert encoded == [2, 3, 4, 5, 0]  # Truncated to seq_len

    tokeniser = Tokeniser(seq_len=15)

    tokeniser.fit_one(text)
    tokeniser.finalise_vocab()

    encoded = tokeniser.encode(text)
    assert len(encoded) == 15

    assert encoded == [
        2,
        3,
        4,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Padded to seq_len


def test_tokeniser_finalise():

    text = "This is an example sentence for testing the Tokeniser. These are a few duplicate words: example example example."

    tokeniser = Tokeniser(max_vocab_size=10)

    tokeniser.fit_one(text)

    assert len(tokeniser._counter) == 6

    tokeniser.finalise_vocab()

    assert len(tokeniser.word2idx) == 8  # 8 most common words + <PAD> + <UNK>

    encoded = tokeniser.encode("This is an unknown word.")

    assert encoded == [1, 1]

    encoded = tokeniser.encode("example example example")
    assert encoded == [2, 2, 2]
