from __future__ import annotations
import pytest
import torch
import os

from review_analysis.datasets import load_dataset


def test_load_dataset():
    data_path = f"{os.path.dirname(__file__)}/test_data/small_test.csv"

    data_1 = load_dataset(data_path, shuffle=False)

    assert len(data_1) == 200

    data_2 = load_dataset(data_path, n=10, shuffle=False)

    assert len(data_2) == 10
    assert data_1[:10].equals(data_2)

    data_3 = load_dataset(data_path, n=10, shuffle=True)

    assert data_2.equals(data_3) is False
