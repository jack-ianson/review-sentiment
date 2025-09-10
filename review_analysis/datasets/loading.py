from __future__ import annotations
import pandas as pd
from pathlib import Path
import torch


    


def load_dataset(path: str | Path, n: int = None, shuffle: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        path (str | Path): Path to the CSV file.
        n (int, optional): Number of samples to return. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(path)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    if n:
        df = df.head(n)

    df.columns = ["label", "title", "review"]

    # remove rows with NaN titles or reviews
    df = df[(df["title"].str.strip() != "") & (df["review"].str.strip() != "")]

    return df
