import pandas as pd
from pathlib import Path


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

    return df
