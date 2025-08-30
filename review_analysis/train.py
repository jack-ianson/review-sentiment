from __future__ import annotations
import torch


from .datasets import load_dataset


def train_bag_of_words(data_root: str = None, validation: bool = True):

    # load the data
    if not data_root:
        data_root = "data"

    training_data = load_dataset(f"{data_root}/train.csv", n=1000)

    if validation:
        testing_data = load_dataset(f"{data_root}/val.csv", n=200)
    else:
        testing_data = load_dataset(f"{data_root}/test.csv", n=200)

    print(
        f"Loaded {len(training_data)} training samples and {len(testing_data)} {'validation' if validation else 'testing'} samples."
    )
