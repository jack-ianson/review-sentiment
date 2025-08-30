from __future__ import annotations
import torch
from torch.utils.data import DataLoader

from .. import datasets


def train_bag_of_words(data_root: str = None, validation: bool = True):

    # load the data
    if not data_root:
        data_root = "data"

    training_data = datasets.load_dataset(f"{data_root}/train.csv", n=1000)

    if validation:
        testing_data = datasets.load_dataset(f"{data_root}/val.csv", n=200)
    else:
        testing_data = datasets.load_dataset(f"{data_root}/test.csv", n=200)

    print(
        f"Loaded {len(training_data)} training samples and {len(testing_data)} {'validation' if validation else 'testing'} samples."
    )

    training_titles = training_data["title"].tolist()
    training_reviews = training_data["review"].tolist()
    training_labels = training_data["label"].tolist()

    testing_titles = testing_data["title"].tolist()
    testing_reviews = testing_data["review"].tolist()
    testing_labels = testing_data["label"].tolist()

    # create the tokeniser and fit on the training data, this ensures no data leakage of new words
    tokeniser = datasets.Tokeniser(max_vocab_size=10000)

    tokeniser.fit_many(training_titles)
    tokeniser.fit_many(training_reviews)

    print(f"Vocabulary size: {len(tokeniser.word2idx)}")

    # create the datasets
    training_dataset = datasets.ReviewDataset(
        titles=training_titles,
        reviews=training_reviews,
        labels=training_labels,
        tokeniser=tokeniser,
    )

    testing_dataset = datasets.ReviewDataset(
        titles=testing_titles,
        reviews=testing_reviews,
        labels=testing_labels,
        tokeniser=tokeniser,
    )

    # dataloaders
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=datasets.bow_collate_fn,
    )
    test_dataloader = DataLoader(
        testing_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=datasets.bow_collate_fn,
    )
