import torch
from torch.utils.data import Dataset

from .tokeniser import Tokeniser


class ReviewDataset(Dataset):
    def __init__(
        self,
        titles: list[str],
        reviews: list[str],
        labels: list[int],
        tokeniser: Tokeniser,
    ):
        self.titles = titles
        self.reviews = reviews
        self.labels = labels
        self.tokeniser = tokeniser

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):

        title = self.titles[index]
        review = self.reviews[index]
        label = self.labels[index]

        title_ids = self.tokeniser.encode(title)
        review_ids = self.tokeniser.encode(review)

        return title_ids, review_ids, label
