from __future__ import annotations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

from .tokeniser import Tokeniser


class ReviewDataset(Dataset):
    def __init__(
        self,
        titles: list[str] | list[torch.Tensor],
        reviews: list[str] | list[torch.Tensor],
        labels: list[int],
        tokeniser: Tokeniser,
        precompute_tokens: bool = False,
    ):
        self.titles = titles
        self.reviews = reviews
        self.labels = labels
        self.tokeniser = tokeniser
        self.precompute_tokens = precompute_tokens

        if precompute_tokens and not isinstance(titles[0], torch.Tensor):
            self.titles = [
                torch.tensor(self.tokeniser.encode(title))
                for title in tqdm(
                    titles, desc="Tokenising titles", total=len(self.titles)
                )
            ]
            self.reviews = [
                torch.tensor(self.tokeniser.encode(review))
                for review in tqdm(
                    reviews, desc="Tokenising reviews", total=len(self.reviews)
                )
            ]

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):

        title = self.titles[index]
        review = self.reviews[index]
        label = self.labels[index]

        if not self.precompute_tokens and not isinstance(self.titles[0], torch.Tensor):
            title = self.tokeniser.encode(title)
            review = self.tokeniser.encode(review)

        return title, review, label - 1

    def cache_tokenisation(
        self, path: Path | str, store_tokeniser: bool = True
    ) -> None:

        path = Path(path)

        if isinstance(self.titles[0], torch.Tensor):
            torch.save(
                {
                    "titles": self.titles,
                    "reviews": self.reviews,
                    "labels": self.labels,
                },
                path,
            )

        if store_tokeniser:
            if path.name.endswith(".pt"):
                self.tokeniser.save(path.parent)
