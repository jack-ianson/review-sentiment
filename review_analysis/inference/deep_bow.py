from __future__ import annotations
from pathlib import Path
import torch

from ..modules import DeepBagOfWords
from ..datasets import Tokeniser


class BagOfWordsInference:
    def __init__(
        self, model_path: Path | str, tokeniser_path: Path | str, device: str = "cpu"
    ):
        self.model_path = model_path
        self.tokeniser_path = tokeniser_path
        self.device = device

        self.tokeniser = Tokeniser()
        self.tokeniser.load(tokeniser_path)

        self.model = DeepBagOfWords(
            vocab_size=len(self.tokeniser.word2idx),
            embedding_dim=64,
            hidden_dims=[128, 128],
            num_classes=2,
        )

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(device)

        self.model.eval()

    def predict(self, title: str, review: str) -> tuple[int, float]:

        title_tokens = self.tokeniser.encode(title)
        review_tokens = self.tokeniser.encode(review)

        with torch.no_grad():

            title_tensor = torch.tensor([title_tokens], dtype=torch.long).to(
                self.device
            )
            review_tensor = torch.tensor([review_tokens], dtype=torch.long).to(
                self.device
            )

            output = self.model(title_tensor, review_tensor)

            predicted_class = torch.argmax(output, dim=1).item()
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        return predicted_class, probabilities
