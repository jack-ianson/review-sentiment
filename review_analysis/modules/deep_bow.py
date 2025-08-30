from __future__ import annotations
import torch
import torch.nn as nn


class DeepBagOfWords(nn.Module):
    def __init__(
        self,
        vocal_size: int,
        embedding_dim: int,
        hidden_dims: list[int] | int,
        num_classes: int = 2,
    ):
        super(DeepBagOfWords, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.embedding = nn.Embedding(vocal_size, embedding_dim)

        layers = nn.Sequential()

        fc_layer_sizes = [embedding_dim * 2] + hidden_dims + [num_classes]

        for i in range(len(fc_layer_sizes) - 1):
            layers.append(nn.Linear(fc_layer_sizes[i], fc_layer_sizes[i + 1]))
            if i < len(fc_layer_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))

        self.fc_layers = layers

    def forward(self, title_inputs: torch.Tensor, review_inputs: torch.Tensor):
        # title_inputs, review_inputs: (batch_size, seq_len)
        title_embedded = self.embedding(title_inputs)
        review_embedded = self.embedding(review_inputs)

        # title_embedded, review_embedded: (batch_size, seq_len, embedding_dim)
        title_bow = torch.sum(title_embedded, dim=1)
        review_bow = torch.sum(review_embedded, dim=1)

        # title_bow, review_bow: (batch_size, embedding_dim)
        combined = torch.cat((title_bow, review_bow), dim=1)

        output = self.fc_layers(combined)

        return output
