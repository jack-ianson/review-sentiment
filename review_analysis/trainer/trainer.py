import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import warnings
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Callable

from ..datasets import ReviewDataset


_BatchType = tuple[Tensor, Tensor, Tensor]


class ReviewsModelTrainer:

    def __init__(
        self,
        model: nn.Module,
        training_dataset: ReviewDataset,
        testing_dataset: ReviewDataset,
        results_path: Path,
        collate_fn: Callable = None,
        device: torch.device | str = None,
        batch_size: int = 32,
        optimiser: torch.optim.Optimizer | str = "adam",
        criterion: nn.Module | str = "mse",
        lr: float = 0.001,
    ):

        # set the device and pass  the model to the device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)

        # store the datasets
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset

        # create dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            testing_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # define the optimiser
        if isinstance(optimiser, str):
            if optimiser.lower() == "adam":
                self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimiser.lower() == "adamw":
                self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=lr)
            else:
                warnings.warn(f"Unknown optimiser '{optimiser}', using Adam instead.")
                self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        elif isinstance(optimiser, torch.optim.Optimizer):
            self.optimiser = optimiser
        else:
            raise ValueError(
                "Optimiser must be a string ('adam' or 'adamw') or an instance of torch.optim.Optimizer."
            )

        # define the loss function
        if isinstance(criterion, str):
            if criterion.lower() == "mse":
                self.criterion = nn.MSELoss()
            elif criterion.lower() == "l1":
                self.criterion = nn.L1Loss()
            elif criterion.lower() == "cross_entropy":
                self.criterion = nn.CrossEntropyLoss()
            else:
                warnings.warn(
                    f"Unknown loss function '{criterion}', using MSE instead."
                )
                self.criterion = nn.MSELoss()

        elif isinstance(criterion, nn.Module):
            self.criterion = criterion

        else:
            raise ValueError(
                "Criterion must be a string ('mse' or 'l1') or an instance of nn.Module."
            )

        # store results path
        self.results_path = results_path

        # state of model for checkpointing
        self._internal_state = {"current_epoch": 0, "total_epochs": 0}

    def train(self, epochs: int = 100, checkpoint_epoch: int = None):
        """
        Train the model. Handles both training and testing based
        on the datasets passed to the __init__ method.

        Args:
            epochs (int, optional): The number of epochs to train for. Defaults to 100.
            checkpoint_epoch (int, optional): Frequency of checkpointing
        """

        self._internal_state["total_epochs"] = (
            self._internal_state["total_epochs"] + epochs
        )

        # create some storages for errors per epoch
        if not hasattr(self, "training_loss"):
            self.training_loss = torch.zeros(epochs, device=self.device)
            self.testing_loss = torch.zeros(epochs, device=self.device)
            self.training_accuracy = torch.zeros(epochs, device=self.device)
            self.testing_accuracy = torch.zeros(epochs, device=self.device)
        else:
            self.training_loss = F.pad(self.training_loss, (0, epochs))
            self.testing_loss = F.pad(self.testing_loss, (0, epochs))
            self.training_accuracy = F.pad(self.training_accuracy, (0, epochs))
            self.testing_accuracy = F.pad(self.testing_accuracy, (0, epochs))

        for epoch in tqdm(
            range(epochs),
            desc="Training...",
            total=self._internal_state["total_epochs"],
            initial=self._internal_state["current_epoch"],
            unit="epochs",
            leave=False,
        ):
            self.model.train()

            # store the loss values for the epoch
            epoch_train_loss = torch.zeros(
                len(self.train_dataloader), device=self.device
            )
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, batch in enumerate(
                tqdm(
                    self.train_dataloader,
                    desc=f"Epoch {self._internal_state['current_epoch']+1}...",
                    unit="steps",
                    total=len(self.train_dataloader),
                    leave=False,
                )
            ):
                # process the training batch
                loss, correct, total = self._run_step(batch)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epoch_train_loss[batch_idx] = loss.item()
                epoch_correct += correct
                epoch_total += total

            self.training_loss[self._internal_state["current_epoch"]] = (
                epoch_train_loss.mean()
            )
            self.training_accuracy[self._internal_state["current_epoch"]] = (
                epoch_correct / epoch_total
            )

            test_loss, test_correct, test_total = self.test(
                _epoch=self._internal_state["current_epoch"]
            )

            self.testing_loss[self._internal_state["current_epoch"]] = test_loss
            self.testing_accuracy[self._internal_state["current_epoch"]] = (
                test_correct / test_total
            )

            self._internal_state["current_epoch"] += 1

            if (
                self._internal_state["current_epoch"] == 0
                and (self._internal_state["current_epoch"]) % 1 == 0
            ):
                self.error_plot(path=self.results_path)
                self.accuracy_plot(path=self.results_path)

        self.training_loss = self.training_loss.cpu()
        self.testing_loss = self.testing_loss.cpu()

        print(f"{self._internal_state['total_epochs']} epochs complete!")

    def test(self, _epoch: int = None) -> tuple[torch.Tensor, int, int]:
        """
        Pass the testing dataset through the model. Automatically
        wraps all code with torch.no_grad()

        Args:
            _epoch (int, optional): the current epoch. Defaults to None.

        Returns:
            torch.Tensor: The average loss for the testing dataset
        """
        with torch.no_grad():

            epoch_test_loss = torch.zeros(len(self.test_dataloader), device=self.device)
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, batch in enumerate(
                tqdm(
                    self.test_dataloader,
                    desc=f"Epoch {_epoch+1}...",
                    unit="steps",
                    total=len(self.test_dataloader),
                    leave=False,
                )
            ):
                # process the training batch
                loss, correct, total = self._run_step(batch)

                epoch_test_loss[batch_idx] = loss.item()
                epoch_correct += correct
                epoch_total += total

        return epoch_test_loss.mean(), epoch_correct, epoch_total

    def _run_step(self, batch: _BatchType) -> tuple[torch.Tensor, int, int]:
        """
        Helper method to process a single batch of data. Avoids
        duplication, can be used for train and test, providing it
        is called within a torch.no_grad for testing.

        Args:
            batch (_BatchType): The batch of data to process

        Returns:
            torch.Tensor: loss value
        """

        titles, reviews, labels = batch

        titles = titles.to(self.device)
        reviews = reviews.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(titles, reviews)

        loss = self.criterion(outputs, labels)

        # accuracy calculation
        if outputs.ndim > 1 and outputs.size(1) > 1:
            # Multi-class: take argmax
            preds = torch.argmax(outputs, dim=1)
        else:
            # Binary: round sigmoid output
            preds = (outputs > 0.5).long()

        correct = (preds == labels).sum().item()
        total = labels.size(0)

        return loss, correct, total

    def save_model(self, path: Path | str) -> None:
        """
        Save the trained model.

        Note: If intermediate model training state is required,
        call `checkpoint` instead.

        Args:
            path (Path | str): Path to save the model to.
        """
        path = Path(path)

        if path.is_dir():
            path = path / "model.pt"

        torch.save(self.model.state_dict(), path)

    def save_checkpoint(self, path: Path | str) -> None:
        """
        Checkpoint a model, including model state, optimiser state,
        current epoch and all previous training and testing loss
        values.

        Args:
            path (Path | str): Path do save the model to.'
            epoch (int): store the current epoch of the checkpoint
        """
        path = Path(path)

        if path.is_dir():
            path = path / f"checkpoint_{self._internal_state['current_epoch']}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimiser.state_dict(),
            "current_epoch": self._internal_state.get("current_epoch", 0),
            "total_epochs": self._internal_state.get("total_epochs", 0),
            "training_loss": getattr(self, "training_loss", None),
            "testing_loss": getattr(self, "testing_loss", None),
            "training_accuracy": getattr(self, "training_accuracy", None),
            "testing_accuracy": getattr(self, "testing_accuracy", None),
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path | str) -> None:
        """
        Load a model checkpoint to resume training

        Args:
            path (Path | str): Path of the checkpoint
        """
        path = Path(path)

        checkpoint: dict = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optim_state_dict"])
        self._internal_state["current_epoch"] = checkpoint.get("current_epoch", 0)
        self._internal_state["total_epochs"] = checkpoint.get("total_epochs", 0)
        self.training_loss = checkpoint.get("training_loss", None)
        self.testing_loss = checkpoint.get("testing_loss", None)
        self.training_accuracy = checkpoint.get("training_accuracy", None)
        self.testing_accuracy = checkpoint.get("testing_accuracy", None)

        if self.training_loss is not None:
            self.training_loss.to(self.device)
            self.testing_loss.to(self.device)

    def error_plot(self, path: Path | str) -> None:
        """
        Create an error plot for training and testing loss

        Args:
            path (Path | str): Path to save the image to
        """
        fig, ax = plt.subplots()

        training_loss = self.training_loss.cpu().numpy()
        testing_loss = self.testing_loss.cpu().numpy()

        ax.plot(
            np.arange(1, self._internal_state["current_epoch"] + 1),
            training_loss[: self._internal_state["current_epoch"]],
            label="Train",
        )
        ax.plot(
            np.arange(1, self._internal_state["current_epoch"] + 1),
            testing_loss[: self._internal_state["current_epoch"]],
            label="Test",
        )
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel(self.criterion.__class__.__name__)
        ax.set_yscale("log")
        ax.set_title("Training Loss")

        fig.savefig(path / "error_plot.png")
        plt.close(fig)

    def accuracy_plot(self, path: Path | str) -> None:
        """
        Create an accuracy plot for training and testing accuracy

        Args:
            path (Path | str): Path to save the image to
        """
        fig, ax = plt.subplots()

        training_accuracy = self.training_accuracy.cpu().numpy() * 100
        testing_accuracy = self.testing_accuracy.cpu().numpy() * 100

        training_accuracy = training_accuracy.round(2)
        testing_accuracy = testing_accuracy.round(2)

        ax.plot(
            np.arange(1, self._internal_state["current_epoch"] + 1),
            training_accuracy[: self._internal_state["current_epoch"]],
            label="Train",
        )
        ax.plot(
            np.arange(1, self._internal_state["current_epoch"] + 1),
            testing_accuracy[: self._internal_state["current_epoch"]],
            label="Test",
        )
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy / %")
        ax.set_yscale("log")
        ax.set_title("Accuracy")

        fig.savefig(path / "accuracy_plot.png")
        plt.close(fig)

    @property
    def trained_model(self) -> nn.Module:
        return self.model
