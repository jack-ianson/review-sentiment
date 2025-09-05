from __future__ import annotations
import argparse

from .train.deep_bow import train_bag_of_words


def main():

    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")

    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.")
    parser.add_argument(
        "--results_path",
        type=str,
        help="Directory to save training results.",
        required=True,
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Use validation set during training. If False, use test set.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )

    args = parser.parse_args()

    train_bag_of_words(
        data_root=args.data_root,
        results_path=args.results_path,
        validation=args.validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
