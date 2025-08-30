from __future__ import annotations
import argparse

from .train.bow import train_bag_of_words


def main():

    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")

    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.")

    args = parser.parse_args()

    train_bag_of_words(data_root=args.data_root)
