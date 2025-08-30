import argparse
from .train import train_model


def main():

    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")

    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.")

    args = parser.parse_args()

    train_model(data_root=args.data_root)
