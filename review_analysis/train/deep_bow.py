from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from .. import datasets
from ..modules import DeepBagOfWords
from ..trainer import ReviewsModelTrainer


def train_bag_of_words(
    data_root: str = None,
    results_path: str | Path = None,
    validation: bool = True,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: int = 0.001,
    cache_data: bool = False
):

    # load the data
    if not data_root:
        data_root = "data"

    results_path = Path(results_path)

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    training_data = datasets.load_dataset(f"{data_root}/train.csv", n=250000)

    if validation:
        testing_data = datasets.load_dataset(f"{data_root}/val.csv", n=50000)
    else:
        testing_data = datasets.load_dataset(f"{data_root}/test.csv", n=50000)

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
    tokeniser = datasets.Tokeniser(max_vocab_size=20000)

    tokeniser.fit_many(training_titles)
    tokeniser.fit_many(training_reviews)

    tokeniser.finalise_vocab()

    print(f"Vocabulary size: {len(tokeniser.word2idx)}")
    print(f"Total unique words: {tokeniser.total_word_count}")
    print(f"Fraction of vocab used: {tokeniser.fraction_words_covered*100:.2f}%")

    tokeniser.save(path=results_path)


    # create the datasets
    training_dataset = datasets.ReviewDataset(
        titles=training_titles,
        reviews=training_reviews,
        labels=training_labels,
        tokeniser=tokeniser,
        precompute_tokens=True
    )

    testing_dataset = datasets.ReviewDataset(
        titles=testing_titles,
        reviews=testing_reviews,
        labels=testing_labels,
        tokeniser=tokeniser,
        precompute_tokens=True
    )

    
    # create cache path
    if cache_data:
        cached_data_path = Path(f"{data_root}/cached_tokens")
        cached_data_path.mkdir()

        training_dataset.cache_tokenisation(f"{data_root}/cached_tokens/training_tokens.pt")
        training_dataset.cache_tokenisation(f"{data_root}/cached_tokens/{'validation' if validation else 'testing'}_tokens.pt")


    model = DeepBagOfWords(
        vocab_size=len(tokeniser.word2idx),
        embedding_dim=128,
        hidden_dims=[128, 256, 128],
        num_classes=2,
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if training_dataset.cache_tokenisation:
        print("Using cached tokenisation collate function.")
        collate_fn = datasets.bow_collate_cached_fn
    else:
        collate_fn = datasets.bow_collate_fn

    trainer = ReviewsModelTrainer(
        model=model,
        training_dataset=training_dataset,
        testing_dataset=testing_dataset,
        results_path=results_path,
        device=device,
        collate_fn=collate_fn,
        batch_size=batch_size,
        optimiser=optimizer,
        criterion=criterion,
    )
    

    trainer.train(epochs=epochs, checkpoint_interval=20)

    trainer.error_plot(path=results_path)
    trainer.accuracy_plot(path=results_path)

    # save the model
    trainer.save_model(path=results_path)
    trainer.save_checkpoint(path=results_path)

