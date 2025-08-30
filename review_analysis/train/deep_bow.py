from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import datasets
from ..modules import DeepBagOfWords


def train_bag_of_words(data_root: str = None, validation: bool = True):

    # load the data
    if not data_root:
        data_root = "data"

    training_data = datasets.load_dataset(f"{data_root}/train.csv", n=100000)

    if validation:
        testing_data = datasets.load_dataset(f"{data_root}/val.csv", n=20000)
    else:
        testing_data = datasets.load_dataset(f"{data_root}/test.csv", n=20000)

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
    tokeniser = datasets.Tokeniser(max_vocab_size=2500)

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
        batch_size=4,
        shuffle=True,
        collate_fn=datasets.bow_collate_fn,
    )
    test_dataloader = DataLoader(
        testing_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=datasets.bow_collate_fn,
    )

    model = DeepBagOfWords(
        vocal_size=len(tokeniser.word2idx),
        embedding_dim=64,
        hidden_dims=[128, 128],
        num_classes=2,
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dataloader, desc="Training", unit="batch", leave=False):
            title_inputs, review_inputs, labels = batch

            title_inputs = title_inputs.to(device)
            review_inputs = review_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(title_inputs, review_inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(
                test_dataloader, desc="Evaluating", unit="batch", leave=False
            ):
                title_inputs, review_inputs, labels = batch

                title_inputs = title_inputs.to(device)
                review_inputs = review_inputs.to(device)
                labels = labels.to(device)

                outputs = model(title_inputs, review_inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # print(
        #     f"{'Validation' if validation else 'Test'} Accuracy after epoch {epoch + 1}: {accuracy:.2f}%"
        # )
