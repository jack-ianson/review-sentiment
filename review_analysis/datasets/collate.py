import torch


def bow_collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    title_ids, review_ids, labels = zip(*batch)

    # pad the sequences to the max length in the batch
    max_title_len = max(len(t) for t in title_ids)
    max_review_len = max(len(r) for r in review_ids)

    padded_title_ids = torch.zeros((len(title_ids), max_title_len), dtype=torch.long)
    padded_review_ids = torch.zeros((len(review_ids), max_review_len), dtype=torch.long)

    for idx, (title_id, review_id) in enumerate(zip(title_ids, review_ids)):
        padded_title_ids[idx, : len(title_id)] = torch.tensor(
            title_id, dtype=torch.long
        )
        padded_review_ids[idx, : len(review_id)] = torch.tensor(
            review_id, dtype=torch.long
        )

    labels = [l[0] if isinstance(l, (list, tuple)) else l for l in labels]
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_title_ids, padded_review_ids, labels
